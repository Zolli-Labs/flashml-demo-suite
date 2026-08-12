"""One trial of a hyperparameter search — the fan-out story.

FlashML expands the `sweep:` in flashml.yaml into one task per combination
and runs this file once per task, appending the axes as flags:

    python /work/inputs/code/trial.py --epochs 8 --batch-size 128 \
        --lr 0.3 --hidden 16

Every trial gets the WHOLE dataset (`split: replica`, inferred for a
non-federated job), trains on the training shards, and scores the held-out
shard. The number it writes into metrics.json under `accuracy` is what the
job's `reduce: {kind: rank}` orders the trials by, and what the
`validators: {keys: [accuracy]}` declaration insists on — a trial whose
metrics.json is intact but carries no `accuracy` fails its attempt and is
retried elsewhere, rather than silently ranking below everything.

WHY THIS IS AN HPO JOB AND NOT A FAN-OUT
----------------------------------------
The router classifies a job from what its config declares, and the answer is
shown in the console with its evidence. A `sweep:` alone would already read
as HPO; the `reduce: rank` is what makes the evidence say *"the trials are
being selected between, which is what makes this a search rather than a
fan-out"*. The distinction is not cosmetic — kind decides which venues the
work is offered to before price is looked at at all.

TRIALS CHECKPOINT TOO
---------------------
Checkpointing is on for every task, a sweep's trials included. This one is
seconds long and genuinely has little state worth saving, but it writes
`ckpt/step-<N>.json` per epoch and reads `/work/inputs/resume.json` on start
anyway, because the convention is the same one a six-hour task depends on
and a demo suite that skipped it here would teach the wrong lesson.

Accuracy is not the point. Six trials over a synthetic 4k-sample problem
will land within a few points of each other; what is being demonstrated is
that six machines each got a different combination, each committed a score,
and one of them was named the winner.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

WORK = os.environ.get("FLASHML_WORK_DIR", "/work")

#: The shard reserved for scoring. `split: replica` hands every trial the
#: whole listing, so the trial itself decides what it trains on and what it
#: is graded by — and both decisions have to be the same in every trial, or
#: the ranking compares numbers measured against different data.
HOLDOUT_MARKER = "holdout"

CHECKPOINT_SCHEMA = "flashml.demo/hpo-checkpoint/1"


def init_params(seed: int, features: int, hidden: int, classes: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=features**-0.5, size=(features, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(scale=hidden**-0.5, size=(hidden, classes)),
        "b2": np.zeros(classes),
    }


def forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["W1"] + params["b1"])
    logits = hidden @ params["W2"] + params["b2"]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return hidden, exp / exp.sum(axis=1, keepdims=True)


def loss_and_grads(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    n = len(x)
    hidden, probs = forward(params, x)
    loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean())
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dhidden = (dlogits @ params["W2"].T) * (1.0 - hidden * hidden)
    return loss, {
        "W1": x.T @ dhidden,
        "b1": dhidden.sum(axis=0),
        "W2": hidden.T @ dlogits,
        "b2": dlogits.sum(axis=0),
    }


def score(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    _, probs = forward(params, x)
    loss = float(-np.log(np.clip(probs[np.arange(len(x)), y], 1e-12, None)).mean())
    return float((probs.argmax(axis=1) == y).mean()), loss


def encode(params: dict) -> dict:
    return {
        name: {"shape": list(value.shape), "data": [float(v) for v in value.ravel()]}
        for name, value in params.items()
    }


def decode(blob: dict) -> dict:
    return {
        name: np.asarray(entry["data"], dtype=float).reshape(entry["shape"])
        for name, entry in blob.items()
    }


def read_npz(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for path in paths:
        with np.load(path) as bundle:
            xs.append(bundle["X"] if "X" in bundle else bundle["x"])
            ys.append(bundle["y"])
    return np.concatenate(xs).astype(float), np.concatenate(ys).astype(np.int64)


def load_split(data_dir: Path) -> tuple[tuple, tuple, list[str]]:
    """(train, holdout, shard names). Both halves, from one listing."""
    if not data_dir.is_dir():
        raise SystemExit(
            f"{data_dir} does not exist. The host agent fetches declared "
            f"datasets before the task starts, so this means flashml.yaml "
            f"declares no `datasets:` — or declares a name other than 'demo'."
        )
    everything = sorted(data_dir.rglob("*.npz"))
    train_files = [p for p in everything
                   if HOLDOUT_MARKER not in str(p.relative_to(data_dir))]
    holdout_files = [p for p in everything
                     if HOLDOUT_MARKER in str(p.relative_to(data_dir))]
    if not train_files or not holdout_files:
        raise SystemExit(
            f"{data_dir} does not hold both halves of the split: found "
            f"{[p.name for p in everything]}. A trial that scored itself on "
            f"its own training data would rank the sweep by the wrong number."
        )
    return read_npz(train_files), read_npz(holdout_files), [p.name for p in everything]


def write_json_atomically(path: Path, document: dict) -> None:
    """Temp file in the same directory, then os.replace.

    The temp name deliberately does not match `step-*.json`: the relay ships
    what it finds, and a half-written file under that glob is a checkpoint
    nothing can resume from.
    """
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    tmp = path.with_name(f".{path.name}.partial")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=f"{WORK}/data/demo")
    parser.add_argument("--out", default=f"{WORK}/out")
    parser.add_argument("--resume", default=f"{WORK}/inputs/resume.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260812)
    # The sweep axes. FlashML appends `--lr <value>` and `--hidden <value>`
    # to this command, one combination per task; the names here must match
    # the keys under `sweep:` exactly.
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    out = Path(args.out)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (x, y), (hold_x, hold_y), shard_names = load_split(Path(args.data))
    features = int(x.shape[1])
    steps_per_epoch = max(1, len(x) // args.batch_size)
    run = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden": int(args.hidden),
        "batch_size": int(args.batch_size),
        "features": features,
    }

    resume = Path(args.resume)
    start_epoch = 0
    params = init_params(run["seed"], features, run["hidden"])
    if resume.exists():
        # Present and readable means the recovery path is live. Present and
        # written under different hyperparameters means this trial's weights
        # belong to a different point in the search — refuse rather than
        # blend two trials into one number the ranking will believe.
        staged = json.loads(resume.read_text())
        if staged.get("run") != run:
            raise SystemExit(
                f"{resume} was written by a trial with different settings "
                f"({staged.get('run')}) — refusing to resume this one from it"
            )
        params = decode(staged["params"])
        start_epoch = int(staged["progress"]["epoch"])
        print(f"RESUMED at epoch {start_epoch}/{run['epochs']}", flush=True)

    for epoch in range(start_epoch, run["epochs"]):
        order = np.random.default_rng([run["seed"], epoch]).permutation(len(x))
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size : (step + 1) * args.batch_size]
            _, grads = loss_and_grads(params, x[batch], y[batch])
            for name in params:
                params[name] -= run["lr"] * grads[name]
        step_reached = (epoch + 1) * steps_per_epoch
        write_json_atomically(
            ckpt_dir / f"step-{step_reached}.json",
            {
                "schema": CHECKPOINT_SCHEMA,
                "run": run,
                "progress": {"epoch": epoch + 1, "step": step_reached},
                "params": encode(params),
            },
        )

    holdout_accuracy, holdout_loss = score(params, hold_x, hold_y)
    print(
        f"lr={run['lr']} hidden={run['hidden']}: holdout accuracy "
        f"{holdout_accuracy:.4f} (loss {holdout_loss:.4f}) over {len(hold_y)} "
        f"held-out samples",
        flush=True,
    )

    write_json_atomically(
        out / "metrics.json",
        {
            "workload": "demo-suite/hpo",
            # THE ranked key. `validators: {keys: [accuracy]}` fails an
            # attempt that omits it, and `reduce: {kind: rank, metric:
            # accuracy}` orders the trials by it. Spell it exactly.
            "accuracy": holdout_accuracy,
            "loss": holdout_loss,
            "lr": run["lr"],
            "hidden": run["hidden"],
            "epochs": run["epochs"],
            "resumed": start_epoch > 0,
            "train_samples": int(len(x)),
            "holdout_samples": int(len(hold_y)),
            "shards": shard_names,
            "duration_seconds": round(time.monotonic() - started, 3),
            "host": os.environ.get("HOSTNAME", "unknown"),
        },
    )
    write_json_atomically(out / "model.json", {"params": encode(params), "run": run})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
