"""One checkpointed, resumable task — the fault-tolerance story, minimal.

FlashML runs this once:

    python /work/inputs/code/train.py --epochs 10 --lr 0.5 --hidden 8

and the only thing that makes it survivable is the checkpoint convention.
Two halves, both fixed, neither configurable:

    write  /work/out/ckpt/step-<N>.json   your resumable state. THAT
                                          directory, THAT filename, N the
                                          integer step reached, one file per
                                          checkpoint, written ATOMICALLY.
                                          The agent relays each new file off
                                          the machine the moment it appears.

    read   /work/inputs/resume.json       the last checkpoint the previous
                                          attempt committed, staged before
                                          this process starts. ABSENT on a
                                          first attempt — "no such file"
                                          means "start from scratch", not an
                                          error.

There is no `checkpoint:` key in flashml.yaml. The relay is on for every
task; the only question is whether your code gives it anything to watch. A
task that writes state anywhere else looks completely healthy and has no
fault tolerance at all: the machine dies, the retry starts from step 0, and
nothing tells you.

`flashruntime.torch.checkpoint()` does NOT satisfy this and is the closest
thing to a trap in the system, because it gets the directory right — its
root defaults to `<out>/ckpt` — and then writes `step-000100/` DIRECTORIES
of model.pt. The relay globs `step-*.json` FILES. Nothing it writes is ever
shipped.

RESUME MUST NOT CHANGE THE ANSWER
---------------------------------
The batch order for epoch *e* is a pure function of `(seed, e)` — not a
serialised PRNG state — so an attempt that starts at epoch 7 draws exactly
the permutation the dead attempt would have drawn. That is the one design
decision that makes a resumed run equal to an uninterrupted one, and it is
worth more than any amount of state stuffed into the checkpoint.

This is a plumbing demo, not a benchmark. The model is a one-hidden-layer
MLP in about forty lines of numpy, trained for seconds. The number to watch
is `resumed_from_step` in metrics.json, not `accuracy`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

#: /work is where FlashML mounts a task's world. FLASHML_WORK_DIR exists so
#: the whole thing can be rehearsed on a laptop without root; FlashML never
#: sets it. The three paths below are FLAGS with /work defaults rather than
#: constants, because a trusted-tier host rewrites `/work`-prefixed argv
#: tokens onto its real workdir — a path this file computed for itself would
#: miss that rewrite.
WORK = os.environ.get("FLASHML_WORK_DIR", "/work")

#: Which shard is the held-out one. The dataset ships several training
#: shards plus a holdout, and `split: replica` (inferred for a non-federated
#: job) hands EVERY task the whole listing — so the holdout arrives here too
#: and has to be kept out of training by name. `examples/demo-suite/evaluate`
#: is the workload that scores against it.
HOLDOUT_MARKER = "holdout"

CHECKPOINT_SCHEMA = "flashml.demo/checkpoint/1"
MODEL_SCHEMA = "flashml.demo/model/1"


# --- the model: forty lines of numpy, deliberately -------------------------


def init_params(seed: int, features: int, hidden: int, classes: int = 2) -> dict:
    """The initial weights, from a seed alone.

    A pure function of the seed so that a fresh start and a resumed start
    can be compared, and so `evaluate/` can rebuild the same starting point
    without being handed a file.
    """
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
    logits = logits - logits.max(axis=1, keepdims=True)  # softmax, stably
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


def accuracy(params: dict, x: np.ndarray, y: np.ndarray) -> float:
    _, probs = forward(params, x)
    return float((probs.argmax(axis=1) == y).mean())


# --- the JSON encoding, shared by every workload in this suite -------------


def encode(params: dict) -> dict:
    """``{"W1": {"shape": [...], "data": [...]}, ...}``.

    The same shape the federated delta protocol uses, on purpose: one
    encoding across the suite means `evaluate/` can read a model this file
    wrote and a model the federated run produced without knowing which.
    """
    return {
        name: {"shape": list(value.shape), "data": [float(v) for v in value.ravel()]}
        for name, value in params.items()
    }


def decode(blob: dict) -> dict:
    return {
        name: np.asarray(entry["data"], dtype=float).reshape(entry["shape"])
        for name, entry in blob.items()
    }


# --- the data --------------------------------------------------------------


def load_training_shards(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every `.npz` the agent placed here except the holdout.

    A missing directory is a real failure and not an empty epoch: it means
    the fetch did not happen, and training on nothing is worse than not
    starting, because it produces a model and a metrics file that look fine.
    """
    if not data_dir.is_dir():
        raise SystemExit(
            f"{data_dir} does not exist. The host agent fetches declared "
            f"datasets before the task starts, so this means flashml.yaml "
            f"declares no `datasets:` — or declares a name other than 'demo'."
        )
    shards = sorted(p for p in data_dir.rglob("*.npz")
                    if HOLDOUT_MARKER not in str(p.relative_to(data_dir)))
    if not shards:
        raise SystemExit(
            f"{data_dir} holds no training shards (found: "
            f"{sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob('*.npz'))}). Every file "
            f"whose name contains {HOLDOUT_MARKER!r} is reserved for the "
            f"evaluate workload."
        )
    xs, ys = [], []
    for shard in shards:
        with np.load(shard) as bundle:
            xs.append(bundle["X"] if "X" in bundle else bundle["x"])
            ys.append(bundle["y"])
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys).astype(np.int64)
    return x, y, [s.name for s in shards]


# --- the checkpoint convention ---------------------------------------------


def write_json_atomically(path: Path, document: dict) -> bytes:
    """Write `document` so that no reader ever sees it half-written.

    The relay ships whatever it finds and can find the file a millisecond
    after it is created, so a plain `write_text` publishes torn JSON often
    enough to matter — and the run it kills is the NEXT one, which makes it a
    bug discovered long after it was written.

    Temp file in the SAME directory (rename is atomic only within one
    filesystem, so /tmp is not an option) whose name does not itself match
    `step-*.json` (or the relay would ship the half-written file), then
    `os.replace`. The two fsyncs make it survive a power cut rather than only
    a killed process.
    """
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    tmp = path.with_name(f".{path.name}.partial")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return data


def load_resume(path: Path, run: dict) -> dict | None:
    """The staged checkpoint, or None on a first attempt.

    An absent file is the normal case. A file that is PRESENT and broken is
    the opposite: it means the recovery path is live and the bytes it
    delivered cannot be trusted, so this raises rather than quietly starting
    over — a silent restart would look like a successful recovery while
    throwing away every minute the dead attempt spent.
    """
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if document.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit(
            f"{path} is not a checkpoint this trainer wrote (schema "
            f"{document.get('schema')!r}) — refusing to resume from it"
        )
    stale = [k for k, v in run.items() if document["run"].get(k) != v]
    if stale:
        raise SystemExit(
            f"{path} was written by a run with different {stale} — refusing to "
            f"continue one run's weights under another run's schedule"
        )
    return document


# --- the run ---------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=f"{WORK}/data/demo")
    parser.add_argument("--out", default=f"{WORK}/out")
    parser.add_argument(
        "--resume",
        default=f"{WORK}/inputs/resume.json",
        help="checkpoint staged by the agent for a resumed attempt; absent on "
        "a first attempt, which is not an error",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    out = Path(args.out)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    x, y, shard_names = load_training_shards(Path(args.data))
    features = int(x.shape[1])
    steps_per_epoch = max(1, len(x) // args.batch_size)

    #: Every number that decides what this run computes. Written into each
    #: checkpoint and compared on resume — continuing a 10-epoch run's
    #: weights under a 4-epoch schedule produces a model no reported number
    #: describes.
    run = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden": int(args.hidden),
        "batch_size": int(args.batch_size),
        "features": features,
        "samples": int(len(x)),
        "steps_per_epoch": steps_per_epoch,
        "steps_total": steps_per_epoch * int(args.epochs),
    }

    resumed = load_resume(Path(args.resume), run)
    if resumed is None:
        params = init_params(run["seed"], features, run["hidden"])
        start_epoch = 0
        resumed_from = None
        print(
            f"fresh start — {run['samples']} samples from {len(shard_names)} "
            f"shard(s), {run['epochs']} epochs x {steps_per_epoch} steps",
            flush=True,
        )
    else:
        params = decode(resumed["params"])
        start_epoch = int(resumed["progress"]["epoch"])
        resumed_from = int(resumed["progress"]["step"])
        print(
            f"RESUMED at step {resumed_from} (epoch {start_epoch}/"
            f"{run['epochs']}) — that much work survived the machine dying",
            flush=True,
        )

    written: list[str] = []
    loss = float("nan")
    train_accuracy = float("nan")
    for epoch in range(start_epoch, run["epochs"]):
        # Drawn from (seed, epoch) and nothing else. This is what makes a
        # resumed run identical to an uninterrupted one.
        order = np.random.default_rng([run["seed"], epoch]).permutation(len(x))
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size : (step + 1) * args.batch_size]
            loss, grads = loss_and_grads(params, x[batch], y[batch])
            for name in params:
                params[name] -= run["lr"] * grads[name]
        train_accuracy = accuracy(params, x, y)

        step_reached = (epoch + 1) * steps_per_epoch
        # THE convention. Directory, filename, atomic write.
        target = ckpt_dir / f"step-{step_reached}.json"
        write_json_atomically(
            target,
            {
                "schema": CHECKPOINT_SCHEMA,
                "run": run,
                "progress": {
                    "epoch": epoch + 1,
                    "step": step_reached,
                    "epochs_total": run["epochs"],
                    "steps_total": run["steps_total"],
                },
                # Recorded on every checkpoint rather than only the first one
                # after a resume: the final metrics must be able to say where
                # this attempt picked up, however many checkpoints later.
                "resumed_from_step": resumed_from,
                "params": encode(params),
                "metrics": {"loss": loss, "train_accuracy": train_accuracy},
            },
        )
        written.append(f"ckpt/{target.name}")
        print(
            f"  epoch {epoch + 1:>2}/{run['epochs']}  step {step_reached:>4}/"
            f"{run['steps_total']}  loss {loss:.4f}  acc {train_accuracy:.3f}"
            f"  -> {target.name}",
            flush=True,
        )

    model_bytes = write_json_atomically(
        out / "model.json",
        {
            "schema": MODEL_SCHEMA,
            "architecture": {
                "kind": "mlp-tanh-softmax",
                "features": features,
                "hidden": run["hidden"],
                "classes": 2,
            },
            "params": encode(params),
            "trained_on": {"shards": shard_names, "samples": run["samples"]},
            "trained_for": {"seed": run["seed"], "epochs": run["epochs"]},
        },
    )

    write_json_atomically(
        out / "metrics.json",
        {
            "workload": "demo-suite/train",
            "loss": loss,
            "train_accuracy": train_accuracy,
            # No holdout number here on purpose: scoring the held-out shard is
            # the evaluate workload's job, and a trainer that grades its own
            # homework is how a demo ends up with two disagreeing accuracies.
            "epochs": run["epochs"],
            "epochs_executed": run["epochs"] - start_epoch,
            "resumed": resumed is not None,
            "resumed_from_step": resumed_from,
            "checkpoints": written,
            "shards": shard_names,
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "duration_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(
        f"done in {time.monotonic() - started:.1f}s — {len(written)} "
        f"checkpoint(s), train accuracy {train_accuracy:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
