"""Score a trained model against the held-out shard — the evaluation story.

FlashML runs this once:

    python /work/inputs/code/evaluate.py --hidden 8

It reads the holdout shard the host agent fetched into /work/data/demo/,
loads the model committed next to this file, and writes one number the job's
`validators:` insists on:

    write  /work/out/metrics.json   {"accuracy": <0..1>, ...}

WHERE THE MODEL COMES FROM, AND THE GAP THAT MAKES THIS AWKWARD
---------------------------------------------------------------
A job cannot name another job's artifact. `flashml.yaml` has `datasets:`
(origins a host fetches) and `local_inputs:` (labels a host lends) and no
third thing, so "score the model job 8821 produced" is not expressible
today — the console's own classifier says as much when it labels a job
EVALUATION: *"nothing names the artifact under test"*.

So the model travels the way everything else in a job travels: **inside the
repo**. Run `../train/`, download its `model.json` artifact from the
console, drop it beside this file, submit. The repo tarball is staged at
/work/inputs/code/, which is where `--model` looks by default.

With no model.json present this scores the untrained initialisation instead
— the same seeded weights `../train/` begins from — and says so in
`model_source`. That is the floor the trained model has to beat, and having
it means this workload runs the day the suite is submitted rather than only
after a training job has finished.

WHY THIS ONE HAS NO CHECKPOINTS
-------------------------------
Preflight warns `no-checkpoint` here, and the warning is correct rather than
an oversight. This task is one forward pass over a few hundred rows; there
is no progress to save, and a machine that dies simply causes the whole
thing to run again somewhere else at no cost. The warning exists because the
opposite mistake — a six-hour job with nothing resumable — is invisible
until a machine dies, so it is worth being told about a job that has no
fault tolerance even when the job does not need any. `../train/` is the
workload that answers it.

Accuracy is not the point. This is a plumbing demo: what it demonstrates is
that a task can be handed one part of a dataset, produce a validated number,
and have that number checked at commit time rather than believed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

WORK = os.environ.get("FLASHML_WORK_DIR", "/work")

#: The repo, as staged inside the container: /work/inputs/code/. Derived
#: from this file rather than hardcoded, so the same command works when the
#: directory is run straight out of a checkout.
CODE_DIR = Path(__file__).resolve().parent

#: The shard this workload exists to read. Everything else in the directory
#: is training data that the model has already seen.
HOLDOUT_MARKER = "holdout"

#: Must match ../train/train.py's --seed default. Only used when no model is
#: supplied, to produce the untrained floor.
INIT_SEED = 20260812


def init_params(seed: int, features: int, hidden: int, classes: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=features**-0.5, size=(features, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(scale=hidden**-0.5, size=(hidden, classes)),
        "b2": np.zeros(classes),
    }


def decode(blob: dict) -> dict:
    return {
        name: np.asarray(entry["data"], dtype=float).reshape(entry["shape"])
        for name, entry in blob.items()
    }


def probabilities(params: dict, x: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ params["W1"] + params["b1"])
    logits = hidden @ params["W2"] + params["b2"]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def load_holdout(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The held-out shard(s), and nothing else.

    An empty match is a failure and not an empty score: a task that reported
    an accuracy over zero rows would publish a number with no data behind it,
    and `validators: {keys: [accuracy]}` would happily accept it.
    """
    if not data_dir.is_dir():
        raise SystemExit(
            f"{data_dir} does not exist. The host agent fetches declared "
            f"datasets before the task starts, so this means flashml.yaml "
            f"declares no `datasets:` — or declares a name other than 'demo'."
        )
    files = sorted(p for p in data_dir.rglob("*.npz")
                   if HOLDOUT_MARKER in str(p.relative_to(data_dir)))
    if not files:
        raise SystemExit(
            f"no file in {data_dir} has {HOLDOUT_MARKER!r} in its name — found "
            f"{sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob('*.npz'))}. This workload "
            f"scores held-out data and refuses to score training data instead."
        )
    xs, ys = [], []
    for path in files:
        with np.load(path) as bundle:
            xs.append(bundle["X"] if "X" in bundle else bundle["x"])
            ys.append(bundle["y"])
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys).astype(np.int64)
    return x, y, [p.name for p in files]


def load_model(path: Path, features: int, hidden: int) -> tuple[dict, str, str | None]:
    """(params, where it came from, sha256 of the file if there was one)."""
    if not path.is_file():
        print(
            f"no model at {path} — scoring the untrained initialisation "
            f"instead. Drop ../train/'s model.json artifact beside "
            f"evaluate.py to score a trained one.",
            flush=True,
        )
        return init_params(INIT_SEED, features, hidden), "untrained-initialisation", None
    raw = path.read_bytes()
    document = json.loads(raw)
    params = decode(document["params"])
    if params["W1"].shape[0] != features:
        raise SystemExit(
            f"{path} was trained on {params['W1'].shape[0]} features and this "
            f"holdout has {features} — refusing to report an accuracy for a "
            f"model that cannot have been trained on this dataset"
        )
    return params, path.name, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=f"{WORK}/data/demo")
    parser.add_argument("--out", default=f"{WORK}/out")
    parser.add_argument(
        "--model",
        default=str(CODE_DIR / "model.json"),
        help="the model to score, committed inside the repo (a job cannot "
        "name another job's artifact today)",
    )
    parser.add_argument("--hidden", type=int, default=8)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x, y, files = load_holdout(Path(args.data))
    params, source, digest = load_model(Path(args.model), int(x.shape[1]), args.hidden)

    probs = probabilities(params, x)
    predicted = probs.argmax(axis=1)
    accuracy = float((predicted == y).mean())
    loss = float(-np.log(np.clip(probs[np.arange(len(x)), y], 1e-12, None)).mean())
    positives = int((predicted == 1).sum())

    print(
        f"{source}: accuracy {accuracy:.4f}, loss {loss:.4f} over {len(y)} "
        f"held-out samples from {', '.join(files)}",
        flush=True,
    )

    (out / "metrics.json").write_text(
        json.dumps(
            {
                "workload": "demo-suite/evaluate",
                # The key `validators: {keys: [accuracy]}` requires. A commit
                # missing it fails the attempt and requeues the task, rather
                # than succeeding with nothing to say.
                "accuracy": accuracy,
                "loss": loss,
                "samples": int(len(y)),
                "predicted_positive": positives,
                "labelled_positive": int((y == 1).sum()),
                "model_source": source,
                "model_sha256": digest,
                "holdout_files": files,
                "host": os.environ.get("HOSTNAME", "unknown"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
