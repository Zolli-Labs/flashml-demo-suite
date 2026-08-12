"""One machine's share of one federated round — the rounds story.

FlashML runs this file once per chunk per round:

    python /work/inputs/code/train.py --hidden 8 --epochs 4 \
        --round R --num-shards K --shard N

and requires three things of it:

    read   /work/inputs/weights.json   the round's starting weights. ABSENT
                                       on round 0 — that absence IS the
                                       signal to use your own
                                       initialisation, not an error.
    write  /work/out/delta.json        how you changed the weights:
                                       {"<param>": {"shape": [...],
                                       "data": [...]}}. On round 0, where
                                       you were given nothing, that is the
                                       trained weights themselves.
    write  /work/out/metrics.json      at least {"chunks_done": [...],
                                       "loss": <number>, "samples": <int>}.

`chunks_done` is the field people miss and it fails SILENTLY: it is the list
of chunk ids this task finished, and the platform credits the contribution
by it, intersected with the ids it can prove it handed this slot. Omit it
and the report is worth nothing — the round does not error, it waits out its
timeout and combines an empty set.

THIS IS NOT DDP
---------------
DistributedDataParallel needs every rank to reach every other rank on each
backward pass. Task containers run with `--network none`, because a
volunteer is lending you a machine and not an open socket. There are no
peers to reach and no rank 0 to rendezvous with. What runs instead is
federated averaging: weights cross the network once per round instead of
once per batch, and a laptop that closes its lid misses a round rather than
breaking the job.

DO NOT SLICE THE DATA AGAIN
---------------------------
`--shard` and `--num-shards` still arrive and still mean what they meant —
they identify this chunk so `chunks_done` can credit it. They are NOT how
data is selected here. `mode: federated` infers `split: shard`, so the host
agent already fetched exactly this task's files. Striding on top of that
would train each machine on a fraction of a fraction, and every round would
silently cover a fraction of the pass it claimed.

WHY THIS ONE HAS NO CHECKPOINTS
-------------------------------
Preflight warns `no-checkpoint` on this file, and the warning is correct
rather than an oversight — read it in the console and then read this
paragraph. A round is a few seconds of local training over one chunk. A
machine that dies mid-round loses that chunk, and the next round covers it,
because coverage is what closes a round rather than a headcount. The unit of
fault tolerance here is the round, not the epoch, and there is no state
between rounds worth relaying: the weights this task started from came from
the coordinator and are already durable there. `../train/` is the workload
that demonstrates the convention, and a longer federated task — minutes of
local work per round rather than seconds — should follow it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("FLASHML_WORK_DIR", "/work"))
DATA_DIR = WORK / "data" / "demo"
WEIGHTS_IN = WORK / "inputs" / "weights.json"
OUT_DIR = WORK / "out"

#: Shared by every machine and deliberately NOT derived from --shard.
#: Averaging N independently-initialised networks gives the mean of N
#: unrelated models, which is worse than any of them and looks like
#: "federated learning does not work" rather than like a bug. Round 0 must
#: start from byte-identical weights everywhere.
INIT_SEED = 12345

#: The held-out shard, reserved for `../evaluate/`. Under `split: shard` the
#: whole listing is cut across the fleet, so whichever machine is handed the
#: holdout must not train on it — otherwise the one number the suite reports
#: as held out was seen by the model.
HOLDOUT_MARKER = "holdout"


def init_params(features: int, hidden: int, classes: int = 2) -> dict:
    rng = np.random.default_rng(INIT_SEED)
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


# --- the encoding the platform averages ------------------------------------


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


def subtract(new: dict, base: dict) -> dict:
    """new - base, in the platform's encoding.

    A flat loss across rounds is the failure to watch for: it means every
    machine trained and nothing was combined, and the usual cause is
    delta.json holding whole weights instead of the change from round 1 on.
    """
    return {
        name: {
            "shape": list(new[name]["shape"]),
            "data": [a - b for a, b in zip(new[name]["data"], base[name]["data"])],
        }
        for name in new
    }


def load_slice() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Exactly the files the agent placed here, minus the holdout.

    No striding and no filtering by --shard: the slicing already happened at
    submit time, against the pinned manifest.
    """
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"{DATA_DIR} does not exist. The host agent fetches declared "
            f"datasets before the sandbox closes, so this means the job "
            f"declared no `datasets:` — or a name other than 'demo'."
        )
    present = sorted(DATA_DIR.glob("*.npz"))
    shards = [p for p in present if HOLDOUT_MARKER not in p.name]
    if not shards:
        # Loud on purpose. Reporting a chunk this task did not train would
        # credit coverage for work that never happened, and the round would
        # close believing a slice of the pass was covered when it was not.
        raise SystemExit(
            f"this slice holds nothing to train on: {[p.name for p in present]}. "
            f"Every file whose name contains {HOLDOUT_MARKER!r} is reserved "
            f"for the evaluate workload."
        )
    xs, ys = [], []
    for shard in shards:
        with np.load(shard) as bundle:
            xs.append(bundle["x"])
            ys.append(bundle["y"])
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys).astype(np.int64)
    return x, y, [s.name for s in shards]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Supplied by FlashML on every federated task. --shard is the only one
    # that differs between machines within a round.
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    # Yours; whatever is under `args:` in flashml.yaml lands here.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=8)
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, names = load_slice()
    params = init_params(int(x.shape[1]), args.hidden)

    # Round 0 has no weights.json, and that absence is the signal to start
    # from the shared initialisation rather than an error to retry.
    base = None
    if WEIGHTS_IN.is_file():
        base = json.loads(WEIGHTS_IN.read_text())
        params = decode(base)
        print(f"round {args.round}: resumed from the averaged weights", flush=True)
    else:
        print(f"round {args.round}: no weights.json — initialising", flush=True)

    # Full-batch over this machine's own slice, a handful of times. Local
    # training is deliberately short: the interesting thing is what happens
    # between rounds, not inside one.
    loss = float("nan")
    for _ in range(args.epochs):
        loss, grads = loss_and_grads(params, x, y)
        for name in params:
            params[name] -= args.lr * grads[name]

    print(
        f"chunk {args.shard}/{args.num_shards}: {len(names)} shard(s) "
        f"({', '.join(names)}), {len(x)} samples, loss {loss:.4f}",
        flush=True,
    )

    trained = encode(params)
    (OUT_DIR / "delta.json").write_text(
        json.dumps(trained if base is None else subtract(trained, base))
    )
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                # THE load-bearing field. One task trains one chunk here,
                # hence the single id. Report only what was trained: ids this
                # task was not handed are discarded, not trusted.
                "chunks_done": [args.shard],
                # Reported for the job view and NOT used as the averaging
                # weight — that is the verified chunk count. A machine that
                # set its own weight would set its own influence over
                # everyone's model.
                "samples": int(len(x)),
                "loss": loss,
                "round": args.round,
                "shard": args.shard,
                # The only place the slice is visible after the fact, which
                # is exactly when a round looks wrong.
                "shard_files": names,
                "host": os.environ.get("HOSTNAME", "unknown"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
