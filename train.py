"""One checkpointed, resumable task that runs on a real GPU — the routing story.

FlashML runs this once:

    python /work/inputs/code/train.py --epochs 8 --lr 0.5 --hidden 64 \
        --batch-size 128

Two things make it worth having next to `../train/`, which is the same idea
in numpy on a CPU.

THE CARD IS THE DECLARATION
---------------------------
`resources: {gpus: 1}` in flashml.yaml, on a job with no sweep and no
partition, is the only way a repo can tell the control plane "this is
training". The console then classifies it TRAINING instead of COMMAND, and
a venue with no card is refused *by name and by reason* before any price is
computed. Nothing in this file causes that — but this file is what makes the
declaration true, and a "GPU job" that quietly ran on a CPU would make the
whole demonstration a lie told in a log.

So the device is checked, not assumed. If CUDA is not there, this stops with
a non-zero exit and says which machine it was standing on. `--allow-cpu`
exists for rehearsing the data, checkpoint and resume paths on a laptop; it
is off by default and prints a banner nobody will mistake for a normal run.

THE CHECKPOINT CONVENTION, UNCHANGED BY THE HARDWARE
----------------------------------------------------
Two halves, both fixed, neither configurable, and identical to every other
workload in this suite:

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
task; the only question is whether your code gives it anything to watch.

`flashruntime.torch.checkpoint()` does NOT satisfy this, and on a torch
workload it is the obvious thing to reach for, which is exactly why it is
named here. It gets the directory right — its root defaults to `<out>/ckpt`
— and then writes `step-000100/` DIRECTORIES of model.pt. The relay globs
`step-*.json` FILES. Nothing it writes is ever shipped, nothing raises, and
the first sign of trouble is a retry that starts from zero. State goes to
JSON here for that reason and no other.

RESUME MUST NOT CHANGE THE ANSWER
---------------------------------
The batch order for epoch *e* is a pure function of `(seed, e)` — not a
serialised PRNG state, and not torch's global RNG — so an attempt that
starts at epoch 5 draws exactly the permutation the dead attempt would have
drawn. The initial weights are drawn the same way, by numpy from the seed
alone, which also means this job starts from the same point `../train/` and
`../evaluate/` do.

That makes the *work* exactly reproducible. It does not make the resulting
floats bit-identical the way `../train/` can claim on a CPU: cuBLAS is free
to reduce in a different order than a CPU kernel, and we have not measured
run-to-run bitwise equality on any card. So this file claims what it has:
the same data, in the same order, from the same starting weights.

NOT A BENCHMARK
---------------
A one-hidden-layer MLP over a few thousand synthetic rows, eight epochs,
seconds of compute. Nothing here is optimised and no number it prints is a
claim about a card's speed or the model's quality. It is sized so the demo
finishes while someone is watching it, and so nobody pays for an hour of
rented GPU to see a routing decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

#: /work is where FlashML mounts a task's world. FLASHML_WORK_DIR exists so
#: the whole thing can be rehearsed on a laptop without root; FlashML never
#: sets it for a sandboxed task. The three paths below are FLAGS with /work
#: defaults rather than constants, because a trusted-tier host rewrites
#: `/work`-prefixed argv tokens onto its real workdir — a path this file
#: computed for itself would miss that rewrite. That tier is the one a rented
#: pod uses, so for this workload it is the common case rather than a corner.
WORK = os.environ.get("FLASHML_WORK_DIR", "/work")

#: Which shard is the held-out one, matched against each file's path
#: RELATIVE TO THE DATA DIRECTORY and never against its name. The dataset
#: lands as `train/*.npz` and `holdout/eval.npz` — one level down — and
#: `holdout/eval.npz` has the name `eval.npz`, so a name-based check trains
#: on the evaluation shard and reports a flattering number for it.
HOLDOUT_MARKER = "holdout"

CHECKPOINT_SCHEMA = "flashml.demo/checkpoint/1"

#: The same model schema `../train/` writes and `../evaluate/` reads. The
#: parameter orientation below (W1 is features x hidden, not torch's
#: out x in) is chosen to match it: a model this file trains on a card can
#: be dropped beside `../evaluate/evaluate.py` and scored with no
#: conversion, which would not be true of a raw torch state_dict.
MODEL_SCHEMA = "flashml.demo/model/1"

CLASSES = 2


# --- the device: checked, never assumed ------------------------------------


class DeviceError(SystemExit):
    """No CUDA device, and the run did not opt into a CPU rehearsal.

    A SystemExit subclass so it exits non-zero with the message as its only
    output — the task fails, loudly, rather than producing a metrics.json
    that a reader would have no reason to distrust.
    """


def select_device(allow_cpu: bool) -> torch.device:
    """The device to train on, after proving it is what was asked for.

    Prints the card's name unconditionally when there is one. That line is
    the evidence a reader needs and the reason the whole heterogeneous-fleet
    demonstration is checkable: "it ran on a GPU" is an assertion, and
    "NVIDIA GeForce RTX 3070" in the task log is a fact.
    """
    print(
        f"torch {torch.__version__}, built against CUDA {torch.version.cuda}",
        flush=True,
    )
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
        print(
            f"CUDA device {index}: {name} — compute capability {major}.{minor}, "
            f"{total_gb:.1f} GiB VRAM",
            flush=True,
        )
        if allow_cpu:
            print(
                "--allow-cpu was passed but a CUDA device is present, so it "
                "changes nothing: the flag only permits a fallback, it never "
                "forces one.",
                flush=True,
            )
        return torch.device("cuda", index)

    if allow_cpu:
        # Loud on purpose, and on stdout rather than stderr so it lands in
        # the task log in order with everything else. A CPU rehearsal that
        # reads like a successful GPU run is the exact failure this workload
        # exists to make impossible.
        bar = "!" * 74
        print(bar, flush=True)
        print(
            "!! --allow-cpu: NO CUDA DEVICE FOUND, RUNNING ON THE CPU.\n"
            "!! This is a rehearsal of the data, checkpoint and resume paths.\n"
            "!! It is NOT a GPU run, it proves nothing about routing, and any\n"
            "!! number it produces is a number from a CPU. Never pass this\n"
            "!! flag in flashml.yaml.",
            flush=True,
        )
        print(bar, flush=True)
        return torch.device("cpu")

    raise DeviceError(
        "no CUDA device is visible to torch on this machine, and this job "
        "declared `resources: {gpus: 1}`. Refusing to train on the CPU and "
        "report it as a GPU run. If a host was placed this task without a "
        "card, that is a placement fault worth seeing; if you are rehearsing "
        "on a laptop, pass --allow-cpu and read the banner it prints."
    )


def assert_on_device(device: torch.device, **tensors: torch.Tensor) -> None:
    """Every named tensor really is on `device`, or stop.

    Cheap, and it catches the one silent failure that matters: a `.to(device)`
    whose result was discarded, which leaves the offending tensor on the CPU
    and lets the whole run proceed at CPU speed with a GPU in the log header.
    """
    wrong = {
        name: str(tensor.device)
        for name, tensor in tensors.items()
        if tensor.device.type != device.type
    }
    if wrong:
        raise DeviceError(
            f"expected every tensor on {device.type!r}, but {wrong} — a "
            f"`.to(device)` result was dropped somewhere and this run would "
            f"have been slower than a CPU run while claiming to be a GPU one"
        )


# --- the model: two matmuls and a tanh, deliberately -----------------------


def init_params(seed: int, features: int, hidden: int) -> dict[str, np.ndarray]:
    """The initial weights, from a seed alone, in numpy.

    Numpy and not torch, and on the host and not the device, for three
    reasons that all point the same way: the starting point is then a pure
    function of the seed rather than of torch's global RNG state; it is
    identical to the one `../train/` begins from and `../evaluate/` scores as
    its untrained floor; and it does not change if a different card, driver
    or torch build runs the job.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=features**-0.5, size=(features, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(scale=hidden**-0.5, size=(hidden, CLASSES)),
        "b2": np.zeros(CLASSES),
    }


def to_device(params: dict[str, np.ndarray], device: torch.device) -> dict:
    """Trainable float32 tensors on `device`, keeping the numpy orientation.

    float32 because that is what a consumer card is fast at and what this
    demo has no reason to exceed. The JSON round-trip below is still exact:
    a float32 widens to a Python float losslessly and narrows back the same
    way, so a checkpoint neither gains nor loses precision.
    """
    return {
        name: torch.tensor(value, dtype=torch.float32, device=device, requires_grad=True)
        for name, value in params.items()
    }


def forward(params: dict, x: torch.Tensor) -> torch.Tensor:
    """Logits. The orientation is the numpy one — `x @ W1`, not `W1 @ x` —
    which is what keeps the emitted model.json readable by `../evaluate/`."""
    hidden = torch.tanh(x @ params["W1"] + params["b1"])
    return hidden @ params["W2"] + params["b2"]


@torch.no_grad()
def accuracy(params: dict, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((forward(params, x).argmax(dim=1) == y).to(torch.float32).mean())


# --- the JSON encoding, shared by every workload in this suite -------------


def encode(params: dict) -> dict:
    """``{"W1": {"shape": [...], "data": [...]}, ...}``.

    The same encoding `../train/` writes and the federated delta protocol
    uses. `.detach().cpu()` is not optional: a tensor that still carries a
    grad_fn or still lives in VRAM is not something `json.dumps` can see.
    """
    return {
        name: {
            "shape": list(value.shape),
            "data": [float(v) for v in value.detach().cpu().reshape(-1).tolist()],
        }
        for name, value in params.items()
    }


def decode(blob: dict, device: torch.device) -> dict:
    return {
        name: torch.tensor(
            entry["data"], dtype=torch.float32, device=device
        )
        .reshape(entry["shape"])
        .requires_grad_(True)
        for name, entry in blob.items()
    }


# --- the data --------------------------------------------------------------


def load_training_shards(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every `.npz` the agent placed here except the holdout.

    `rglob`, not `glob`: the shards arrive one level down (`train/*.npz`,
    `holdout/eval.npz`) and a flat glob finds nothing at all.

    The holdout is excluded by the path RELATIVE TO `data_dir`, never by
    `p.name`. `holdout/eval.npz` is named `eval.npz`, so a name-based check
    silently folds the evaluation shard into training — a bug that produces a
    working run, a plausible model and a held-out score that is not held out.

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
    everything = sorted(data_dir.rglob("*.npz"))
    shards = [
        p for p in everything if HOLDOUT_MARKER not in str(p.relative_to(data_dir))
    ]
    if not shards:
        raise SystemExit(
            f"{data_dir} holds no training shards (found: "
            f"{[str(p.relative_to(data_dir)) for p in everything]}). Every "
            f"path containing {HOLDOUT_MARKER!r} is reserved for the evaluate "
            f"workload."
        )
    xs, ys = [], []
    for shard in shards:
        with np.load(shard) as bundle:
            xs.append(bundle["X"] if "X" in bundle else bundle["x"])
            ys.append(bundle["y"])
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    return x, y, [str(p.relative_to(data_dir)) for p in shards]


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

    The device the previous attempt used is deliberately NOT compared. A
    checkpoint is weights and a step count; the whole point of relaying it is
    that a 3090 can pick up what a 3070 was doing.
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="fall back to the CPU when no CUDA device is present, printing a "
        "banner that says so. For rehearsing the data, checkpoint and resume "
        "paths on a laptop ONLY — a run with this flag proves nothing about "
        "GPU routing and must never appear in flashml.yaml's args.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    device = select_device(args.allow_cpu)

    out = Path(args.out)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host_x, host_y, shard_names = load_training_shards(Path(args.data))
    features = int(host_x.shape[1])
    steps_per_epoch = max(1, len(host_x) // args.batch_size)

    # The whole dataset is a few thousand rows, so it goes to the card once
    # and stays there. Streaming it per batch would be the realistic shape
    # for a real workload and would spend the entire run in host-to-device
    # copies for this one.
    x = torch.from_numpy(host_x).to(device)
    y = torch.from_numpy(host_y).to(device)
    assert_on_device(device, features_tensor=x, labels_tensor=y)

    #: Every number that decides what this run computes. Written into each
    #: checkpoint and compared on resume — continuing an 8-epoch run's
    #: weights under a 4-epoch schedule produces a model no reported number
    #: describes. Nothing about the hardware belongs in here: a checkpoint
    #: written by one card must be resumable on another.
    run = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden": int(args.hidden),
        "batch_size": int(args.batch_size),
        "features": features,
        "samples": int(len(host_x)),
        "steps_per_epoch": steps_per_epoch,
        "steps_total": steps_per_epoch * int(args.epochs),
    }

    resumed = load_resume(Path(args.resume), run)
    if resumed is None:
        params = to_device(init_params(run["seed"], features, run["hidden"]), device)
        start_epoch = 0
        resumed_from = None
        print(
            f"fresh start — {run['samples']} samples from {len(shard_names)} "
            f"shard(s), {run['epochs']} epochs x {steps_per_epoch} steps on "
            f"{device.type}",
            flush=True,
        )
    else:
        params = decode(resumed["params"], device)
        start_epoch = int(resumed["progress"]["epoch"])
        resumed_from = int(resumed["progress"]["step"])
        print(
            f"RESUMED at step {resumed_from} (epoch {start_epoch}/"
            f"{run['epochs']}) — that much work survived the machine dying. "
            f"The checkpoint records no device, so it does not matter whether "
            f"the card below is the one that wrote it",
            flush=True,
        )
    assert_on_device(device, **params)

    if device.type == "cuda":
        print(
            f"resident on the card: "
            f"{torch.cuda.memory_allocated(device) / 1024**2:.1f} MiB "
            f"(data + parameters, before the first backward pass)",
            flush=True,
        )

    optimizer = torch.optim.SGD(list(params.values()), lr=run["lr"])
    objective = torch.nn.CrossEntropyLoss()

    written: list[str] = []
    loss_value = float("nan")
    train_accuracy = float("nan")
    for epoch in range(start_epoch, run["epochs"]):
        # Drawn from (seed, epoch) and nothing else — not torch's global RNG,
        # which a resumed attempt has no way to reconstruct. This is what
        # makes a resumed run do exactly the work the dead one would have.
        order = np.random.default_rng([run["seed"], epoch]).permutation(len(host_x))
        index = torch.from_numpy(order).to(device)
        for step in range(steps_per_epoch):
            batch = index[step * args.batch_size : (step + 1) * args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(forward(params, x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        # One sync per epoch rather than one per step: reading a scalar off a
        # tensor blocks until the queued kernels finish, and doing that every
        # step would make the launch overhead the thing being measured.
        # `.detach()` first, or torch warns that a grad-tracking tensor is
        # being collapsed to a Python float.
        loss_value = float(loss.detach())
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
                "metrics": {"loss": loss_value, "train_accuracy": train_accuracy},
            },
        )
        written.append(f"ckpt/{target.name}")
        print(
            f"  epoch {epoch + 1:>2}/{run['epochs']}  step {step_reached:>4}/"
            f"{run['steps_total']}  loss {loss_value:.4f}  acc "
            f"{train_accuracy:.3f}  -> {target.name}",
            flush=True,
        )

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )

    model_bytes = write_json_atomically(
        out / "model.json",
        {
            "schema": MODEL_SCHEMA,
            "architecture": {
                "kind": "mlp-tanh-softmax",
                "features": features,
                "hidden": run["hidden"],
                "classes": CLASSES,
            },
            "params": encode(params),
            "trained_on": {"shards": shard_names, "samples": run["samples"]},
            "trained_for": {"seed": run["seed"], "epochs": run["epochs"]},
            # Which machine produced these weights. Recorded in the model and
            # not only in the metrics because the model file is the thing that
            # outlives the job.
            "trained_with": {"device": device.type, "device_name": device_name},
        },
    )

    write_json_atomically(
        out / "metrics.json",
        {
            "workload": "demo-suite/gpu-train",
            "loss": loss_value,
            "train_accuracy": train_accuracy,
            # No holdout number here on purpose: scoring the held-out shard is
            # the evaluate workload's job, and a trainer that grades its own
            # homework is how a demo ends up with two disagreeing accuracies.
            "device": device.type,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
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
        f"done in {time.monotonic() - started:.1f}s on {device_name} — "
        f"{len(written)} checkpoint(s), train accuracy {train_accuracy:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
