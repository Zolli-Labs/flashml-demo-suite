# federated — one model, several machines, a different shard each

Four rounds. Every machine gets the same starting weights and a different
slice of the data, trains alone, uploads how much it changed the weights,
and the platform averages those changes into the next round's weights.

This is the same protocol as [`../../federated/`](../../federated/) and
[`../../federated-dataset/`](../../federated-dataset/); what is different
here is that it is numpy rather than torch, and that it shares the suite's
one dataset with the other three workloads. Read those two first if this is
the first federated example you have opened — they explain the protocol at
length, and this README does not repeat them.

## This is not DDP

DistributedDataParallel needs every rank to reach every other rank on each
backward pass. Task containers run with `--network none`, because a
volunteer is lending you a machine and not an open socket. There are no
peers to reach and no rank 0 to rendezvous with.

Weights cross the network once per round instead of once per batch. That is
what makes it survivable on laptops over home wifi, and it is a worse
convergence rate per round than synchronous DDP. That is the trade.

## The contract this file implements

| | |
|---|---|
| read `/work/inputs/weights.json` | the round's starting weights. **Absent on round 0** — that absence is the signal to use your own initialisation |
| write `/work/out/delta.json` | `{"<param>": {"shape": [...], "data": [...]}}`. On round 0, where you were given nothing, write the trained weights themselves |
| write `/work/out/metrics.json` | at least `{"chunks_done": [...], "loss": <number>, "samples": <int>}` |

`chunks_done` is the one people miss and it fails **silently**: a
contribution reporting no chunks is credited for nothing however long it
trained, the round does not error, it waits out its timeout and combines an
empty set.

Preflight refuses this job outright (`federated-contract`, an error rather
than a warning) if the entrypoint never mentions those paths — because
without them a federated run does not degrade, it burns N rounds of
volunteers' electricity and then fails.

## No shard count, no quorum

Both absences are the point, and both are refused if you write them.

`shards` was a guess about the fleet, made before submitting, by the person
with the least information about it. The platform cuts one pass into chunks
and hands each machine as many as it can finish.

`min_participants` went with it. A round used to close on a headcount; it
now closes when the chunks that came back **cover** `sync_every` of a pass.
That is what a quorum was standing in for, and unlike a headcount it cannot
be satisfied by one fast machine reporting first.

Round count is derived — `epochs / sync_every` — and shown in the console.
You never type it. `epochs: 4` here is four rounds.

## Do not slice the data again

`--shard` and `--num-shards` still arrive and still mean what they meant:
they identify this chunk so `chunks_done` can credit it. **They are no
longer how data is selected.** `mode: federated` infers `split: shard`, so
the host agent already fetched exactly this task's files before the sandbox
closed.

```python
# WRONG with datasets: — trains on a fraction of a fraction
x, y = stride(x, y, args.shard, args.num_shards)

# RIGHT — the agent already gave you exactly your slice
for shard in sorted(DATA_DIR.glob("*.npz")):
    ...
```

Two consequences of `split: shard` worth knowing before you scale this up:
the fleet can never use more machines than the dataset has files, and the
split balances **bytes**, so one dominant file can strand machines even when
there are as many files as chunks.

## The holdout shard, and the one failure this can produce

The suite's dataset carries a holdout shard, and under `split: shard` the
whole listing is cut across the fleet — so whichever machine is handed the
holdout must not train on it, or the number `../evaluate/` reports as held
out was seen by the model. `train.py` drops it by name.

If a machine's slice turns out to be *only* the holdout, the task fails
loudly rather than reporting a chunk it did not train. That is the right
disposition: crediting coverage for work that never happened would close the
round believing a slice of the pass was covered when it was not. A `select:`
glob in `flashml.yaml` would remove the case entirely and is a one-line
change — see the suite README for why it is not written here.

## Why this one has no checkpoints

Preflight warns `no-checkpoint` on this file. The warning is correct and the
absence is deliberate.

A round here is a few seconds of local training over one chunk. A machine
that dies loses that chunk, and the next round covers it, because coverage
is what closes a round. The unit of fault tolerance at this layer is the
round, not the epoch, and there is nothing between rounds worth relaying:
the weights this task started from came from the coordinator and are already
durable there.

A *longer* federated task — minutes of local work per round rather than
seconds — should follow [`../train/`](../train/) and checkpoint inside the
round. Resume stays inside the round in any case: each round is a separate
coordinator job, so round 4's `task-000` can never resume round 3's.

## Verified locally

Four slots, two shards each, driven through FlashML's own `reduce_deltas`
rather than a reimplementation of it:

```
round 0: mean loss 0.6403   (slots: 0.6341 0.6414 0.6368 0.6491)
round 1: mean loss 0.5550
round 2: mean loss 0.4428
round 3: mean loss 0.3287
```

**A flat loss is the failure to watch for.** It means every machine trained
and nothing was combined — usually `delta.json` holding whole weights
instead of the change, from round 1 onward.
