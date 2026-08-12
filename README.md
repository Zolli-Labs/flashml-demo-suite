# The FlashML demo suite

Four small jobs, one dataset, every execution mode the product supports.
Each directory is a complete, submittable job: a `flashml.yaml`, one Python
file, and a README explaining the one thing it exists to show.

| Directory | Mode | Console labels it | What it demonstrates |
|---|---|---|---|
| [`train/`](train/) | one task, `mode: independent` | **COMMAND** | Checkpointing and resume — the fault-tolerance story |
| [`hpo/`](hpo/) | `sweep:` over two axes | **HPO** | Fan-out into six ranked trials |
| [`federated/`](federated/) | `mode: federated` | **FEDERATED** | Rounds, one shard per machine, averaged between |
| [`evaluate/`](evaluate/) | one task, validated | **EVALUATION** | Scoring a model against held-out data |

They are meant to be read in that order. `train/` produces a model,
`evaluate/` scores it, `hpo/` searches for a better one, and `federated/`
trains the same model across machines that never talk to each other.

## These are correctness demos, not benchmarks

Nothing here is optimised and no number here is a claim about FlashML's
speed, a machine's speed, or the model's quality. The model is a
one-hidden-layer MLP in about forty lines of numpy; the dataset is a few
thousand rows of synthetic, linearly separable data; a task finishes in well
under a second of actual compute.

That is deliberate, and it is not only about the demo's attention span. The
demo fleet is **arm64 and the curated images are amd64**, so every task runs
under qemu emulation — several times its native cost, on hardware nobody is
being paid for. A workload sized to be interesting would be a workload
nobody watches finish.

What these demonstrate is that the loop is *correct*: the data arrives
sliced, the task runs sandboxed with no network, the checkpoint relay picks
up what the convention tells it to, a killed attempt resumes exactly where
it left off, and the results come back validated and reduced. If you want
numbers about performance, the benchmark suite in `flashruntime` is the
place that reports them honestly, host and all.

## The dataset

All four jobs declare the same one, and the whole declaration is:

```yaml
datasets:
  - name: demo
    source: https://zolli-flashml-datasets.oss-ap-southeast-1.aliyuncs.com/datasets/mlp-demo/manifest.json
```

It resolves to several `.npz` training shards plus a holdout shard, readable
with numpy alone — which is why `.npz` and not parquet: numpy ships in
`sklearn`, `pytorch-cpu` and `pytorch-cuda` alike, and no curated image
ships pyarrow.

The task never downloads anything. The **host agent** fetches the shards
before the sandbox closes, verifies each sha256 from the pinned manifest,
and the task starts with its files already at `/work/data/demo/`. The
container still runs with `--network none`; nothing about the isolation
weakens.

**The holdout shard is separated by filename, not by `select:`.** Every
script here treats a file whose name contains `holdout` as reserved for
`evaluate/`. A `select:` glob in the YAML would be the sharper tool and is a
one-line change — but a glob that does not match anything is refused at
submit time for the whole job, so the filter lives in the code where a
surprise costs one task instead of the run.

## Where your own data lives — FlashML is not a dataset host

This example's data sits in a bucket we own because it is **our** example
data, shipped the way any project ships fixtures. It is not an upload
destination, and FlashML does not offer one.

**You host your data; you point us at it.** The control plane reads the file
listing, pins it to a revision, and hands each machine a list of URLs. The
shards travel from your origin straight to the machine that needs them. We
never hold a copy, which is also why we never hold your storage bill, your
retention obligations, or your takedown requests.

Four schemes are addressable — `hf://`, `s3://`, `r2://` and `https://` — and
today **all of them must be public**. That is a statement about
*authorization*, not about the scheme: `https://` does not mean "public", it
means "addressed by URL, with empty authorization". A private or gated origin
is refused **at submit, by name**, rather than by trying a token we do not
have and letting thirty machines fail one at a time.

**Private origins are deferred, not designed away.** The intended shape is
that you grant FlashML read access to data that is already yours — a scoped,
revocable role rather than a long-lived key handed over — and each machine
receives an expiring, single-object capability for only the shards assigned to
it. Until that lands, private data keeps using `local_inputs`. See
`flashml-cloud/docs/superpowers/specs/2026-08-12-private-datasets-design-note.md`.

## Submitting one

`flashml.yaml` is read from the **root** of a repository. These live in
subdirectories of the `flashml` monorepo, so pasting this repo's URL into
the console does not submit any of them. Copy the directory you want into a
repository of its own — it is self-contained, which is why the model code is
repeated across the four rather than shared from a common module.

```
cp -r examples/demo-suite/train  ../my-flashml-demo
cd ../my-flashml-demo && git init && git add . && git commit -m "demo"
```

Then paste that repo's URL into the console.

## What the console says, and why it matters

Every job is classified before it is priced, because kind decides which
venues the work is offered to at all. The classification and its evidence
are both shown, so they can be argued with:

| | Kind | Evidence |
|---|---|---|
| `train/` | COMMAND | *"says what to run and not what kind of work it is: no sweep, no partition, no mode: federated, no reducer, no validators, no GPU requirement, no base model"* |
| `hpo/` | HPO | *"sweep over lr, hidden expands to 6 independent trials sharing one entrypoint, ranked by reduce.kind: rank — the trials are being selected between, which is what makes this a search rather than a fan-out"* |
| `federated/` | FEDERATED | *"mode: federated — 4 pass(es) over the data combining every 1, so 4 round(s). A round's tasks are chunks of one model's data, not independent work, and this API aggregates between rounds"* |
| `evaluate/` | EVALUATION | *"declares validators over accuracy and asks for no GPU — the tasks emit scores rather than weights, in a single task"* |

**`train/` is a training job that the product cannot call TRAINING**, and
that is not a bug in the example. `TRAINING` means *one long job that wants
a real card for its duration*, and the only way to say so is
`resources: {gpus: N}` with N ≥ 1. There is no GPU on this fleet, so the
honest answer is COMMAND — an unclaimed kind rather than a confident wrong
one. Adding `validators:` to make the label prettier would classify it
EVALUATION, which would be worse.

## Two workloads warn at submit, on purpose

Preflight emits `no-checkpoint` (a warning, never a refusal) when an
entrypoint mentions none of `/work/out/ckpt`, a `step-<N>.json` filename, or
`/work/inputs/resume.json`. `federated/` and `evaluate/` both trip it, both
correctly:

- **`evaluate/`** is one forward pass over a few hundred rows. There is no
  progress to save; a machine that dies costs a rerun.
- **`federated/`** is a few seconds of local training per round. A machine
  that dies loses that chunk, and the next round covers it — coverage is
  what closes a round, not a headcount. The unit of fault tolerance is the
  round, not the step.

The warning exists because the *opposite* mistake — a six-hour job with
nothing resumable — is invisible until a machine dies. `train/` is the
answer to it, and the only file here that has to be read closely.

There is no `checkpoint:` key in any of these files, and adding one is
refused with an explanation rather than ignored: checkpointing is
unconditional, and every compiled job carries `parameters["checkpoint"] =
{}` whether the entrypoint uses it or not.

## Verified locally

Run against a stand-in dataset of the documented shape — 8 shards of 512
rows × 20 features, plus a 512-row holdout — while the published manifest
was still going up:

```
train/      10 epochs x 32 steps, train accuracy 0.973, 10 checkpoints
            resumed from ckpt/step-192.json -> model.json BYTE-IDENTICAL
            to the uninterrupted run (sha256 ea669548...)

evaluate/   untrained initialisation  0.4414   <- the floor
            train/'s model.json       0.9590   over 512 held-out rows

hpo/        lr=0.1  hidden=16   0.9707   <- ranked first
            lr=0.1  hidden=4    0.9629
            lr=0.3  hidden=16   0.9629
            lr=0.3  hidden=4    0.9570
            lr=1.0  hidden=16   0.9531
            lr=1.0  hidden=4    0.9473

federated/  4 slots x 2 shards, averaged with flashruntime's own
            reduce_deltas — not a reimplementation
            round 0: mean loss 0.6403   (0.6341 0.6414 0.6368 0.6491)
            round 1: mean loss 0.5550
            round 2: mean loss 0.4428
            round 3: mean loss 0.3287
```

The byte-identical resume is the one number worth caring about. It holds
because the batch order for epoch *e* is a pure function of `(seed, e)`
rather than a serialised PRNG state, so a resumed attempt draws exactly the
permutation the dead one would have. A flat federated loss is the failure to
watch for there: it means every machine trained and nothing was combined.

## What the schema cannot say

Recorded here rather than worked around:

- **A job cannot name another job's artifact.** `evaluate/` scores a model
  committed *inside its own repo*, because `datasets:` and `local_inputs:`
  are the only two ways data reaches a task and neither can say "the
  model job 8821 produced". The classifier admits as much in its own
  evidence for an EVALUATION job.
- **A CPU training run has no way to declare itself training** — see the
  table above.
- **`sync_every` accepts only `1.0` today.** Combining more often than once
  per pass needs an entrypoint that trains a *sequence* of chunks and
  reports every id it finished; a repo's entrypoint is handed one.
- **`epochs` is a federated-only key.** The other three pass `--epochs`
  through `args:` instead; writing it at the top level of an independent
  config is refused, which is the right refusal and a surprising one.

## Related

- [`../federated/`](../federated/) — the same federated protocol in PyTorch,
  with data generated inside the task and a `simulate.py` that rehearses the
  whole loop on a laptop.
- [`../federated-dataset/`](../federated-dataset/) — the same, against a
  dataset you host on Hugging Face.
