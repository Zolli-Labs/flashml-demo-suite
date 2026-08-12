# gpu-train — one task on a real card, checkpointed and resumable

The routing story. The other four workloads in this suite are CPU-only, and
the console classifies every one of them as something other than training —
which is honest, and which also means that on a fleet of RTX 3070s, 3090s and
4090s the cards would be decorative. This is the workload that makes the
hardware part of the demonstration instead of part of the scenery.

It is the same shape as [`../train/`](../train/): one task, no sweep, one
checkpoint per epoch, resumable from wherever a dead machine left off. Two
lines are different, and both matter.

```yaml
image: pytorch-cuda
resources:
  gpus: 1
```

## What those two lines buy

`resources.gpus >= 1`, on a job with no sweep and no partition, is the **only
way a `flashml.yaml` can say "this is training"**. The classifier reads it and
answers, with its evidence:

> **TRAINING** — *one task, no sweep and no partition, asking for 1 GPU(s) per
> task under a 1800s timeout — a single long job that wants a real card for
> its duration*

`../train/` is a training job by any ordinary reading and the product calls it
COMMAND, because there is no GPU on the CPU fleet and an unclaimed kind beats a
confident wrong one. `../README.md` records that as something the schema
cannot say. This directory is the sentence that *can* be said, and the whole
point of saying it is what happens next: kind decides which venues the work is
offered to **before** any price is computed.

| Venue | | Why, in the router's own words |
|---|---|---|
| `owned` | suited | *a home rig in the gpu-24gb class is a real training machine… The caveat is interruption: a long single job loses the most when a volunteer machine disappears, and only checkpointed work survives it* |
| `runpod` | suited | *a rented pod is a card held for the whole run, which is what a long job needs and what a volunteer fleet cannot promise* |
| `fc-gpu` | suited | *T4 16 GB through Hopper 96 GB… Not integrated, so it can hold no work today* |
| `fc-sandbox` | **refused** | *2 vCPU, 2 GB RAM and no GPU (gpuConfig: null…). Gradient work does not fit in that at any duration — this is the venue's hard limit, not a judgement about its price* |

That last row is the demonstration. `fc-sandbox` is refused **on hardware**,
by name, with a reason, and no price ever gets quoted for it. Submit
`../train/` and the same venue is a perfectly good answer. Nothing about the
model changed; the declaration did.

The requirement also survives compilation. `resources: {gpus: 1}` becomes
`spec.resources.gpuPerTask = 1`, and the coordinator's expansion puts
`gpus: 1` in the task payload where the placement gate reads it — so a host
that never advertised a card is not offered the task at all.

## The device is checked, not assumed

A "GPU job" that quietly ran on a CPU is the exact failure this demo must not
have, because nothing about it looks wrong: the task succeeds, the metrics are
plausible, and the log says "training" the whole way down.

So `train.py` refuses. No CUDA device means a non-zero exit and a message
naming the situation, not a fallback. When there *is* a device it prints what
it found before it prints anything else — this is the shape of those lines,
with the values left as placeholders because nobody here has run it on a card
yet:

```
torch <version>, built against CUDA <version>
CUDA device 0: <torch.cuda.get_device_name(0)> — compute capability <x.y>, <N> GiB VRAM
resident on the card: <N> MiB (data + parameters, before the first backward pass)
```

The card's name comes straight from `torch.cuda.get_device_name(0)`, so the
task log — not a submission form — is what says which machine did the work.
The script then checks that the tensors it is about to train on really are on
that device — a dropped `.to(device)` leaves a tensor on the host, runs the whole
job at CPU speed, and reports a GPU in the header the entire time.

`--allow-cpu` exists for rehearsing on a laptop. It is **off by default**, it
only permits a fallback (it never forces one), and it prints a banner nobody
will mistake for a real run. It must never appear in `args:`.

## The checkpoint convention — unchanged by the hardware, and one new trap

Identical to [`../train/`](../train/), which is the file to read for the full
version:

| | |
|---|---|
| write `/work/out/ckpt/step-<N>.json` | that directory, that filename, `N` the integer step reached, one file per checkpoint, written **atomically** |
| read `/work/inputs/resume.json` | the last committed checkpoint, staged before the process starts. **Absent on a first attempt** — "no such file" means "start from scratch", not an error |

There is no `checkpoint:` key in `flashml.yaml` and nothing to switch on:
`parameters["checkpoint"] = {}` is emitted for every job. Preflight's
`no-checkpoint` warning does not fire here, and should not.

The trap that is specific to a torch workload:
**`flashruntime.torch.checkpoint()` does not satisfy this.** It is the obvious
helper to reach for, it gets the *directory* right — its root defaults to
`<out>/ckpt` — and then writes `step-000100/` **directories** of `model.pt`.
The relay globs `step-*.json` **files**. Nothing it writes is ever shipped,
nothing raises, and the first sign is a retry that starts from zero. State
goes to JSON here for that reason and no other.

Weights are stored in the same `{"shape": [...], "data": [...]}` encoding the
rest of the suite uses, in the numpy orientation (`W1` is *features × hidden*,
not torch's *out × in*). That costs nothing and buys interoperability — see
"Verified locally" below.

## Reading the dataset — the bug that already bit four workloads

The shards land **one level down**: `/work/data/demo/train/*.npz` and
`/work/data/demo/holdout/eval.npz`. Two consequences, both of which have
already been got wrong once:

- **`rglob`, not `glob`.** A flat glob of the data directory finds nothing at
  all.
- **Match the holdout against the path relative to the data directory, never
  against `p.name`.** `holdout/eval.npz` is *named* `eval.npz`, so a
  name-based check folds the evaluation shard into training and produces a
  run that works, a model that looks fine, and a held-out score that is not
  held out.

The job is not federated, so `split` is inferred as `replica` and this single
task is handed the **whole** listing — holdout included. The compiler confirms
it: `entries = ['holdout/eval.npz', 'train/shard-000.npz', … 'train/shard-005.npz']`.
Keeping the holdout out of training is entirely the code's job.

Arrays are `X` (float32) and `y` (int64); lowercase `x` is accepted as a
fallback, as the siblings do.

## The unsandboxed host

A rented pod is itself a container and cannot run Docker-in-Docker, so
flashnode's `trusted` runner executes argv directly and **installs the job's
dependencies instead of using the image**. That makes the choice of image a
dependency decision, not just a runtime one.

`compile.py` refuses a job whose image is not curated and which declares no
`dependencies:`, because an unsandboxed host would have no way to reproduce
its environment. `pytorch-cuda` **is** curated, so the compiler resolves its
`requirements.txt` as the job's dependency base. Verified by compiling this
config and reading the emitted parameter:

```
parameters['dependencies'] = ['--index-url https://download.pytorch.org/whl/cu124',
                              'torch==2.4.1',
                              'numpy==1.26.4']
```

That is a CUDA build of torch from PyTorch's own index, and it is what the pod
installs into a venv keyed by the hash of that list. Nothing else is needed,
so **this directory declares no `dependencies:`**, and that absence is a
decision rather than an omission. Declaring extras would also emit
`extra_dependencies`, which the coordinator's placement gate reads as *"this
job needs a host that can install"* — routing it away from the GPU **container**
hosts that run it correctly today. Adding a package here costs eligibility, so
add one only when the job actually needs it.

`--index-url` is the first line for a reason: it governs the lines that follow
it, which is the difference between a CUDA wheel and a CPU one. Extras are
appended after the base, never before.

## Not a benchmark

A one-hidden-layer MLP over a few thousand synthetic rows, eight epochs. It is
sized to finish in seconds so that nobody pays for an hour of rented GPU to
watch a routing decision, and so a person can watch it end. **No number this
job prints is a claim about a card's speed or about the model's quality.** The
lines worth reading are the device banner and `resumed_from_step`.

Almost all of the wall clock on a machine's first task is spent elsewhere:
pulling several gigabytes of CUDA image on a container host, or downloading a
torch cu124 wheel on an unsandboxed one. `timeout_seconds: 1800` is sized for
that, is paid once per machine, and is not the model's fault.

## Verified locally

`--allow-cpu`, against the real published dataset (6 training shards of 6000
rows × 24 features, plus a 4000-row holdout) — so these are **CPU numbers from
a rehearsal**, and the only thing they are evidence for is that the data,
checkpoint, resume and interop paths are correct before anyone rents a card:

```
$ FLASHML_WORK_DIR=./w python train.py --epochs 8 --lr 0.5 --hidden 64 \
      --batch-size 128 --allow-cpu
[the six-line --allow-cpu banner, elided here, is printed first every time]
fresh start — 36000 samples from 6 shard(s), 8 epochs x 281 steps on cpu
  epoch  1/8  step  281/2248  loss 0.4512  acc 0.801  -> step-281.json
  ...
  epoch  8/8  step 2248/2248  loss 0.3459  acc 0.808  -> step-2248.json
done in 1.0s on cpu — 8 checkpoint(s), train accuracy 0.8083

$ cp w/out/ckpt/step-1124.json w/inputs/resume.json && rm -rf w/out
$ FLASHML_WORK_DIR=./w python train.py --epochs 8 --lr 0.5 --hidden 64 \
      --batch-size 128 --allow-cpu
RESUMED at step 1124 (epoch 4/8) — that much work survived the machine dying.
The checkpoint records no device, so it does not matter whether the card below
is the one that wrote it
  epoch  5/8 ... epoch 8/8
done in 0.6s on cpu — 4 checkpoint(s), train accuracy 0.8083

model.json sha256 03dedfcc… — identical to the uninterrupted run
```

Six shards, not seven: the holdout was excluded, by relative path.

Run `../train/`'s numpy trainer over the same data with the same
hyperparameters and the **per-epoch log lines are identical, all eight of
them** — same loss to four decimals, same accuracy to three. In full
precision, `0.34589275…` against `0.34589271…`, which is float32 against
float64 and not a difference in the arithmetic. That is the cheapest evidence
that this is a port of the suite's model and not a second model that happens
to share its name.

The `model.json` this workload wrote was then scored, unmodified, by
[`../evaluate/`](../evaluate/):

```
$ python ../evaluate/evaluate.py --hidden 64 --model w/out/model.json
model.json: accuracy 0.8043, loss 0.4223 over 4000 held-out samples from eval.npz
```

against an untrained floor of `0.5100` on the same shard. That works only
because the emitted parameters keep the suite's orientation rather than a raw
torch `state_dict`.

**What is not claimed:** that a resumed run on a GPU produces bit-identical
weights. `../train/` can claim that on a CPU and does. Here the *work* is
exactly reproduced — same starting weights (drawn by numpy from the seed
alone), same batch order (a pure function of `(seed, epoch)`, never torch's
global RNG), same data — but cuBLAS is free to reduce in a different order
than a CPU kernel and we have not measured run-to-run bitwise equality on any
card. When someone measures it, this paragraph should say what they found.

The classification, the preflight findings, the compiled dependency list and
the task count above were all produced by running the control plane's own
`parse_flashml_yaml`, `resolve_image`, `preflight`, `compile_to_jobspec` and
`router.classify` over this directory — not read off the schema.

## What the schema cannot say

Recorded here rather than worked around, in addition to the four in
[`../README.md`](../README.md):

- **A repo cannot ask for an unsandboxed host, or for a particular card.**
  `isolation.tier` is fixed at `sandboxed` by the compiler and
  `allowFallback` is set only when the job is submitted **into a team pool** —
  a console decision, not a `flashml.yaml` key, because a submitter must never
  be able to lower the isolation their own arbitrary code runs under. So a
  rented pod runs this only when it is a member of the pool the job was
  submitted to. `resources.gpus` is a *count*; there is no `gpu_type`,
  `vram_gb` or compute-capability floor, so "a 3090, not a 3070" is not
  expressible and a fleet of mixed cards is chosen between by the scheduler
  and the price, not by this file.
- **`resources.gpus` is the only training signal there is.** The classifier
  would rather read a declared base-model input, and none exists; a
  `--pretrained`-style flag in `args:` is the nearest thing and would classify
  this FINETUNE instead. Both route identically today, which is the only
  reason that is survivable.
- **A CPU training run still has no way to declare itself training.** This
  directory does not fix that — it sidesteps it by genuinely wanting a card.

## Submitting it

`flashml.yaml` is read from the **root** of a repository, so this
subdirectory cannot be submitted from the monorepo. It is self-contained —
which is why the model code is repeated rather than shared — so copy it out:

```
cp -r examples/demo-suite/gpu-train  ../my-flashml-gpu-demo
cd ../my-flashml-gpu-demo && git init && git add . && git commit -m "demo"
```

Then paste that repo's URL into the console, and submit it **into a pool that
has a GPU host in it** — see "What the schema cannot say" above for why the
pool is what unlocks a rented pod.
