# train — one task, checkpointed and resumable

The fault-tolerance story, and the only file in this suite that has to be
read closely.

One task. No sweep, no rounds, no fan-out. What makes it interesting is that
it can be killed at any point and picked up somewhere else without losing
more than one epoch — and that the model it finally writes is **byte-for-byte
identical** to the one an uninterrupted run would have written.

## The convention, in full

There is no `checkpoint:` key in `flashml.yaml`. Writing one is refused with
an explanation, because there is nothing to switch on: every compiled job
carries the checkpoint relay, and the only question is whether your code
gives it anything to watch.

Two halves. Both fixed, both hardcoded in the host agent:

| | |
|---|---|
| write `/work/out/ckpt/step-<N>.json` | That directory, that filename, `N` the integer step reached, one file per checkpoint. The agent globs `step-*.json` in that one directory and ships each new file off the machine the moment it appears |
| read `/work/inputs/resume.json` | The last checkpoint the previous attempt committed, staged before your process starts. **Absent on a first attempt** — "no such file" means "start from scratch", not an error |

Anything else is not a checkpoint. `/work/out/checkpoints/` is the plausible
spelling and is the one that quietly fails: the file is collected as an
ordinary artifact when the task finishes, and lost with the machine if it
does not. Preflight warns about that one specifically (`checkpoint-path`).

**`flashruntime.torch.checkpoint()` does not satisfy this.** It is the
closest thing to a trap in the system because it gets the directory *right*
— its root defaults to `<out>/ckpt` — and then writes `step-000100/`
**directories** of `model.pt`. The relay globs `step-*.json` **files**. None
of it is ever shipped, nothing raises, and the first sign is a retry that
starts from zero.

## Write it atomically

The relay ships what it finds and can find your file a millisecond after you
create it. A checkpoint caught half-written is a checkpoint nothing can
resume from — and the run it kills is the *next* one, which makes it a bug
you discover long after you wrote it.

`write_json_atomically` in `train.py` is the full-strength version: temp file
**in the same directory** (rename is atomic only within one filesystem, so
`/tmp` is not an option), a temp name that does **not** match `step-*.json`
(or the relay would ship the half-written file you were hiding), `os.replace`,
then fsync the file and the directory so it survives a power cut and not just
a killed process.

## Resume must not change the answer

The batch order for epoch *e* is a pure function of `(seed, e)` — not of a
serialised PRNG state. An attempt that starts at epoch 7 therefore draws
exactly the permutation the dead attempt would have drawn, which is the one
design decision that makes a resumed run *equal* to an uninterrupted one
rather than merely *similar* to it.

Verified locally, against a stand-in dataset of the published shape:

```
$ FLASHML_WORK_DIR=./w python train.py --epochs 10 --lr 0.5 --hidden 8
fresh start — 4096 samples from 8 shard(s), 10 epochs x 32 steps
  epoch  1/10  step   32/320  loss 0.1440  acc 0.967  -> step-32.json
  ...
  epoch 10/10  step  320/320  loss 0.0621  acc 0.973  -> step-320.json

$ cp w/out/ckpt/step-192.json w/inputs/resume.json && rm -rf w/out
$ FLASHML_WORK_DIR=./w python train.py --epochs 10 --lr 0.5 --hidden 8
RESUMED at step 192 (epoch 6/10) — that much work survived the machine dying
  epoch  7/10 ... epoch 10/10

model.json sha256 identical to the uninterrupted run
```

`FLASHML_WORK_DIR` exists so the whole thing can be rehearsed without root.
FlashML never sets it; in production every path defaults to `/work`.

## The other refusals

Two things `train.py` refuses rather than works around, both because the
alternative looks like success:

- A **`resume.json` that is present and broken** raises instead of starting
  over. A silent restart would look like a successful recovery while
  throwing away every minute the dead attempt spent.
- A **checkpoint written under different hyperparameters** is refused.
  Continuing a 10-epoch run's weights under a 4-epoch schedule produces a
  model that no reported number describes.

## What it is not

Not a benchmark. Ten epochs of a forty-line numpy MLP over 4096 synthetic
rows, finishing in well under a second natively — sized so a person can
watch it, and so the demo fleet's qemu emulation does not turn a
demonstration into a wait. `train_accuracy` is in `metrics.json` because it
is the cheapest evidence that the loop learned anything at all; the held-out
number belongs to [`../evaluate/`](../evaluate/), because a trainer that
grades its own homework is how a demo ends up reporting two accuracies that
disagree.
