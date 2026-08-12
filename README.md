# hpo — a search, not a fan-out

Two axes, six trials, one winner. This is the shape a pull fleet of ordinary
machines is strictly good at: many short, independent trials that never have
to reach each other.

```yaml
sweep:
  lr: [0.1, 0.3, 1.0]
  hidden: [4, 16]           # 3 x 2 = 6 tasks
```

Each key becomes a CLI flag, so `trial.py` is run six times, once per
combination, as `--lr 0.3 --hidden 16`. Nothing in the sweep block says how
many machines; six trials spread across whoever is online.

Keys must be plain identifiers — a sweep key is both a flag name and a
substitution field, so `max-depth` cannot work and `max_depth` can. The cap
is 100 combinations, an order of magnitude below the point where anything
downstream complains, because a sweep that large is far more often a stray
axis than a plan.

## What makes it HPO and not a fan-out

The router classifies every job before it is priced, because kind decides
which venues the work is offered to at all — and it shows its evidence in
the console, so the answer can be argued with. This job's:

> *sweep over lr, hidden expands to 6 independent trials sharing one
> entrypoint, ranked by reduce.kind: rank — the trials are being selected
> between, which is what makes this a search rather than a fan-out*

The `sweep:` alone already reads as HPO. The **`reduce: {kind: rank}`** is
what earns the second half of that sentence. Six tasks that each write a
file and are never compared are a fan-out; six tasks whose scores are
ordered and whose best is named are a search.

## The two declarations that do work

```yaml
validators:
  keys: [accuracy]          # checked at COMMIT, on every attempt

reduce:
  kind: rank
  metric: accuracy
  maximize: true
```

`validators` is the only place a submitter can say what *valid* means. The
platform already checks that a result arrived and that it hashes correctly;
only you know that your `metrics.json` is meaningless without `accuracy`. A
trial that commits an intact file with no `accuracy` in it fails that
attempt and is retried elsewhere — rather than ranking silently at the
bottom, which is what happens without this line.

`reduce` is what turns a directory of six task outputs into an answer. It is
refused without a `kind`, because `reduce: {metric: accuracy}` is a
plausible typo that would otherwise mean *no reduction at all*, discovered
after the job ran.

`allow_partial: true` is there for the same reason it is in every honest
sweep: one closed laptop must not discard the other five machines' work. The
job then finishes as **PARTIAL**, a distinct terminal state, deliberately
not "succeeded".

## Every trial gets the whole dataset

`split: replica` is inferred for a non-federated job, so all six tasks
receive the entire file listing. Each trains on the training shards and
scores itself on the holdout shard, and — this is the part that matters —
**every trial makes that split the same way**, or the ranking would be
comparing numbers measured against different data.

## Trials checkpoint too

Checkpointing is on for every task, sweep trials included, and `trial.py`
writes `ckpt/step-<N>.json` per epoch and reads `/work/inputs/resume.json`
on start. It is seconds long and genuinely has little state worth saving —
it does it anyway, because the convention is the same one a six-hour task
depends on, and a demo suite that skipped it here would teach the wrong
lesson. See [`../train/`](../train/) for the full treatment.

## Verified locally

Against a stand-in dataset of the published shape:

```
lr=0.1  hidden=16   accuracy 0.9707   <- ranked first
lr=0.1  hidden=4    accuracy 0.9629
lr=0.3  hidden=16   accuracy 0.9629
lr=0.3  hidden=4    accuracy 0.9570
lr=1.0  hidden=16   accuracy 0.9531
lr=1.0  hidden=4    accuracy 0.9473
```

Six trials within three points of each other on a synthetic problem is not a
result and is not meant to be one. What it demonstrates is that six machines
each received a different combination, each committed a validated score, and
the platform named one of them the winner.
