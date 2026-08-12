# evaluate — score a model against the holdout shard

The one workload in this suite that produces a number instead of a model.

It reads the held-out shard the host agent fetched into `/work/data/demo/`,
loads the model committed beside it, and writes exactly one thing the job
declares as required:

```yaml
validators:
  keys: [accuracy]
```

An attempt whose `metrics.json` arrives intact and hashes correctly but
carries no `accuracy` **fails and is retried elsewhere**. Without that line,
"the file arrived" is the only check there is.

## The console calls this EVALUATION, and says why

> *declares validators over accuracy and asks for no GPU — the tasks emit
> scores rather than weights, in a single task*

That whole reading rests on the `validators:` block. Delete it and this job
classifies as COMMAND: nothing else in the file distinguishes "scores a
model" from "runs a script".

## Where the model comes from — and the gap that makes this awkward

**A job cannot name another job's artifact.** `flashml.yaml` has `datasets:`
(origins a host fetches) and `local_inputs:` (labels a host lends) and no
third thing, so "score the model job 8821 produced" is not expressible
today. The classifier admits it in its own evidence for an EVALUATION job:
*nothing names the artifact under test*.

So the model travels the way everything else in a job travels — **inside the
repo**. The repo tarball is staged at `/work/inputs/code/`, which is where
`--model` looks by default:

```bash
# 1. run ../train/ and download its model.json artifact from the console
# 2. drop it beside evaluate.py
# 3. submit
cp ~/Downloads/model.json examples/demo-suite/evaluate/model.json
```

With no `model.json` present it scores the **untrained initialisation**
instead — the same seeded weights `../train/` starts from — and says so in
`model_source`. That is the floor the trained model has to beat, and having
it means this workload runs the day the suite is submitted rather than only
after a training job has finished.

```
untrained-initialisation   accuracy 0.4414
train/'s model.json        accuracy 0.9590      over 512 held-out rows
```

## `split: replica`, and why the holdout is picked by name

A non-federated job infers `split: replica`, so this single task receives the
**whole** file listing — training shards included. It picks out the file
whose name contains `holdout` and ignores the rest. A few hundred kilobytes
of training data arrive and are never opened, which is cheaper than a
`select:` glob that has to be kept in step with the dataset's filenames and
that refuses the entire job at submit time when it does not match.

An empty match is a hard failure rather than an empty score: a task
reporting an accuracy over zero rows would publish a number with no data
behind it, and `validators: {keys: [accuracy]}` would happily accept it.

## Why this one has no checkpoints

Preflight warns `no-checkpoint` here. The warning is correct rather than an
oversight: this is one forward pass over a few hundred rows, there is no
progress to save, and a machine that dies costs a rerun.

The check exists because the opposite mistake — a six-hour job with nothing
resumable — is invisible until a machine dies. It is a warning and never a
refusal for exactly this case. [`../train/`](../train/) is the workload that
answers it.

## Why there is no `reduce:`

One task's `metrics.json` *is* the answer, and `reduce: {kind: aggregate,
metric: accuracy}` over a single task would report the mean of one number.

Add it the day this fans out over several holdout shards. It also changes
what the console says: the evidence moves from *"the tasks emit scores
rather than weights"* to *"the job's result is a score combined from its
tasks' outputs, not a model"*.
