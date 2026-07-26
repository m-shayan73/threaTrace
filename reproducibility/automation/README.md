# threaTrace Reproducibility Automation

This directory contains the supported automation for reproducing and
inspecting the threaTrace experiments documented under `reproducibility/`.
The runner exposes the complete experiment catalog through one
non-interactive command-line interface and creates an isolated evidence
directory for every invocation.

One invocation runs exactly one `dataset + mode + experiment` combination.
This keeps commands, failures, logs, models, and metrics attributable to one
exact configuration. Automated callers can query the JSON catalog and invoke
the required combinations in sequence.

Run all commands in this document from the root of the threaTrace repository.

## 1. Quick start

List the complete experiment catalog:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py list
```

The JSON form is intended for an orchestrator:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py list --json
```

Run a fresh StreamSpot pretrained experiment, including input download and
preprocessing:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset streamspot \
  --mode pretrained \
  --experiment baseline \
  --download
```

At completion, the runner prints the path to:

```text
reproducibility/runs/<run-id>/results/results.json
```

Read `results/report.md` first for the human-readable outcome. Use
`results/results.json` for scripts or an orchestrator.

## 2. Environment

### Conda

The manual environment setup and the dependency versions are documented in
`reproducibility/README.md`. The expected environment name is
`threatrace_env`, using Python 3.6.13 and the original threaTrace dependency
versions.

After creating the environment, verify the runner without starting an
experiment:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py list --json
```

Automatic DARPA downloads also require the `gdown` command in this same
environment:

```bash
conda run -n threatrace_env python -m pip install "gdown==4.7.3"
```

The DARPA adapter deliberately invokes the `gdown` executable beside the
active Python interpreter. This prevents a base-environment downloader from
being mixed with `threatrace_env`.

The upstream generated `run_benign.sh` and `run_attack.sh` files call the
unqualified command `python`. The automation prepends the active interpreter's
directory to `PATH`, so those scripts also use `threatrace_env` rather than the
shell's base Python.

### Docker

Build the supplied image from the repository root:

```bash
docker build -f reproducibility/Dockerfile -t threatrace .
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  threatrace bash
```

Inside the container:

```bash
python reproducibility/automation/run.py list --json
```

The Dockerfile installs the original CPU/CUDA-capable package versions and
patches the known `torch._six` incompatibility in torch-geometric.

The Dockerfile does not install `gdown`. Install it inside the container before
using automatic DARPA downloads, or manually provide the DARPA archives:

```bash
python -m pip install "gdown==4.7.3"
```

### CPU and CUDA behavior

Every catalog entry has `device_policy: cpu`. This preserves the
behavior used by the original system and the manual reproduction. Installing
CUDA-capable PyTorch packages does not make a GPU necessary, and the runner
does not provide a CUDA-selection argument.

## 3. Command-line interface

The command shape is:

```bash
python reproducibility/automation/run.py run \
  --dataset DATASET \
  --mode pretrained|scratch \
  --experiment EXPERIMENT \
  [--data-dir PATH] \
  [--download] \
  [--rebuild-data] \
  [--output-dir PATH]
```

### Required selectors

- `--dataset` selects the dataset or scene exactly as named in the catalog.
- `--mode pretrained` evaluates the model material supplied by threaTrace.
- `--mode scratch` trains new model material before evaluation.
- `--experiment` selects a defined branch such as `baseline`,
  `threshold-0`, or `no-test-validation`.

The catalog in `experiments.json` is the source of truth. An unknown
combination is rejected rather than being interpreted approximately.

### Input options

`--data-dir PATH`

: Searches a user-supplied directory for the prepared dataset, raw archives,
  extracted raw files, and, where relevant, ground truth. Supplied source
  files are not intentionally modified. Their resolved locations are recorded
  in `files/inputs/manifest.json`.

`--download`

: Downloads missing public inputs. StreamSpot and Unicorn use public HTTP
  sources. DARPA TC3 uses the Google Drive identifiers recorded in
  `experiments.json` and therefore requires `gdown`.

`--rebuild-data`

: Ignores reusable preprocessed graph files and requires raw inputs to be
  parsed again. It can be combined with `--data-dir` or `--download`.

Without `--rebuild-data`, a complete preprocessed dataset is preferred. If
neither usable prepared data nor sufficient raw data is available, the run
ends as `blocked` with an explanation.

`--output-dir PATH`

: Places new run directories under another root. This is useful because DARPA
  source data, parsed graphs, source snapshots, and trained models can require
  substantial disk space.

## 4. Experiment catalog

### Supported and intentionally unsupported combinations

| Dataset | Mode | Experiment | Catalog behavior |
|---|---|---|---|
| `streamspot` | pretrained | `baseline` | Evaluate supplied models |
| `streamspot` | scratch | `baseline` | Train and evaluate |
| `sc-1` | pretrained | `baseline` | Unsupported: no SC-1 pretrained models were supplied |
| `sc-1` | scratch | `baseline` | Train and evaluate with the retained SC-1 parser |
| `sc-2` | pretrained | `baseline` | Expected failure: supplied artifacts omit `threshold_unicorn.txt` |
| `sc-2` | pretrained | `threshold-0` | Evaluate with the documented manual threshold value 0 |
| `sc-2` | pretrained | `threshold-1` | Evaluate with the documented manual threshold value 1 |
| `sc-2` | scratch | `baseline` | Train, generate a threshold, and evaluate |
| `cadets` | pretrained | `baseline` | DARPA TC3 supplied-model evaluation |
| `cadets` | scratch | `baseline` | DARPA TC3 upstream training and evaluation |
| `cadets` | scratch | `no-test-validation` | Train without test-graph model selection, then evaluate |
| `fivedirections` | pretrained | `baseline` | DARPA TC3 supplied-model evaluation |
| `fivedirections` | scratch | `baseline` | DARPA TC3 upstream training and evaluation |
| `fivedirections` | scratch | `no-test-validation` | Train without test-graph model selection, then evaluate |
| `theia` | pretrained | `baseline` | DARPA TC3 supplied-model evaluation |
| `theia` | scratch | `baseline` | DARPA TC3 upstream training and evaluation |
| `theia` | scratch | `no-test-validation` | Train without test-graph model selection, then evaluate |
| `trace` | pretrained | `baseline` | DARPA TC3 supplied-model evaluation |
| `trace` | scratch | `baseline` | DARPA TC3 upstream training and evaluation |
| `trace` | scratch | `no-test-validation` | Train without test-graph model selection, then evaluate |
| `cadets-e5`, `fivedirections-e5`, `theia-e5`, `trace-e5` | pretrained, scratch | `baseline` | Unsupported with a recorded explanation |

### Why unsupported entries remain in the catalog

Unsupported experiments are retained so a person or orchestrator receives an
explicit reproducibility finding instead of an ambiguous “unknown dataset”
error.

- SC-1 pretrained is unsupported because the supplied Unicorn models are for
  SC-2; no SC-1 pretrained model was provided.
- DARPA TC5/E5 is unsupported because the authoritative raw-file selection is
  undocumented and the author reported that the malicious-node
  ground-truth-extraction source was lost.

Selecting one of these entries prints `UNSUPPORTED: <reason>` and returns exit
code 3 without creating a misleading numerical result.

## 5. Dataset workflows

### StreamSpot

Pretrained:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset streamspot --mode pretrained --experiment baseline --download
```

Scratch-trained:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset streamspot --mode scratch --experiment baseline --download
```

Accepted input forms are:

- `all.tar.gz`;
- extracted `all.tsv`; or
- a prepared directory containing at least 600 nonempty graph `.txt` files
  under scene directories `1` through `6`.

The adapter verifies the graph count, runs `setup.py`, initializes GraphChi,
stages or trains models, executes both generated test scripts, verifies their
result-row counts, and runs the original `evaluate_streamspot.py`.

The original evaluator classifies a graph as anomalous only when its anomaly
count is greater than 2. The automation does not change this threshold. It
also calculates and saves normalized metrics from the raw result rows using
the same threshold.

Changing the threshold solely to force numerical agreement is not part of the
baseline experiment. Raw benign and attack result rows are retained so the
reported classification can be independently verified.

### Unicorn SC-1

SC-1 supports scratch training only:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset sc-1 --mode scratch --experiment baseline --download
```

Accepted input forms are:

- all 16 `camflow-attack-*.gz.tar` and `camflow-benign-*.gz.tar` archives from
  the `camflow-apt` repository; or
- all 150 nonempty preprocessed graph files, `0.txt` through `149.txt`.

The repository's saved parser is SC-2-specific. The automation therefore
retains a dedicated SC-1 parser under `automation/experiments/` and copies it
only into the run's source snapshot. The main checkout is not patched during
the experiment.

The test evidence must contain exactly 25 benign rows and 25 attack rows.
Attack graph IDs must be 125 through 149 in order; benign IDs must be within
0 through 124.

### Unicorn SC-2

Scratch training:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset sc-2 --mode scratch --experiment baseline --download
```

The accepted input shapes and count checks are the same as SC-1, but downloads
come from the `shellshock-apt` repository and preprocessing uses the upstream
SC-2 parser.

The supplied pretrained artifacts omit `threshold_unicorn.txt`. The direct
baseline reproduces that documented failure:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset sc-2 --mode pretrained --experiment baseline --download
```

If evaluation fails at the expected stage with the expected missing-threshold
signature, the report says `EXPECTED FAILURE OBSERVED`. This is a successful
reproduction of the documented finding and the process returns 0.

The manual reproduction tried arbitrary threshold values 0 and 1. They remain
separate, explicit experiment branches:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset sc-2 --mode pretrained --experiment threshold-0 --download

conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset sc-2 --mode pretrained --experiment threshold-1 --download
```

The threshold file is created only inside that run's source/model snapshot.
The baseline and repository source are not changed.

### DARPA TC3

The four supported scenes are `cadets`, `fivedirections`, `theia`, and
`trace`. Substitute one of those names for `SCENE`.

Evaluate supplied models:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset SCENE --mode pretrained --experiment baseline --download
```

Train with the original workflow:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset SCENE --mode scratch --experiment baseline --download
```

Train without test-data validation:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset SCENE --mode scratch \
  --experiment no-test-validation --download
```

Accepted input forms are:

- the scene-specific raw archives named in `experiments.json`;
- extracted raw JSON shards; or
- nonempty `<scene>_train.txt` and `<scene>_test.txt` prepared graphs.

Ground truth is required independently of the model mode. The adapter searches
the supplied data directory and the repository ground-truth directory, then
copies a private run-scoped file to `scripts/groundtruth_uuid.txt`. This
automates a step required by pretrained evaluation but omitted from the
upstream instructions.

When raw data is parsed, all extracted JSON shards are retained and exposed to
the parser. A shard outside the direct train/test selection may still define
entities referenced by later event records. The parser is patched only inside
the run snapshot to select the requested scene and isolate its outputs.

The evaluator must produce TP, FP, TN, and FN counts plus precision, recall,
and F1. The adapter verifies that:

- the confusion counts sum to the node count in `alarm.txt`;
- the reported precision, recall, and F1 agree with those counts; and
- the required alarm, ground-truth, ID-mapping, and UUID evidence files exist.

#### Baseline versus `no-test-validation`

The upstream scratch-training flow loads the test graph during `validate()`
and deletes models that do not meet test-derived precision and recall
conditions. That is the behavior preserved by `scratch + baseline`.

The `no-test-validation` branch copies
`automation/experiments/train_darpatc_no_test_validation.py` into the isolated
source snapshot. This wrapper imports the upstream training program and
replaces its validation function with one that skips test-graph validation.
The rest of upstream training still runs, and final evaluation still uses the
test graph normally.

This is functionally aligned with the manual README experiment in which the
validation code was commented out. It is not claimed to be the exact
line-for-line manual edit because that historical patch was not retained.

## 6. Downloads and manual fallback

Use `--download` on a fresh machine. Downloads are placed inside that run's
`files/inputs/` directory and recorded in its manifest.

Google Drive may reject a valid DARPA download with:

```text
Too many users have viewed or downloaded this file recently
```

This is an external quota failure, not a model or parser failure. The run
records it at the `download` stage. Wait for the quota to reset, or manually
download every archive named for the dataset in `experiments.json`, place them
under one directory, and run:

```bash
conda run -n threatrace_env python \
  reproducibility/automation/run.py run \
  --dataset SCENE --mode pretrained --experiment baseline \
  --data-dir /absolute/path/to/downloaded/files
```

Credentials are not embedded in the runner. When access requires
authentication, provide credentials through the downloader's normal mechanism
or manually provide the files.

## 7. Run directory and retained evidence

Every supported invocation creates a unique UTC-stamped directory:

```text
reproducibility/runs/<dataset>-<mode>-<experiment>-<UTC timestamp>/
├── files/
│   ├── source/
│   │   ├── automation/       runner/catalog snapshot used by this run
│   │   ├── scripts/          run-scoped upstream and experiment source
│   │   ├── graphchi-cpp-master/
│   │   ├── example_models/
│   │   └── groundtruth/
│   ├── inputs/
│   │   └── manifest.json     resolved input paths, roles, and file sizes
│   ├── intermediate/
│   │   ├── graph_data/       prepared graph data used by the run
│   │   ├── raw/              extracted raw data where applicable
│   │   └── test_results/     copied test/evaluation evidence
│   └── models/               supplied or scratch-trained model artifacts
└── results/
    ├── report.md             human-readable status and observed metrics
    ├── results.json          machine-readable complete run record
    ├── run.log               combined commands and command output
    └── normalized_metrics.*  adapter-produced metrics where applicable
```

The source snapshot is essential evidence: it preserves the source and
experiment-specific changes actually used by the run. The runner records the
repository commit and whether the checkout was dirty at run creation.

Large inputs may be linked into the isolated layout instead of duplicated.
The input manifest records the resolved source paths. Therefore, retain any
externally supplied data for as long as its run must remain manually
inspectable.

## 8. Result interpretation

`execution_status` in `results.json` is authoritative:

| Status | Meaning |
|---|---|
| `completed` | All required stages ran successfully |
| `expected_failure` | The catalogued failure occurred at the expected stage with an expected signature |
| `unexpected_success` | An experiment catalogued to fail completed instead |
| `blocked` | Required inputs, models, downloader, or preparation material were unavailable |
| `unexpected_failure` | A command or runner failed outside the documented expected outcome |
| `running` | The run started but has not yet finalized |

Do not infer experiment success from a shell message alone. Check
`execution_status`, `message`, `observed_outcome`, and the command records in
`results.json`.

Runner exit codes are:

| Exit code | Meaning |
|---:|---|
| 0 | Completed normally, or the documented expected failure was observed |
| 1 | Unexpected command failure or interruption |
| 2 | A documented expected failure unexpectedly did not occur |
| 3 | Unsupported catalog entry or blocked run |
| 4 | Invalid CLI usage or an internal runner error |

The shell exit code is useful for orchestration, but the result JSON provides
the necessary distinction between `completed` and `expected_failure`.

Observed numerical results are displayed and retained but are not converted
into automatic pass/fail thresholds. `expected_results.json` is marked
`not_vetted`, and the research training code does not consistently set a
random seed. A completed run can therefore expose a numerical discrepancy
with the paper or manual README without becoming an execution failure.

## 9. Long-running execution

For a run that must survive SSH disconnection, start it in detached `tmux`:

```bash
tmux new-session -d -s threatrace_run \
  'cd /absolute/path/to/threaTrace && conda run -n threatrace_env python reproducibility/automation/run.py run --dataset sc-1 --mode scratch --experiment baseline --download'
```

Inspect it later:

```bash
tmux attach-session -t threatrace_run
```

For sequential automation, query `run.py list --json`, invoke `run.py run`
once for each selected catalog entry, and inspect the printed `results.json`
after every invocation. Sequential drivers should distinguish `completed`,
`expected_failure`, `blocked`, and `unexpected_failure` rather than treating
all non-completed research outcomes as equivalent.

## 10. Common messages and troubleshooting

### `=== REPORT FOR sharder() ===`

This is normal diagnostic output from GraphChi initialization. It is printed
by the upstream GraphChi binary and does not indicate failure. Use the
stage's recorded exit code to determine whether initialization succeeded.

### `rm: cannot remove 'result_*': No such file or directory`

The upstream setup script may print this when no earlier result files exist.
If `setup` records exit 0, the message is harmless.

### `ModuleNotFoundError: No module named 'torch'` from a test shell script

This means an unqualified `python` resolved to the wrong environment. The
automation addresses this by prepending the active runner
interpreter's directory to `PATH`. Run the top-level command with
`conda run -n threatrace_env python ...`.

### Google Drive quota failure

Wait and retry, or manually download the exact archive names and use
`--data-dir`. Repeating the same download immediately for every mode normally
reproduces the same external quota failure.

### Disk pressure

DARPA archives, extracted shards, parsed graphs, source snapshots, and multiple
scratch runs can consume hundreds of gigabytes. Check space before starting:

```bash
df -h .
```

Use `--output-dir` for another filesystem when appropriate. Do not delete
completed run evidence unless it has been deliberately archived elsewhere.

### Interrupted run

The runner catches SIGTERM and records an unexpected interruption when
possible. A machine crash, forced process kill, or full filesystem can prevent
finalization. In that case, inspect `results/results.json`, `results/run.log`,
and the stage artifacts before deciding whether the experiment must restart.

## 11. Implementation map

- `run.py`: stable CLI, catalog lookup, source snapshot creation, adapter
  dispatch, and final metric parsing.
- `experiments.json`: authoritative experiment matrix, expected execution
  behavior, download identifiers, and unsupported explanations.
- `expected_results.json`: numerical-comparison metadata. It is marked
  `not_vetted`, so results are reported without an automatic agreement gate.
- `runner_common.py`: run lifecycle, safe archive extraction, downloads,
  command logging, input manifest, statuses, reports, and exit codes.
- `adapter_common.py`: shared upstream setup and GraphChi environment setup.
- `adapter_streamspot.py`: StreamSpot input preparation, model handling,
  testing, and normalized evaluation.
- `adapter_unicornsc.py`: SC-1/SC-2 acquisition, parsing, threshold branches,
  training, test validation, and normalized evaluation.
- `adapter_darpa.py`: TC3 raw/preprocessed data handling, ground truth, model
  staging/training, evaluation, and evidence validation.
- `experiments/parse_unicornsc_sc1.py`: retained SC-1-specific parser used only
  in run snapshots.
- `experiments/train_darpatc_no_test_validation.py`: retained DARPA training
  branch that skips test-graph validation.

The original repository source is not edited during an experiment. Required
parser adjustments, threshold files, and alternate experiment sources are
applied only inside `files/source/` for that run and remain available for
manual verification.
