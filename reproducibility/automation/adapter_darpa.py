"""Shared threaTrace adapter for DARPA TC3 scenes.

All parser and experiment changes are applied to the source snapshot retained
inside a run.  The repository's upstream scripts are never changed by this
adapter.
"""

import importlib.util
import re
import shutil
import sys
import tarfile
from pathlib import Path

from runner_common import (
    BlockedRun,
    CommandFailure,
    download_http,
    find_named as _find_named,
    link_path,
    nonempty_file as _nonempty,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_DIR = Path(__file__).resolve().parent
NO_TEST_VALIDATION_SOURCE = (
    AUTOMATION_DIR / "experiments" / "train_darpatc_no_test_validation.py"
)

GROUNDTRUTH_URL = (
    "https://raw.githubusercontent.com/threaTrace-detector/threaTrace/"
    "master/groundtruth/{dataset}.txt"
)

SCENES = {
    "cadets": {
        "archives": [
            (
                "ta1-cadets-e3-official.json.tar.gz",
                "1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb",
            ),
            (
                "ta1-cadets-e3-official-2.json.tar.gz",
                "1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1",
            ),
        ],
        "parser_roots": [
            "ta1-cadets-e3-official.json",
            "ta1-cadets-e3-official-2.json",
        ],
        "train_member": "ta1-cadets-e3-official.json.1.txt",
        "test_member": "ta1-cadets-e3-official-2.json.txt",
        "pretrained_models": 22,
    },
    "fivedirections": {
        "archives": [
            (
                "ta1-fivedirections-e3-official-2.json.tar.gz",
                "1BeP80zUUmm4eZl0UuU43PsKNkl_xgskj",
            ),
        ],
        "parser_roots": ["ta1-fivedirections-e3-official-2.json"],
        "train_member": "ta1-fivedirections-e3-official-2.json.txt",
        "test_member": "ta1-fivedirections-e3-official-2.json.23.txt",
        "pretrained_models": 20,
    },
    "theia": {
        "archives": [
            (
                "ta1-theia-e3-official-1r.json.tar.gz",
                "10cecNtR3VsHfV0N-gNEeoVeB89kCnse5",
            ),
            (
                "ta1-theia-e3-official-6r.json.tar.gz",
                "1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0",
            ),
        ],
        "parser_roots": [
            "ta1-theia-e3-official-1r.json",
            "ta1-theia-e3-official-6r.json",
        ],
        "train_member": "ta1-theia-e3-official-1r.json.txt",
        "test_member": "ta1-theia-e3-official-6r.json.8.txt",
        "pretrained_models": 10,
    },
    "trace": {
        "archives": [
            (
                "ta1-trace-e3-official-1.json.tar.gz",
                "1GG1aUnPjjzzdbxznVTN8X6oVfA-K4oIV",
            ),
        ],
        "parser_roots": ["ta1-trace-e3-official-1.json"],
        "train_member": "ta1-trace-e3-official-1.json.txt",
        "test_member": "ta1-trace-e3-official-1.json.4.txt",
        "pretrained_models": 10,
    },
}

NO_TEST_VALIDATION_EXPERIMENTS = {
    "no-test-validation",
    "without-test-validation",
    "no_test_validation",
}


def _safe_extract_tar(archive, destination):
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    try:
        with tarfile.open(str(archive), "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                if member.issym() or member.islnk():
                    raise BlockedRun(
                        "DARPA archive contains a link member, which is not "
                        "accepted: {}".format(member.name)
                    )
                target = (base / member.name).resolve()
                if target != base and base not in target.parents:
                    raise BlockedRun(
                        "DARPA archive contains an unsafe path: {}".format(
                            member.name
                        )
                    )
            handle.extractall(str(destination), members=members)
    except (OSError, tarfile.TarError) as error:
        raise BlockedRun(
            "Could not extract DARPA archive {}: {}".format(archive, error)
        )


def _archive_definitions(spec, dataset):
    configured = spec.get("raw_archives")
    if not configured:
        return list(SCENES[dataset]["archives"])
    result = []
    for item in configured:
        if not isinstance(item, dict) or not item.get("filename"):
            raise BlockedRun(
                "raw_archives entries must provide filename and drive_id or url."
            )
        location = item.get("drive_id") or item.get("url")
        if not location:
            raise BlockedRun(
                "No download location is configured for {}.".format(
                    item["filename"]
                )
            )
        result.append((item["filename"], location))
    return result


def _download_gdrive(context, location, destination):
    if importlib.util.find_spec("gdown") is None:
        raise BlockedRun(
            "Automatic DARPA downloads require gdown in the active environment. "
            "Install gdown or manually download the required archives and pass "
            "their directory with --data-dir."
        )
    location = str(location)
    match = re.search(r"/file/d/([^/]+)", location)
    file_id = match.group(1) if match else location
    gdown_executable = Path(sys.executable).with_name("gdown")
    if not gdown_executable.is_file():
        raise BlockedRun(
            "The active environment provides the gdown package but not its "
            "console executable: {}.".format(gdown_executable)
        )
    context.run_command(
        [
            gdown_executable,
            "--id",
            file_id,
            "--output",
            destination,
        ],
        context.inputs_dir,
        "download",
    )
    if not _nonempty(destination):
        raise BlockedRun(
            "Google Drive did not produce a nonempty archive: {}. Download it "
            "manually and pass its directory with --data-dir.".format(
                Path(destination).name
            )
        )


def _download_archive(context, location, destination):
    location = str(location)
    if "drive.google.com" in location or not location.startswith(
        ("http://", "https://")
    ):
        _download_gdrive(context, location, destination)
    else:
        download_http(location, destination)
    return destination


def _stage_matching_shards(source_root, raw_stage, parser_root):
    source_root = Path(source_root)
    candidates = []
    if source_root.is_dir():
        candidates = sorted(
            (
                item
                for item in source_root.rglob(parser_root + "*")
                if item.is_file()
                and not item.name.endswith((".tar.gz", ".txt"))
            ),
            key=lambda item: str(item),
        )
    elif source_root.is_file() and source_root.name.startswith(parser_root):
        candidates = [source_root]
    for source in candidates:
        destination = raw_stage / source.name
        if not destination.exists():
            link_path(source, destination)


def _direct_shard(raw_stage, scripts, name):
    source = _find_named(raw_stage, name)
    if not _nonempty(source):
        return False
    link_path(source, scripts / name)
    return True


def _required_raw_names(scene):
    config = SCENES[scene]
    return [
        config["train_member"][:-4],
        config["test_member"][:-4],
    ]


def _patch_scene_parser(parser_path, dataset):
    config = SCENES[dataset]
    text = Path(parser_path).read_text()

    text, archive_count = re.subn(
        r"^os\.system\('tar -zxvf .*?'\)\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    if archive_count != 6:
        raise BlockedRun(
            "Cannot isolate the DARPA parser: expected six archive commands, "
            "found {}.".format(archive_count)
        )

    roots = "path_list = {!r}".format(config["parser_roots"])
    text, path_count = re.subn(
        r"^path_list = \[.*\]\s*$",
        roots,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if path_count != 1:
        raise BlockedRun(
            "Cannot isolate the DARPA parser's path_list for {}.".format(dataset)
        )

    text, copy_count = re.subn(
        r"^os\.system\('cp ta1-.*?'\)\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    if copy_count != 8:
        raise BlockedRun(
            "Cannot isolate DARPA parser outputs: expected eight copy commands, "
            "found {}.".format(copy_count)
        )

    output_commands = (
        "os.system('cp {} ../graphchi-cpp-master/graph_data/darpatc/"
        "{}_train.txt')\n"
        "os.system('cp {} ../graphchi-cpp-master/graph_data/darpatc/"
        "{}_test.txt')\n"
        "os.system('rm ta1-*')"
    ).format(
        config["train_member"],
        dataset,
        config["test_member"],
        dataset,
    )
    old_cleanup = "os.system('rm ta1-*')"
    if text.count(old_cleanup) != 1:
        raise BlockedRun(
            "Cannot isolate DARPA parser cleanup: expected one cleanup command."
        )
    Path(parser_path).write_text(text.replace(old_cleanup, output_commands, 1))


def _reuse_preprocessed(context, data_path, target, dataset):
    if not data_path.exists():
        return False
    train = _find_named(data_path, "{}_train.txt".format(dataset))
    test = _find_named(data_path, "{}_test.txt".format(dataset))
    if not (_nonempty(train) and _nonempty(test)):
        return False
    context.record_input(train, "preprocessed {} training graph".format(dataset))
    context.record_input(test, "preprocessed {} testing graph".format(dataset))
    link_path(train, target / "{}_train.txt".format(dataset))
    link_path(test, target / "{}_test.txt".format(dataset))
    return True


def _prepare_raw_data(context, args, data_path, target, dataset):
    config = SCENES[dataset]
    scripts = context.source_dir / "scripts"
    raw_stage = context.intermediate_dir / "raw" / dataset
    raw_stage.mkdir(parents=True, exist_ok=True)

    for parser_root in config["parser_roots"]:
        if data_path.exists():
            _stage_matching_shards(data_path, raw_stage, parser_root)

    for filename, location in _archive_definitions(context.spec, dataset):
        archive = _find_named(data_path, filename) if data_path.exists() else None
        if not _nonempty(archive) and getattr(args, "download", False):
            archive = context.inputs_dir / filename
            context.log("Downloading DARPA archive {}".format(filename))
            _download_archive(context, location, archive)
        if _nonempty(archive):
            context.record_input(archive, "{} raw archive".format(dataset))
            _safe_extract_tar(archive, raw_stage)

    for parser_root in config["parser_roots"]:
        # Normalize archives with an enclosing directory to the names expected
        # by the upstream parser.
        matches = sorted(
            (
                item
                for item in raw_stage.rglob(parser_root + "*")
                if item.is_file()
                and not item.name.endswith((".tar.gz", ".txt"))
            ),
            key=lambda item: str(item),
        )
        for source in matches:
            destination = raw_stage / source.name
            if source != destination and not destination.exists():
                link_path(source, destination)

    missing = [
        name
        for name in _required_raw_names(dataset)
        if not _direct_shard(raw_stage, scripts, name)
    ]
    if missing:
        required_archives = ", ".join(
            name for name, _ in _archive_definitions(context.spec, dataset)
        )
        raise BlockedRun(
            "{} raw input is incomplete; missing {}. Supply extracted shards "
            "or archives ({}) with --data-dir, or use --download.".format(
                dataset, ", ".join(missing), required_archives
            )
        )

    # Link every shard, not only the selected split, because the parser's first
    # pass uses them to resolve entity types referenced by event records.
    for parser_root in config["parser_roots"]:
        for source in sorted(
            raw_stage.glob(parser_root + "*"), key=lambda item: str(item)
        ):
            if source.is_file() and not source.name.endswith(".txt"):
                link_path(source, scripts / source.name)

    parser_path = scripts / "parse_darpatc.py"
    _patch_scene_parser(parser_path, dataset)
    context.run_command(
        [sys.executable, "parse_darpatc.py"],
        scripts,
        "preprocess",
    )
    train = target / "{}_train.txt".format(dataset)
    test = target / "{}_test.txt".format(dataset)
    missing_outputs = [
        path.name for path in (train, test) if not _nonempty(path)
    ]
    if missing_outputs:
        raise CommandFailure(
            "preprocess",
            [sys.executable, "parse_darpatc.py"],
            1,
            "DARPA parser did not produce nonempty outputs: {}".format(
                ", ".join(missing_outputs)
            ),
        )


def _prepare_data(context, args, dataset):
    configured = (
        Path(args.data_dir).expanduser().resolve()
        if getattr(args, "data_dir", None)
        else None
    )
    data_path = configured or (
        REPO_ROOT / "graphchi-cpp-master" / "graph_data" / "darpatc"
    )
    target = context.intermediate_dir / "graph_data" / "darpatc"
    target.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "rebuild_data", False):
        if _reuse_preprocessed(context, data_path, target, dataset):
            return
    _prepare_raw_data(context, args, data_path, target, dataset)


def _prepare_groundtruth(context, args, data_path, dataset):
    candidates = []
    if data_path is not None and data_path.exists():
        candidates.append(_find_named(data_path, "{}.txt".format(dataset)))
    candidates.extend(
        [
            context.source_dir / "groundtruth" / "{}.txt".format(dataset),
            REPO_ROOT / "groundtruth" / "{}.txt".format(dataset),
        ]
    )
    source = next((path for path in candidates if _nonempty(path)), None)
    if source is None and getattr(args, "download", False):
        source = context.inputs_dir / "{}.txt".format(dataset)
        download_http(
            context.spec.get(
                "groundtruth_url",
                GROUNDTRUTH_URL.format(dataset=dataset),
            ),
            source,
        )
    if not _nonempty(source):
        raise BlockedRun(
            "{} ground truth is missing. Supply {}.txt with --data-dir or use "
            "--download.".format(dataset, dataset)
        )
    context.record_input(source, "{} ground truth".format(dataset))
    destination = context.source_dir / "scripts" / "groundtruth_uuid.txt"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    # Use a private copy because train_darpatc.py copies the repository
    # ground-truth file onto this path.  A symlink here could make that upstream
    # command overwrite a user-supplied --data-dir file.
    shutil.copy2(str(source), str(destination))


def _run_setup(context):
    context.run_command(
        [sys.executable, "setup.py"],
        context.source_dir / "scripts",
        "setup",
    )


def _stage_pretrained(context, dataset):
    source = context.source_dir / "example_models" / "darpatc" / dataset
    config = SCENES[dataset]
    required = ["feature.txt", "label.txt"] + [
        "model_{}".format(index)
        for index in range(config["pretrained_models"])
    ]
    missing = [
        name for name in required if not _nonempty(source / name)
    ]
    if missing:
        raise BlockedRun(
            "Supplied {} pretrained models are incomplete: {}.".format(
                dataset, ", ".join(missing)
            )
        )
    _run_setup(context)
    for name in required:
        shutil.copy2(str(source / name), str(context.models_dir / name))
    context.record_input(source, "threaTrace supplied {} models".format(dataset))


def _stage_no_test_validation_experiment(context):
    if not _nonempty(NO_TEST_VALIDATION_SOURCE):
        raise BlockedRun(
            "Missing no-test-validation experiment source: {}".format(
                NO_TEST_VALIDATION_SOURCE
            )
        )
    destination = (
        context.source_dir
        / "scripts"
        / "train_darpatc_no_test_validation.py"
    )
    shutil.copy2(str(NO_TEST_VALIDATION_SOURCE), str(destination))
    return destination


def _train_scratch(context, dataset, experiment):
    scripts = context.source_dir / "scripts"
    if experiment == "baseline":
        command = [sys.executable, "train_darpatc.py", "--scene", dataset]
    elif experiment in NO_TEST_VALIDATION_EXPERIMENTS:
        _stage_no_test_validation_experiment(context)
        command = [
            sys.executable,
            "train_darpatc_no_test_validation.py",
            "--scene",
            dataset,
        ]
    else:
        raise BlockedRun(
            "Scratch DARPA mode does not support experiment {!r}.".format(
                experiment
            )
        )
    context.run_command(command, scripts, "train")


def _validate_models(context, dataset, mode):
    required = ("feature.txt", "label.txt")
    missing = [
        name for name in required if not _nonempty(context.models_dir / name)
    ]
    models = sorted(
        (
            path
            for path in context.models_dir.glob("model_[0-9]*")
            if _nonempty(path)
        ),
        key=lambda item: str(item),
    )
    if missing or not models:
        details = list(missing)
        if not models:
            details.append("model_0 or another nonempty model_N")
        raise CommandFailure(
            "train" if mode == "scratch" else "setup",
            ["validate-models", dataset],
            1,
            "{} model preparation is incomplete: {}.".format(
                dataset, ", ".join(details)
            ),
        )


def _metric_value(output, label):
    matches = re.findall(
        r"^{}\s*:\s*([0-9.eE+-]+)\s*$".format(re.escape(label)),
        output,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_evaluation(output):
    count_matches = re.findall(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        output,
        flags=re.MULTILINE,
    )
    if not count_matches:
        raise ValueError("missing TP/FP/TN/FN count line")
    tp, fp, tn, fn = [int(value) for value in count_matches[-1]]
    metrics = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _metric_value(output, "Precision"),
        "recall": _metric_value(output, "Recall"),
        "f1": _metric_value(output, "F-Score"),
    }
    missing = [
        name
        for name in ("precision", "recall", "f1")
        if metrics[name] is None
    ]
    if missing:
        raise ValueError(
            "missing normalized metrics: {}".format(", ".join(missing))
        )
    return metrics


def _validate_metric_consistency(metrics, alarm_path):
    try:
        first_line = Path(alarm_path).read_text().splitlines()[0]
        total = int(first_line)
    except (IndexError, OSError, ValueError):
        raise ValueError("alarm.txt does not begin with a node count")
    observed = (
        metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    )
    if observed != total:
        raise ValueError(
            "evaluation counts sum to {}, but alarm.txt reports {} nodes".format(
                observed, total
            )
        )
    eps = 1e-10
    precision = metrics["tp"] / (metrics["tp"] + metrics["fp"] + eps)
    recall = metrics["tp"] / (metrics["tp"] + metrics["fn"] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    for name, expected in (
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
    ):
        if abs(metrics[name] - expected) > 1e-8:
            raise ValueError(
                "{}={} is inconsistent with counts (expected {})".format(
                    name, metrics[name], expected
                )
            )


def _append_normalized_metrics(context, metrics, evidence_dir):
    labels = [
        ("TP", metrics["tp"]),
        ("FP", metrics["fp"]),
        ("TN", metrics["tn"]),
        ("FN", metrics["fn"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("F-Score", metrics["f1"]),
    ]
    text = "\n".join("{}: {}".format(name, value) for name, value in labels) + "\n"
    with context.log_path.open("a") as handle:
        handle.write(text)
    sys.stdout.write(text)
    sys.stdout.flush()
    (evidence_dir / "normalized_metrics.txt").write_text(text)


def _evaluate(context, dataset):
    scripts = context.source_dir / "scripts"
    test_command = [sys.executable, "test_darpatc.py", "--scene", dataset]
    context.run_command(test_command, scripts, "test")
    required_test_files = (
        "alarm.txt",
        "groundtruth_uuid.txt",
        "groundtruth_nodeId.txt",
        "id_to_uuid.txt",
    )
    missing = [
        name for name in required_test_files if not _nonempty(scripts / name)
    ]
    if missing:
        raise CommandFailure(
            "test",
            test_command,
            1,
            "{} testing omitted required artifacts: {}.".format(
                dataset, ", ".join(missing)
            ),
        )

    evaluate_command = [sys.executable, "evaluate_darpatc.py"]
    output = context.run_command(evaluate_command, scripts, "evaluate")
    try:
        metrics = _parse_evaluation(output)
        _validate_metric_consistency(metrics, scripts / "alarm.txt")
    except ValueError as error:
        raise CommandFailure(
            "evaluate",
            evaluate_command,
            1,
            "Invalid DARPA evaluation output: {}".format(error),
        )

    evidence_dir = context.intermediate_dir / "test_results"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in required_test_files:
        shutil.copy2(str(scripts / name), str(evidence_dir / name))
    _append_normalized_metrics(context, metrics, evidence_dir)


def run_darpa(context, args):
    """Run one DARPA TC3 experiment selected by the catalog."""
    dataset = str(context.spec["dataset"]).lower()
    if dataset not in SCENES:
        raise BlockedRun(
            "DARPA adapter does not support dataset {!r}.".format(dataset)
        )
    mode = context.spec["mode"]
    experiment = context.spec["experiment"]
    if mode == "pretrained" and experiment != "baseline":
        raise BlockedRun(
            "Pretrained DARPA mode supports only the baseline experiment."
        )
    if mode == "scratch" and (
        experiment != "baseline"
        and experiment not in NO_TEST_VALIDATION_EXPERIMENTS
    ):
        raise BlockedRun(
            "Unknown scratch DARPA experiment {!r}.".format(experiment)
        )

    configured = (
        Path(args.data_dir).expanduser().resolve()
        if getattr(args, "data_dir", None)
        else None
    )
    _prepare_data(context, args, dataset)
    _prepare_groundtruth(context, args, configured, dataset)

    if mode == "pretrained":
        _stage_pretrained(context, dataset)
    elif mode == "scratch":
        _train_scratch(context, dataset, experiment)
    else:
        raise BlockedRun(
            "DARPA adapter does not support mode {!r}.".format(mode)
        )
    _validate_models(context, dataset, mode)
    _evaluate(context, dataset)
