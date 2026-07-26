"""Unicorn SC-1/SC-2 workflow adapter for the threaTrace runner.

The central runner owns the run directory and source snapshot.  This adapter
prepares one Unicorn dataset inside that snapshot, runs the upstream
training/testing workflow, and emits a normalized metric summary without
modifying the repository's original experiment sources.
"""

import json
import re
import shutil
import stat
import sys
from pathlib import Path

from adapter_common import initialize_graphchi, run_setup
from runner_common import (
    BlockedRun,
    CommandFailure,
    download_http,
    find_named as _find_named,
    link_path,
    nonempty_file as _nonempty_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_DIR = Path(__file__).resolve().parent
SC1_PARSER = AUTOMATION_DIR / "experiments" / "parse_unicornsc_sc1.py"

SUPPORTED_DATASETS = ("sc-1", "sc-2")
GRAPH_IDS = tuple(range(150))
ATTACK_IDS = tuple(range(125, 150))
ARCHIVES = tuple(
    ("attack", "camflow-attack-{}.gz.tar".format(index))
    for index in range(3)
) + tuple(
    ("benign", "camflow-benign-{}.gz.tar".format(index))
    for index in range(13)
)
DEFAULT_REPOSITORIES = {
    "sc-1": "camflow-apt",
    "sc-2": "shellshock-apt",
}
THRESHOLD_EXPERIMENTS = {
    "threshold-0": 0,
    "threshold-1": 1,
}
RESULT_PATTERN = re.compile(
    r"^\s*(?P<graph_id>[0-9]+)\s+finished\.\s+fp:\s+"
    r"(?P<anomaly_count>[0-9]+)\b"
)
COMMAND_GRAPH_PATTERN = re.compile(
    r"(?:^|\s)test_unicornsc\.py\s+\S+\s+\S+\s+(?P<graph_id>[0-9]+)(?:\s|$)"
)


def _complete_preprocessed_directory(root):
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = [root]
    candidates.extend(
        sorted(
            {path.parent for path in root.rglob("0.txt")},
            key=lambda item: str(item),
        )
    )
    for candidate in candidates:
        complete = all(
            _nonempty_file(candidate / "{}.txt".format(index))
            for index in GRAPH_IDS
        )
        if complete:
            return candidate
    return None


def _download_url(context, dataset, role, filename):
    configured = context.spec.get("raw_files", {})
    if filename in configured:
        return configured[filename]
    repository = context.spec.get(
        "data_repository", DEFAULT_REPOSITORIES[dataset]
    )
    return (
        "https://github.com/margoseltzer/{}/raw/master/{}/{}"
    ).format(repository, role, filename)


def _acquire_archives(context, args, dataset, data_path):
    archives = {}
    for role, filename in ARCHIVES:
        candidate = _find_named(data_path, filename)
        if _nonempty_file(candidate):
            archives[filename] = candidate

    if len(archives) != len(ARCHIVES) and getattr(args, "download", False):
        for role, filename in ARCHIVES:
            if filename in archives:
                continue
            destination = context.inputs_dir / dataset / filename
            context.log("Downloading Unicorn archive {}".format(filename))
            try:
                download_http(
                    _download_url(context, dataset, role, filename),
                    destination,
                )
            except Exception as error:
                raise BlockedRun(
                    "Could not download {} for {}: {}".format(
                        filename, dataset, error
                    )
                )
            archives[filename] = destination

    missing = [
        filename for _, filename in ARCHIVES if filename not in archives
    ]
    if missing:
        raise BlockedRun(
            "{} raw data is incomplete. Missing nonempty archives: {}. "
            "Supply all 16 archives with --data-dir or use --download.".format(
                dataset, ", ".join(missing)
            )
        )
    return archives


def _validate_preprocessed_graphs(directory):
    missing = [
        "{}.txt".format(index)
        for index in GRAPH_IDS
        if not _nonempty_file(Path(directory) / "{}.txt".format(index))
    ]
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += ", ..."
        raise BlockedRun(
            "Unicorn preprocessing did not produce all 150 nonempty graph "
            "files. Missing/empty: {}.".format(preview)
        )


def _stage_sc1_parser(context):
    if not SC1_PARSER.is_file():
        raise BlockedRun(
            "The dedicated SC-1 parser is missing: {}".format(SC1_PARSER)
        )
    destination = context.source_dir / "scripts" / SC1_PARSER.name
    shutil.copy2(str(SC1_PARSER), str(destination))
    text = destination.read_text()
    if "open('camflow-normal.txt.'" in text or (
        "os.system('rm camflow-normal.txt.'" in text
    ):
        raise BlockedRun(
            "The dedicated SC-1 parser still references SC-2 camflow-normal "
            "fragments."
        )
    required_fragments = (
        "open('camflow-benign.txt.'",
        "os.system('rm camflow-benign.txt.'",
    )
    if not all(fragment in text for fragment in required_fragments):
        raise BlockedRun(
            "The dedicated SC-1 parser is missing a documented benign "
            "fragment reference."
        )
    return destination


def _prepare_data(context, args, dataset):
    configured = (
        Path(args.data_dir).expanduser()
        if getattr(args, "data_dir", None)
        else None
    )
    data_path = (
        configured
        if configured is not None
        else REPO_ROOT / "graphchi-cpp-master" / "graph_data" / "unicornsc"
    )
    target = context.intermediate_dir / "graph_data" / "unicornsc"

    reusable = None
    if not getattr(args, "rebuild_data", False):
        reusable = _complete_preprocessed_directory(data_path)
    if reusable is not None:
        context.record_input(
            reusable, "{} preprocessed Unicorn graph directory".format(dataset)
        )
        link_path(reusable, target)
        _validate_preprocessed_graphs(target)
        return

    archives = _acquire_archives(context, args, dataset, data_path)
    target.mkdir(parents=True, exist_ok=True)
    for _, filename in ARCHIVES:
        source = archives[filename]
        context.record_input(
            source, "{} Unicorn raw archive".format(dataset)
        )
        link_path(source, target / filename)

    scripts = context.source_dir / "scripts"
    if dataset == "sc-1":
        parser = _stage_sc1_parser(context)
    else:
        parser = scripts / "parse_unicornsc.py"
        if not parser.is_file():
            raise BlockedRun(
                "The source snapshot is missing scripts/parse_unicornsc.py."
            )
    context.run_command(
        [sys.executable, parser.name],
        scripts,
        "preprocess",
    )
    _validate_preprocessed_graphs(target)


def _clean_generated_controls(context):
    scripts = context.source_dir / "scripts"
    generated = (
        "models_list.txt",
        "threshold_unicorn.txt",
        "run_benign.sh",
        "run_attack.sh",
        "result_benign.txt",
        "result_attack.txt",
        "pid.txt",
    )
    for name in generated:
        path = scripts / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _run_setup(context):
    run_setup(context)
    _clean_generated_controls(context)


def _required_control_files(scripts):
    required = (
        "models_list.txt",
        "run_benign.sh",
        "run_attack.sh",
    )
    missing = [name for name in required if not _nonempty_file(scripts / name)]
    if missing:
        raise BlockedRun(
            "Unicorn model preparation did not produce: {}.".format(
                ", ".join(missing)
            )
        )


def _stage_pretrained(context):
    source = context.source_dir / "example_models" / "unicornsc"
    scripts = context.source_dir / "scripts"
    _required_control_files(source)
    for name in ("feature.txt", "label.txt"):
        if not _nonempty_file(source / name):
            raise BlockedRun(
                "Pretrained Unicorn material is missing {}.".format(name)
            )

    model_ids = [
        line.strip()
        for line in (source / "models_list.txt").read_text().splitlines()
        if line.strip()
    ]
    if not model_ids:
        raise BlockedRun("Pretrained Unicorn models_list.txt is empty.")
    missing_models = [
        "{}_0".format(model_id)
        for model_id in model_ids
        if not _nonempty_file(source / "{}_0".format(model_id))
    ]
    if missing_models:
        raise BlockedRun(
            "Pretrained Unicorn base models are missing: {}.".format(
                ", ".join(missing_models)
            )
        )

    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(context.models_dir / item.name))
    for name in ("models_list.txt", "run_benign.sh", "run_attack.sh"):
        shutil.copy2(str(source / name), str(scripts / name))
    context.record_input(
        source, "threaTrace supplied Unicorn SC-2 pretrained models"
    )


def _prepare_scratch(context, environment):
    scripts = context.source_dir / "scripts"
    context.run_command(
        [sys.executable, "train_unicornsc.py"],
        scripts,
        "train",
        env=environment,
    )
    _required_control_files(scripts)
    threshold = scripts / "threshold_unicorn.txt"
    if not _nonempty_file(threshold):
        raise BlockedRun(
            "Scratch Unicorn training did not produce threshold_unicorn.txt."
        )


def _threshold_override(context):
    configured = context.spec.get("threshold_override")
    if configured is not None:
        try:
            return int(configured)
        except (TypeError, ValueError):
            raise BlockedRun(
                "threshold_override must be an integer, got {!r}.".format(
                    configured
                )
            )
    return THRESHOLD_EXPERIMENTS.get(context.spec["experiment"])


def _stage_threshold_override(context):
    threshold = _threshold_override(context)
    if threshold is None:
        return
    scripts_path = context.source_dir / "scripts" / "threshold_unicorn.txt"
    models_path = context.models_dir / "threshold_unicorn.txt"
    content = "{}\n".format(threshold)
    scripts_path.write_text(content)
    models_path.write_text(content)
    context.record_input(
        scripts_path,
        "documented manual Unicorn evaluation threshold override",
    )


def _script_commands(path):
    commands = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.strip() != "set -e"
    ]
    graph_ids = []
    for command in commands:
        match = COMMAND_GRAPH_PATTERN.search(command)
        if match is None:
            raise BlockedRun(
                "{} contains an unsupported test command: {}".format(
                    Path(path).name, command
                )
            )
        graph_ids.append(int(match.group("graph_id")))
    return commands, graph_ids


def _validate_test_graph_ids(role, graph_ids, script_name):
    if len(set(graph_ids)) != len(graph_ids):
        raise BlockedRun(
            "{} contains duplicate graph IDs.".format(script_name)
        )
    if role == "attack":
        if graph_ids != list(ATTACK_IDS):
            raise BlockedRun(
                "{} must test attack graphs 125 through 149 in order.".format(
                    script_name
                )
            )
    elif any(graph_id < 0 or graph_id >= 125 for graph_id in graph_ids):
        raise BlockedRun(
            "{} contains an ID outside the benign range 0 through 124.".format(
                script_name
            )
        )


def _parse_result_rows(path, expected_graph_ids):
    path = Path(path)
    if not path.is_file():
        raise CommandFailure(
            "test",
            ["bash", path.name],
            1,
            "{} was not produced.".format(path.name),
        )
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) != 25:
        raise CommandFailure(
            "test",
            ["bash", path.name],
            1,
            "{} contains {} result rows; expected exactly 25.".format(
                path.name, len(lines)
            ),
        )
    rows = []
    for line_number, line in enumerate(lines, 1):
        match = RESULT_PATTERN.match(line)
        if match is None:
            raise CommandFailure(
                "test",
                ["bash", path.name],
                1,
                "{} row {} has an unexpected format: {}".format(
                    path.name, line_number, line
                ),
            )
        rows.append(
            (
                int(match.group("graph_id")),
                int(match.group("anomaly_count")),
            )
        )
    observed_graph_ids = [row[0] for row in rows]
    if observed_graph_ids != list(expected_graph_ids):
        raise CommandFailure(
            "test",
            ["bash", path.name],
            1,
            "{} graph IDs do not match its test commands. Expected {}; "
            "observed {}.".format(
                path.name, list(expected_graph_ids), observed_graph_ids
            ),
        )
    return rows


def _execute_tests(context, environment):
    scripts = context.source_dir / "scripts"
    output_dir = context.intermediate_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_class = {}
    test_files = (
        ("benign", "run_benign.sh", "result_benign.txt"),
        ("attack", "run_attack.sh", "result_attack.txt"),
    )
    for role, script_name, result_name in test_files:
        script_path = scripts / script_name
        if not _nonempty_file(script_path):
            raise BlockedRun(
                "{} was not produced by model preparation.".format(script_name)
            )
        commands, expected_graph_ids = _script_commands(script_path)
        if len(commands) != 25:
            raise BlockedRun(
                "{} contains {} test commands; expected exactly 25.".format(
                    script_name, len(commands)
                )
            )
        _validate_test_graph_ids(role, expected_graph_ids, script_name)
        result_path = scripts / result_name
        if result_path.exists():
            result_path.unlink()
        original = script_path.read_text()
        if not original.startswith("set -e\n"):
            script_path.write_text("set -e\n" + original)
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
        context.run_command(
            ["bash", script_name],
            scripts,
            "test",
            env=environment,
        )
        rows_by_class[role] = _parse_result_rows(
            result_path, expected_graph_ids
        )
        shutil.copy2(str(result_path), str(output_dir / result_name))
        shutil.copy2(str(script_path), str(output_dir / script_name))
    return rows_by_class


def _read_threshold(context):
    path = context.source_dir / "scripts" / "threshold_unicorn.txt"
    if not _nonempty_file(path):
        raise BlockedRun("threshold_unicorn.txt is missing after evaluation.")
    lines = path.read_text().splitlines()
    if not lines:
        raise BlockedRun("threshold_unicorn.txt is empty.")
    try:
        return int(lines[0].strip())
    except ValueError:
        raise BlockedRun(
            "threshold_unicorn.txt does not begin with an integer threshold."
        )


def _normalized_metrics(rows_by_class, threshold):
    fp = sum(value > threshold for _, value in rows_by_class["benign"])
    tn = len(rows_by_class["benign"]) - fp
    tp = sum(value > threshold for _, value in rows_by_class["attack"])
    fn = len(rows_by_class["attack"]) - tp
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    fpr = fp / (fp + tn + eps)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "threshold": threshold,
    }


def _emit_normalized_metrics(context, metrics):
    ordered = (
        ("TP", metrics["tp"]),
        ("TN", metrics["tn"]),
        ("FP", metrics["fp"]),
        ("FN", metrics["fn"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F-Score", metrics["f1"]),
        ("Accuracy", metrics["accuracy"]),
        ("FPR", metrics["fpr"]),
        ("Threshold", metrics["threshold"]),
    )
    with context.log_path.open("a") as handle:
        for label, value in ordered:
            line = "{}: {}".format(label, value)
            print(line)
            handle.write(line + "\n")
    metrics_path = context.results_dir / "normalized_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _validate_combination(context):
    dataset = context.spec["dataset"]
    mode = context.spec["mode"]
    experiment = context.spec["experiment"]
    if dataset not in SUPPORTED_DATASETS:
        raise BlockedRun(
            "The Unicorn adapter does not support dataset {!r}.".format(dataset)
        )
    if dataset == "sc-1" and mode != "scratch":
        raise BlockedRun(
            "SC-1 has no dataset-specific supplied pretrained models."
        )
    if mode == "scratch" and experiment != "baseline":
        raise BlockedRun(
            "Scratch Unicorn runs support only experiment='baseline'."
        )
    if mode == "pretrained" and experiment not in (
        "baseline",
        "threshold-0",
        "threshold-1",
    ):
        raise BlockedRun(
            "Pretrained Unicorn runs support baseline, threshold-0, or "
            "threshold-1."
        )


def run_unicornsc(context, args):
    """Run one catalog-selected Unicorn SC experiment."""
    _validate_combination(context)
    dataset = context.spec["dataset"]
    _prepare_data(context, args, dataset)
    _run_setup(context)
    environment = initialize_graphchi(
        context,
        context.source_dir / "example_models" / "unicornsc",
        "empty_stream_unicornsc.txt",
    )

    if context.spec["mode"] == "scratch":
        _prepare_scratch(context, environment)
    else:
        _stage_pretrained(context)
        _stage_threshold_override(context)

    rows_by_class = _execute_tests(context, environment)
    context.run_command(
        [sys.executable, "evaluate_unicornsc.py"],
        context.source_dir / "scripts",
        "evaluate",
    )
    threshold = _read_threshold(context)
    _emit_normalized_metrics(
        context, _normalized_metrics(rows_by_class, threshold)
    )
