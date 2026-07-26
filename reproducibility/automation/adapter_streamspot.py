"""StreamSpot workflow adapter for the threaTrace reproducibility runner."""

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
    find_named,
    link_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ANOMALY_PATTERN = re.compile(r"\bfp:\s*([0-9]+)\b")


def _preprocessed_graphs(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return []
    return [
        graph
        for graph in path.glob("[1-6]/*.txt")
        if graph.is_file() and graph.stat().st_size > 0
    ]


def _prepare_data(context, args):
    configured = Path(args.data_dir).expanduser() if args.data_dir else None
    data_path = (
        configured
        if configured
        else REPO_ROOT / "graphchi-cpp-master" / "graph_data" / "streamspot"
    )
    target = context.intermediate_dir / "graph_data" / "streamspot"
    target.mkdir(parents=True, exist_ok=True)

    graphs = [] if args.rebuild_data else _preprocessed_graphs(data_path)
    if len(graphs) >= 600:
        context.record_input(data_path, "preprocessed StreamSpot graph directory")
        for scene in range(1, 7):
            scene_source = data_path / str(scene)
            if scene_source.exists():
                link_path(scene_source, target / str(scene))
        return

    raw = find_named(data_path, "all.tsv") if data_path.exists() else None
    if raw is not None and raw.is_file() and raw.stat().st_size > 0:
        context.record_input(raw, "StreamSpot extracted raw data")
        link_path(raw, context.source_dir / "scripts" / "all.tsv")
        parser_path = context.source_dir / "scripts" / "parse_streamspot.py"
        parser_text = parser_path.read_text()
        archive_command = (
            "os.system('tar -zxvf "
            "../graphchi-cpp-master/graph_data/streamspot/all.tar.gz')"
        )
        if archive_command not in parser_text:
            raise BlockedRun(
                "The saved StreamSpot parser no longer contains the expected "
                "archive extraction line."
            )
        parser_path.write_text(parser_text.replace(archive_command, "", 1))
        context.run_command(
            [sys.executable, "parse_streamspot.py"],
            context.source_dir / "scripts",
            "preprocess",
        )
        generated = _preprocessed_graphs(target)
        if len(generated) < 600:
            raise BlockedRun(
                "StreamSpot preprocessing generated {} graph files; expected "
                "600.".format(len(generated))
            )
        return

    archive = (
        find_named(data_path, "all.tar.gz") if data_path.exists() else None
    )
    if archive is not None and (
        not archive.is_file() or archive.stat().st_size == 0
    ):
        archive = None
    if archive is None and args.download:
        archive = context.inputs_dir / "all.tar.gz"
        context.log("Downloading the public StreamSpot archive")
        download_http(context.spec["data_url"], archive)
    if archive is None:
        raise BlockedRun(
            "StreamSpot preprocessing requires all.tsv, all.tar.gz, or a "
            "preprocessed directory containing 600 graph .txt files. Supply "
            "--data-dir or use --download."
        )
    context.record_input(archive, "StreamSpot raw archive")
    link_path(archive, target / "all.tar.gz")
    context.run_command(
        [sys.executable, "parse_streamspot.py"],
        context.source_dir / "scripts",
        "preprocess",
    )
    generated = _preprocessed_graphs(target)
    if len(generated) < 600:
        raise BlockedRun(
            "StreamSpot preprocessing generated {} graph files; expected 600.".format(
                len(generated)
            )
        )


def _stage_pretrained(context):
    source = context.source_dir / "example_models" / "streamspot"
    required = [
        "feature.txt",
        "label.txt",
        "models_list.txt",
        "run_benign.sh",
        "run_attack.sh",
    ]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise BlockedRun(
            "Pretrained StreamSpot material is incomplete: {}".format(
                ", ".join(missing)
            )
        )
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(context.models_dir / item.name))
    scripts = context.source_dir / "scripts"
    for name in ("models_list.txt", "run_benign.sh", "run_attack.sh"):
        shutil.copy2(str(source / name), str(scripts / name))
    context.record_input(source, "threaTrace supplied StreamSpot models")


def _execute_tests(context, environment):
    scripts = context.source_dir / "scripts"
    result_names = {
        "run_benign.sh": "result_benign.txt",
        "run_attack.sh": "result_attack.txt",
    }
    for name, result_name in result_names.items():
        path = scripts / name
        if not path.exists():
            raise BlockedRun(
                "{} was not produced by model preparation.".format(name)
            )
        commands = [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        path.write_text("set -e\n" + path.read_text())
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        context.run_command(["bash", name], scripts, "test", env=environment)
        result_path = scripts / result_name
        observed_lines = (
            len(result_path.read_text().splitlines())
            if result_path.exists()
            else 0
        )
        if observed_lines != len(commands):
            message = (
                "{} produced {} result lines for {} test commands.".format(
                    name, observed_lines, len(commands)
                )
            )
            raise CommandFailure("test", ["bash", name], 1, message)
    context.run_command(
        [sys.executable, "evaluate_streamspot.py"],
        scripts,
        "evaluate",
    )
    output_dir = context.intermediate_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("result_benign.txt", "result_attack.txt"):
        path = scripts / name
        if path.exists():
            shutil.copy2(str(path), str(output_dir / name))
    _emit_normalized_metrics(context, output_dir)


def _result_counts(path):
    values = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        match = ANOMALY_PATTERN.search(line)
        if match is None:
            raise CommandFailure(
                "evaluate",
                [sys.executable, "evaluate_streamspot.py"],
                1,
                "{} row {} does not contain an fp count.".format(
                    Path(path).name, line_number
                ),
            )
        values.append(int(match.group(1)))
    return values


def _emit_normalized_metrics(context, output_dir):
    benign = _result_counts(output_dir / "result_benign.txt")
    attack = _result_counts(output_dir / "result_attack.txt")
    threshold = 2
    fp = sum(value > threshold for value in benign)
    tn = len(benign) - fp
    tp = sum(value > threshold for value in attack)
    fn = len(attack) - tp
    eps = 1e-10
    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / float(tp + tn + fp + fn),
        "fpr": fp / (fp + tn + eps),
        "threshold": threshold,
    }
    metrics["f1"] = (
        2
        * metrics["precision"]
        * metrics["recall"]
        / (metrics["precision"] + metrics["recall"] + eps)
    )
    labels = (
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
        for label, value in labels:
            line = "{}: {}".format(label, value)
            print(line)
            handle.write(line + "\n")
    (context.results_dir / "normalized_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )


def run_streamspot(context, args):
    """Run the configured StreamSpot experiment in an isolated run snapshot."""

    _prepare_data(context, args)
    run_setup(context)
    environment = initialize_graphchi(
        context,
        context.source_dir / "example_models" / "streamspot",
        "empty_stream_streamspot.txt",
    )
    if context.spec["mode"] == "pretrained":
        _stage_pretrained(context)
    else:
        context.run_command(
            [sys.executable, "train_streamspot.py"],
            context.source_dir / "scripts",
            "train",
            env=environment,
        )
    _execute_tests(context, environment)
