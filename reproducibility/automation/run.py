#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import stat
import sys
from pathlib import Path

from adapter_darpa import run_darpa
from adapter_streamspot import run_streamspot
from adapter_unicornsc import run_unicornsc
from runner_common import (
    BlockedRun,
    RunContext,
    catalog_payload,
    copy_path,
    execute_workflow,
    link_path,
    load_experiment,
    read_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_DIR = Path(__file__).resolve().parent
CATALOG_PATH = AUTOMATION_DIR / "experiments.json"


class RunnerArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(4, "{}: error: {}\n".format(self.prog, message))


def build_parser():
    parser = RunnerArgumentParser(
        description="threaTrace reproducibility runner"
    )
    subparsers = parser.add_subparsers(dest="command")
    list_parser = subparsers.add_parser("list", help="List supported experiments")
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    run_parser = subparsers.add_parser("run", help="Run one experiment")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--mode", choices=["pretrained", "scratch"], required=True)
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--data-dir")
    run_parser.add_argument("--download", action="store_true")
    run_parser.add_argument("--rebuild-data", action="store_true")
    run_parser.add_argument("--output-dir")
    return parser


def snapshot_source(context):
    automation_snapshot = context.source_dir / "automation"
    automation_snapshot.mkdir(parents=True, exist_ok=True)
    for name in (
        "run.py",
        "adapter_common.py",
        "adapter_darpa.py",
        "adapter_streamspot.py",
        "adapter_unicornsc.py",
        "runner_common.py",
        "experiments.json",
        "expected_results.json",
        "README.md",
    ):
        copy_path(AUTOMATION_DIR / name, automation_snapshot / name)
    experiment_sources = AUTOMATION_DIR / "experiments"
    if experiment_sources.exists():
        copy_path(
            experiment_sources,
            automation_snapshot / "experiments",
            ignore_patterns=["__pycache__", "*.pyc"],
        )
    for name in ("README.md", "LICENSE"):
        copy_path(REPO_ROOT / name, context.source_dir / name)
    copy_path(
        REPO_ROOT / "scripts",
        context.source_dir / "scripts",
        ignore_patterns=[
            "__pycache__",
            "*.pyc",
            "result_*.txt",
            "alarm.txt",
            "groundtruth_uuid.txt",
            "groundtruth_nodeId.txt",
            "id_to_uuid.txt",
            "models_list.txt",
            "run_benign.sh",
            "run_attack.sh",
        ],
    )
    copy_path(
        REPO_ROOT / "graphchi-cpp-master",
        context.source_dir / "graphchi-cpp-master",
        ignore_patterns=["graph_data"],
    )
    copy_path(REPO_ROOT / "example_models", context.source_dir / "example_models")
    copy_path(REPO_ROOT / "groundtruth", context.source_dir / "groundtruth")
    link_path(context.models_dir, context.source_dir / "models")

    graph_data = context.intermediate_dir / "graph_data"
    graph_data.mkdir(parents=True, exist_ok=True)
    original_gdata = REPO_ROOT / "graphchi-cpp-master" / "graph_data" / "gdata"
    if original_gdata.exists():
        shutil.copy2(str(original_gdata), str(graph_data / "gdata"))
    link_path(
        graph_data,
        context.source_dir / "graphchi-cpp-master" / "graph_data",
    )

    binaries = (
        context.source_dir / "graphchi-cpp-master" / "bin" / "example_apps"
    )
    for filename in ("train", "test"):
        binary = binaries / filename
        if binary.exists():
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


def workflow_factory(args):
    def workflow(context):
        snapshot_source(context)
        adapter = context.spec.get("adapter")
        if adapter == "streamspot":
            run_streamspot(context, args)
        elif adapter == "unicornsc":
            run_unicornsc(context, args)
        elif adapter == "darpa":
            run_darpa(context, args)
        else:
            raise BlockedRun(
                "No threaTrace automation adapter is registered for {!r}.".format(
                    adapter
                )
            )

    return workflow


def last_metric(text, labels):
    value = None
    for label in labels:
        matches = re.findall(
            r"^{}\s*:\s*([0-9.eE+-]+)".format(re.escape(label)),
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if matches:
            try:
                value = float(matches[-1])
            except ValueError:
                value = matches[-1]
    return value


def parse_metrics(log_path):
    text = (
        Path(log_path).read_text(errors="replace")
        if Path(log_path).exists()
        else ""
    )
    metrics = {
        "tp": last_metric(text, ["TP"]),
        "tn": last_metric(text, ["TN"]),
        "fp": last_metric(text, ["FP"]),
        "fn": last_metric(text, ["FN"]),
        "precision": last_metric(text, ["Precision"]),
        "recall": last_metric(text, ["Recall"]),
        "f1": last_metric(text, ["F-Score", "F1"]),
        "accuracy": last_metric(text, ["Accuracy"]),
        "auc": last_metric(text, ["AUC"]),
        "fpr": last_metric(text, ["FPR"]),
        "threshold": last_metric(text, ["Threshold"]),
    }
    counts = [metrics[key] for key in ("tp", "tn", "fp", "fn")]
    if all(value is not None for value in counts):
        tp, tn, fp, fn = counts
        total = tp + tn + fp + fn
        if metrics["accuracy"] is None and total:
            metrics["accuracy"] = (tp + tn) / total
        if metrics["fpr"] is None and fp + tn:
            metrics["fpr"] = fp / (fp + tn)
    return metrics


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        payload = catalog_payload(read_json(CATALOG_PATH))
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for item in payload["experiments"]:
                print(
                    "{dataset} {mode} {experiment} supported={supported}".format(
                        **item
                    )
                )
        return 0
    if args.command != "run":
        parser.print_help()
        return 4
    try:
        catalog, spec = load_experiment(
            CATALOG_PATH, args.dataset, args.mode, args.experiment
        )
    except ValueError as error:
        parser.error(str(error))
    if not spec.get("supported", False):
        print(
            "UNSUPPORTED: {}".format(
                spec.get(
                    "unsupported_reason",
                    "This combination is unsupported.",
                )
            )
        )
        return 3
    context = RunContext(
        REPO_ROOT, catalog["system"], spec, output_root=args.output_dir
    )
    return execute_workflow(context, workflow_factory(args), parse_metrics)


if __name__ == "__main__":
    sys.exit(main())
