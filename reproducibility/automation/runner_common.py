import datetime
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


SCHEMA_VERSION = 1


class BlockedRun(Exception):
    pass


class InterruptedRun(Exception):
    pass


class CommandFailure(Exception):
    def __init__(self, stage, command, returncode, output):
        super(CommandFailure, self).__init__(
            "Command failed during {} with exit code {}".format(stage, returncode)
        )
        self.stage = stage
        self.command = command
        self.returncode = returncode
        self.output = output


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def read_json(path):
    with Path(path).open("r") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def repository_state(repo_root):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            universal_newlines=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root),
                universal_newlines=True,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return None, None


def load_experiment(catalog_path, dataset, mode, experiment):
    catalog = read_json(catalog_path)
    for item in catalog.get("experiments", []):
        if (
            item.get("dataset") == dataset
            and item.get("mode") == mode
            and item.get("experiment") == experiment
        ):
            return catalog, item
    raise ValueError(
        "Unknown combination: dataset={!r}, mode={!r}, experiment={!r}".format(
            dataset, mode, experiment
        )
    )


def catalog_payload(catalog):
    return {
        "schema_version": SCHEMA_VERSION,
        "system": catalog["system"],
        "experiments": catalog.get("experiments", []),
    }


def copy_path(source, destination, ignore_patterns=None):
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise BlockedRun("Required source path is missing: {}".format(source))
    if source.is_dir():
        ignore = shutil.ignore_patterns(*(ignore_patterns or []))
        shutil.copytree(str(source), str(destination), symlinks=True, ignore=ignore)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))


def link_path(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(str(destination))
        else:
            destination.unlink()
    destination.symlink_to(source, target_is_directory=source.is_dir())


def link_or_copy(source, destination, copy_file=False):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(str(destination))
        else:
            destination.unlink()
    if copy_file and source.is_file():
        shutil.copy2(str(source), str(destination))
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def nonempty_file(path):
    return (
        path is not None
        and Path(path).is_file()
        and Path(path).stat().st_size > 0
    )


def find_named(path, name):
    path = Path(path)
    if path.is_file():
        return path if path.name == name else None
    if not path.is_dir():
        return None
    direct = path / name
    if direct.exists():
        return direct
    matches = sorted(path.rglob(name), key=lambda item: str(item))
    return matches[0] if matches else None


def download_http(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "artifact-runner/1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    if partial.stat().st_size == 0:
        raise BlockedRun("Downloaded file is empty: {}".format(url))
    os.replace(str(partial), str(destination))
    return destination


def extract_tar(archive, destination):
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with tarfile.open(str(archive), "r:*") as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    raise BlockedRun(
                        "Archive contains an unsafe path: {}".format(member.name)
                    )
            handle.extractall(str(destination))
    except (OSError, tarfile.TarError) as error:
        raise BlockedRun("Could not extract {}: {}".format(archive, error))


def detect_device(policy):
    if policy == "cpu":
        return {"policy": policy, "actual": "cpu", "gpu_name": None}
    if policy == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "policy": policy,
                    "actual": "cuda:0",
                    "gpu_name": torch.cuda.get_device_name(0),
                }
            return {"policy": policy, "actual": "cpu", "gpu_name": None}
        except Exception:
            pass
    return {"policy": policy, "actual": "unknown", "gpu_name": None}


class RunContext(object):
    def __init__(self, repo_root, system, spec, output_root=None):
        self.repo_root = Path(repo_root).resolve()
        self.system = system
        self.spec = spec
        self.started_at = utc_now()
        stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = "{}-{}-{}-{}".format(
            safe_name(spec["dataset"]),
            safe_name(spec["mode"]),
            safe_name(spec["experiment"]),
            stamp,
        )
        runs_root = (
            Path(output_root).resolve()
            if output_root
            else self.repo_root / "reproducibility" / "runs"
        )
        self.run_dir = runs_root / base
        index = 1
        while self.run_dir.exists():
            self.run_dir = runs_root / "{}-{}".format(base, index)
            index += 1
        self.run_id = self.run_dir.name
        self.files_dir = self.run_dir / "files"
        self.source_dir = self.files_dir / "source"
        self.inputs_dir = self.files_dir / "inputs"
        self.intermediate_dir = self.files_dir / "intermediate"
        self.models_dir = self.files_dir / "models"
        self.notebooks_dir = self.files_dir / "notebooks"
        self.database_dir = self.files_dir / "database"
        self.results_dir = self.run_dir / "results"
        for path in (
            self.source_dir,
            self.inputs_dir,
            self.intermediate_dir,
            self.models_dir,
            self.notebooks_dir,
            self.database_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / "run.log"
        self.result_path = self.results_dir / "results.json"
        self.report_path = self.results_dir / "report.md"
        self.commands = []
        self.inputs = []
        write_json(self.inputs_dir / "manifest.json", {"inputs": self.inputs})
        signal.signal(signal.SIGTERM, self._handle_termination)
        commit, dirty = repository_state(self.repo_root)
        self.result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "system": system,
            "dataset": spec["dataset"],
            "mode": spec["mode"],
            "experiment": spec["experiment"],
            "execution_status": "running",
            "comparison_status": "not_evaluated",
            "started_at": self.started_at,
            "finished_at": None,
            "repository_commit": commit,
            "repository_dirty": dirty,
            "source_path": "../files/source",
            "log_path": "run.log",
            "expected_outcome": spec.get(
                "expected_execution", {"type": "success"}
            ),
            "observed_outcome": None,
            "device": detect_device(spec.get("device_policy", "unknown")),
            "metrics": {},
            "message": None,
            "commands": self.commands,
        }
        self._write_result()
        self.log("Run created: {}".format(self.run_dir))

    def _handle_termination(self, signum, frame):
        raise InterruptedRun("The runner received termination signal {}.".format(signum))

    def log(self, message):
        line = "[{}] {}".format(utc_now(), message)
        print(line)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")

    def record_input(self, path, role):
        path = Path(path)
        self.inputs.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "is_symlink": path.is_symlink(),
                "size": (
                    path.stat().st_size
                    if path.exists() and path.is_file()
                    else None
                ),
            }
        )
        write_json(self.inputs_dir / "manifest.json", {"inputs": self.inputs})

    def run_command(self, command, cwd, stage, env=None):
        command = [str(value) for value in command]
        cwd = Path(cwd).resolve()
        self.log(
            "START stage={} cwd={} command={}".format(
                stage, cwd, " ".join(command)
            )
        )
        record = {
            "stage": stage,
            "cwd": str(cwd),
            "command": command,
            "started_at": utc_now(),
            "finished_at": None,
            "returncode": None,
        }
        self.commands.append(record)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        output = []
        with self.log_path.open("a") as handle:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
                output.append(line)
        process.stdout.close()
        returncode = process.wait()
        record["finished_at"] = utc_now()
        record["returncode"] = returncode
        text = "".join(output)
        self._write_result()
        self.log("END stage={} exit={}".format(stage, returncode))
        if returncode != 0:
            raise CommandFailure(stage, command, returncode, text)
        return text

    def _write_result(self):
        write_json(self.result_path, self.result)

    def finish(self, status, message, metrics, observed):
        self.result["execution_status"] = status
        self.result["message"] = message
        self.result["metrics"] = metrics or {}
        self.result["observed_outcome"] = observed
        self.result["finished_at"] = utc_now()
        self._write_result()
        self._write_report()
        self.log("{}: {}".format(status.upper(), message))

    def _write_report(self):
        lines = [
            "# Reproducibility Run Report",
            "",
            "- Run ID: `{}`".format(self.run_id),
            "- System: `{}`".format(self.system),
            "- Dataset: `{}`".format(self.spec["dataset"]),
            "- Mode: `{}`".format(self.spec["mode"]),
            "- Experiment: `{}`".format(self.spec["experiment"]),
            "- Execution status: `{}`".format(self.result["execution_status"]),
            "- Comparison status: `{}`".format(self.result["comparison_status"]),
            "- Device: `{}`".format(self.result["device"]["actual"]),
            "",
        ]
        headings = {
            "expected_failure": "EXPECTED FAILURE OBSERVED",
            "unexpected_failure": "UNEXPECTED FAILURE",
            "unexpected_success": "UNEXPECTED SUCCESS",
            "blocked": "BLOCKED",
            "completed": "Outcome",
        }
        lines.extend(
            [
                "## {}".format(headings.get(self.result["execution_status"], "Outcome")),
                "",
                self.result["message"],
                "",
                "## Observed metrics",
                "",
                "| Metric | Observed |",
                "|---|---:|",
            ]
        )
        metrics = self.result.get("metrics") or {}
        if metrics:
            for key in sorted(metrics):
                lines.append("| {} | {} |".format(key, metrics[key]))
        else:
            lines.append("| _none reported_ |  |")
        lines.extend(
            [
                "",
                "## Evidence",
                "",
                "- Raw log: [`run.log`](./run.log)",
                "- Source snapshot: `../files/source/`",
                "- Input manifest: `../files/inputs/manifest.json`",
                "",
            ]
        )
        self.report_path.write_text("\n".join(lines))


def classify_failure(spec, failure):
    expected = spec.get("expected_execution", {"type": "success"})
    if expected.get("type") == "failure":
        markers = expected.get("signatures", [])
        marker_match = any(
            str(marker).lower() in failure.output.lower() for marker in markers
        )
        if expected.get("stage") == failure.stage and marker_match:
            return "expected_failure", expected.get("reason", "Documented failure")
    return (
        "unexpected_failure",
        "The experiment failed unexpectedly during {} (exit {}).".format(
            failure.stage, failure.returncode
        ),
    )


def execute_workflow(context, workflow, parse_metrics):
    try:
        workflow(context)
        metrics = parse_metrics(context.log_path)
        if context.spec.get("expected_execution", {}).get("type") == "failure":
            context.finish(
                "unexpected_success",
                "The documented expected failure was not observed.",
                metrics,
                {"type": "success", "stage": "evaluate", "command_exit_code": 0},
            )
            code = 2
        else:
            context.finish(
                "completed",
                "The selected workflow completed successfully.",
                metrics,
                {"type": "success", "stage": "evaluate", "command_exit_code": 0},
            )
            code = 0
    except BlockedRun as error:
        context.finish(
            "blocked",
            str(error),
            parse_metrics(context.log_path),
            {"type": "blocked", "stage": "prepare", "command_exit_code": None},
        )
        code = 3
    except CommandFailure as error:
        status, message = classify_failure(context.spec, error)
        context.finish(
            status,
            message,
            parse_metrics(context.log_path),
            {
                "type": "failure",
                "stage": error.stage,
                "command_exit_code": error.returncode,
            },
        )
        code = 0 if status == "expected_failure" else 1
    except (InterruptedRun, KeyboardInterrupt) as error:
        message = str(error) or "The run was interrupted by the user."
        context.finish(
            "unexpected_failure",
            message,
            parse_metrics(context.log_path),
            {"type": "failure", "stage": "runner", "command_exit_code": None},
        )
        code = 1
    except Exception as error:
        context.finish(
            "unexpected_failure",
            "Runner error: {}: {}".format(type(error).__name__, error),
            parse_metrics(context.log_path),
            {"type": "failure", "stage": "runner", "command_exit_code": None},
        )
        code = 4
    print(str(context.result_path))
    return code
