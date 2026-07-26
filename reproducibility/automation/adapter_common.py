"""Shared setup helpers for threaTrace dataset adapters."""

import os
import shutil
import stat
import sys

from runner_common import BlockedRun, nonempty_file


def run_setup(context):
    context.run_command(
        [sys.executable, "setup.py"],
        context.source_dir / "scripts",
        "setup",
    )


def graphchi_environment(context):
    environment = os.environ.copy()
    # Upstream ``run_*.sh`` files invoke the unqualified ``python`` command.
    # Make that resolve to the interpreter running this artifact runner (the
    # selected Conda environment), rather than to a caller's base environment.
    python_bin = os.path.dirname(os.path.abspath(sys.executable))
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    environment["GRAPHCHI_ROOT"] = str(
        context.source_dir / "graphchi-cpp-master"
    )
    return environment


def initialize_graphchi(context, example_model_directory, empty_filename):
    environment = graphchi_environment(context)
    temporary_model_files = []
    for name in ("feature.txt", "label.txt"):
        source = example_model_directory / name
        destination = context.models_dir / name
        if not nonempty_file(source):
            raise BlockedRun(
                "GraphChi initialization requires a supplied {}.".format(name)
            )
        if not destination.exists():
            shutil.copy2(str(source), str(destination))
            temporary_model_files.append(destination)

    empty_stream = (
        context.intermediate_dir / "graph_data" / empty_filename
    )
    empty_stream.write_text("")
    binary = (
        context.source_dir
        / "graphchi-cpp-master"
        / "bin"
        / "example_apps"
        / "test"
    )
    if not binary.is_file():
        raise BlockedRun("The source snapshot is missing the GraphChi test binary.")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    context.run_command(
        [
            binary,
            "file",
            context.intermediate_dir / "graph_data" / "gdata",
            "filetype",
            "edgelist",
            "stream_file",
            empty_stream,
            "batch",
            "200000",
        ],
        context.source_dir / "scripts",
        "graphchi_initialize",
        env=environment,
    )
    for path in temporary_model_files:
        path.unlink()
    return environment
