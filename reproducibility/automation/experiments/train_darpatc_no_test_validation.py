#!/usr/bin/env python3
"""Train a DARPA TC3 model without selecting it on the testing graph.

The upstream ``train_darpatc.py`` calls ``validate()`` after every training
attempt.  That function loads the selected scene's testing graph and deletes
models that do not meet its precision and recall thresholds.  This experiment
keeps the upstream training implementation intact but replaces that validation
step with an immediate success result.  The resulting model is therefore the
first model trained from the training graph alone.
"""

import os

import train_darpatc as upstream


def skip_test_graph_validation():
    print(
        "Skipping DARPA test-graph validation for the "
        "no-test-validation experiment."
    )
    return 1


def main():
    upstream.validate = skip_test_graph_validation
    upstream.main()


if __name__ == "__main__":
    graphchi_root = os.path.abspath(
        os.path.join(os.getcwd(), "../graphchi-cpp-master")
    )
    os.environ["GRAPHCHI_ROOT"] = graphchi_root
    main()
