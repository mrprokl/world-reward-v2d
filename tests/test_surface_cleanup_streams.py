"""Execute original shell ownership checks with fake streams; never call Docker."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ("run_surface_identity_qualify.sh", "run_surface_qslim_qualify.sh")
CID = "a" * 64


def invoke(wrapper, stdout, stderr, status):
    text = (ROOT / "infra" / wrapper).read_text()
    function = "inspect_owned() {" + text.split("inspect_owned() {", 1)[1].split("\ncleanup() {", 1)[0]
    script = ("set -u\nNAME=manufactured;IMAGE=sha256:manufactured;REV=manufactured\n"
              + "timeout() { printf '%s' \"$MOCK_STDOUT\"; printf '%s' \"$MOCK_STDERR\" >&2; return \"$MOCK_STATUS\"; }\n"
              + function + "\ninspect_owned \"$CID\"\n")
    environment = dict(os.environ, CID=CID, MOCK_STDOUT=stdout, MOCK_STDERR=stderr, MOCK_STATUS=str(status))
    return subprocess.run(["bash", "-c", script], env=environment, capture_output=True, timeout=5).returncode


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("prefix", ("Error: No such object: ", "error: no such object: ",
    "Error: No such container: ", "Error response from daemon: No such container: "))
def test_exact_same_cid_absence_with_leading_empty_lines(wrapper, prefix):
    assert invoke(wrapper, "\n", prefix + CID + "\n", 1) == 1
    assert invoke(wrapper, "\n\n", prefix + CID + "\n", 1) == 1


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_original_empty_stdout_and_actual_lowercase_streams_are_absence(wrapper):
    error = "error: no such object: " + CID + "\n"
    assert invoke(wrapper, "", error, 1) == 1
    assert invoke(wrapper, "\n", error, 1) == 1


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("fault", ("array_stdout", "foreign_cid", "daemon", "extra_line", "text_prefix",
    "space_prefix", "wrong_rc", "timeout", "success_error"))
def test_other_streams_or_status_are_not_container_absence(wrapper, fault):
    stdout, stderr, status = "\n", "error: no such object: " + CID + "\n", 1
    if fault == "array_stdout": stdout = "[]\n"
    elif fault == "foreign_cid": stderr = "error: no such object: " + "b" * 64 + "\n"
    elif fault == "daemon": stderr = "error: cannot connect to daemon\n"
    elif fault == "extra_line": stderr += "unexpected text\n"
    elif fault == "text_prefix": stdout = "unexpected text\n"
    elif fault == "space_prefix": stdout = " \n"
    elif fault == "wrong_rc": status = 2
    elif fault == "timeout": status = 124
    elif fault == "success_error": status = 0
    assert invoke(wrapper, stdout, stderr, status) == 2


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("fault", (None, "name", "image", "revision", "extra"))
def test_success_requires_exact_owned_container_identity(wrapper, fault):
    name, image, revision = "manufactured", "sha256:manufactured", "manufactured"
    if fault == "name": name = "foreign"
    elif fault == "image": image = "sha256:foreign"
    elif fault == "revision": revision = "foreign"
    stdout = f"/{name} {image} {revision}\n"
    if fault == "extra": stdout += "unexpected text\n"
    assert invoke(wrapper, stdout, "", 0) == (0 if fault is None else 2)


def test_same_minimal_failure_parser_and_rc_capture_order():
    functions = []
    for name in WRAPPERS:
        raw = (ROOT / "infra" / name).read_text()
        function = raw.split("inspect_owned() {", 1)[1].split("\ncleanup() {", 1)[0]
        assert function.index("rc=$?") < function.index("while [[") < function.index("if (( rc == 1 ))")
        functions.append(function)
    assert functions[0] == functions[1]
