import subprocess
import sys

import pytest

from regression import check_report, evaluation_exit_code


def good_report():
    return "\n".join(
        [f"[PASS] {i}. Phase" for i in range(1, 13)]
        + [
            "  mean_ms: 15.6",
            "  rps: 64.3",
            "  effective_fps: 102.2",
            "  detect_mean_ms: 30.0",
        ]
    )


@pytest.mark.parametrize(
    "statuses,strict,expected",
    [
        (["PASS"] * 12, True, 0),
        (["PASS", "FAIL"], False, 1),
        (["PASS"] * 11, True, 1),
        (["PASS"] * 11 + ["SKIP"], True, 1),
        (["PASS"] * 11 + ["WARN"], True, 1),
        ([], False, 1),
        (["PASS", "SKIP"], False, 0),
        (["UNKNOWN"], False, 1),
    ],
)
def test_exit_policy(statuses, strict, expected):
    assert evaluation_exit_code(statuses, strict=strict) == expected


def test_accepts_complete_report():
    assert check_report(good_report()) == []


@pytest.mark.parametrize(
    "old,new",
    [
        ("[PASS] 3.", "[FAIL] 3."),
        ("[PASS] 3.", "[SKIP] 3."),
        ("[PASS] 3.", "[WARN] 3."),
        ("[PASS] 3.", "[PASS] 2."),
        ("  mean_ms: 15.6", ""),
        ("15.6", "nan"),
        ("15.6", "inf"),
        ("15.6", "invalid"),
        ("15.6", "-1"),
        ("15.6", "21"),
        ("64.3", "54"),
        ("102.2", "89"),
        ("30.0", "41"),
    ],
)
def test_rejects_incomplete_or_regressed_report(old, new):
    assert check_report(good_report().replace(old, new))


def test_rejects_duplicate_measurement():
    assert check_report(good_report() + "\n  mean_ms: 1")


def test_cli_failure_exit(tmp_path):
    report = tmp_path / "report.txt"
    report.write_text(good_report().replace("[PASS] 4.", "[FAIL] 4."))
    result = subprocess.run(
        [sys.executable, "regression.py", str(report)], capture_output=True
    )
    assert result.returncode == 1


def test_missing_report_fails(tmp_path):
    result = subprocess.run(
        [sys.executable, "regression.py", str(tmp_path / "missing")],
        capture_output=True,
    )
    assert result.returncode == 1
