"""Failure policy and performance gates, independent of GPU/model dependencies."""

import argparse
import math
import re
from pathlib import Path

GATES = {
    "mean_ms": ("max", 20.0),
    "rps": ("min", 55.0),
    "effective_fps": ("min", 90.0),
    "detect_mean_ms": ("max", 40.0),
}


def evaluation_exit_code(statuses, *, strict=False):
    if not statuses or any(s not in {"PASS", "WARN", "SKIP"} for s in statuses):
        return 1
    if strict and (len(statuses) != 12 or any(s != "PASS" for s in statuses)):
        return 1
    return 0


def check_report(text):
    errors = []
    phases = re.findall(r"^\[(PASS|FAIL|WARN|SKIP)\] (.+)$", text, re.M)
    numbers = [
        int(m.group(1)) if (m := re.match(r"(\d+)\.", name)) else 0
        for _, name in phases
    ]
    if sorted(numbers) != list(range(1, 13)) or any(s != "PASS" for s, _ in phases):
        errors.append(
            "All twelve numbered phases must be present exactly once and PASS"
        )
    for key, (direction, limit) in GATES.items():
        values = re.findall(r"^\s+" + re.escape(key) + r":\s+(\S+)\s*$", text, re.M)
        if len(values) != 1:
            errors.append(f"{key}: expected one measurement, found {len(values)}")
            continue
        try:
            value = float(values[0])
        except ValueError:
            errors.append(f"{key}: invalid measurement")
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{key}: measurement must be finite and positive")
        elif (direction == "max" and value > limit) or (
            direction == "min" and value < limit
        ):
            errors.append(f"{key}: {value} violates {direction} {limit}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        errors = check_report(args.report.read_text(encoding="utf-8"))
    except OSError as exc:
        errors = [str(exc)]
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print("PASS: all twelve phases and four performance gates")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
