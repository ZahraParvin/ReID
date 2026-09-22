from unittest.mock import Mock

import pytest
import requests

import evaluate_system as evaluation


@pytest.mark.parametrize("mode", ["timeout", "http500"])
def test_failed_requests_are_not_latency_samples(monkeypatch, mode):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500")
    post = (
        Mock(side_effect=requests.Timeout("offline"))
        if mode == "timeout"
        else Mock(return_value=response)
    )
    monkeypatch.setattr(evaluation.requests, "post", post)
    phase = evaluation.phase_performance("http://unused", n=2)
    assert phase.status == "FAIL"
    assert "mean_ms" not in phase.details


def test_partial_request_failure_still_fails(monkeypatch):
    monkeypatch.setattr(
        evaluation.requests, "post", Mock(side_effect=[requests.Timeout(), Mock()])
    )
    assert evaluation.phase_performance("http://unused", n=2).status == "FAIL"


def test_cli_reports_server_failure_and_creates_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation.requests, "post", Mock(side_effect=requests.ConnectionError())
    )
    monkeypatch.setattr(
        evaluation.requests, "get", Mock(side_effect=requests.ConnectionError())
    )
    report = tmp_path / "nested" / "report.txt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_system.py",
            "--no-realtime",
            "--no-pipeline",
            "--report",
            str(report),
        ],
    )
    assert evaluation.main() == 1
    assert "[FAIL]" in report.read_text(encoding="utf-8")
