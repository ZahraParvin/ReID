import pytest
from unittest.mock import patch, MagicMock, mock_open
import requests

import logger


@pytest.fixture(autouse=True)
def reset_server_url():
    """Ensure SERVER_URL is consistent across tests."""
    original = logger.SERVER_URL
    yield
    logger.SERVER_URL = original


class TestDownloadFullLog:
    def test_successful_download(self, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"col1,col2\nval1,val2\n"

        with patch("logger.requests.get", return_value=mock_response) as mock_get, \
             patch("builtins.open", mock_open()) as mock_file:
            logger.download_full_log()

        mock_get.assert_called_once_with(f"{logger.SERVER_URL}/get_log", timeout=10)
        mock_file().write.assert_called_once_with(b"col1,col2\nval1,val2\n")

    def test_non_200_response(self, capsys):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("logger.requests.get", return_value=mock_response):
            logger.download_full_log()

        captured = capsys.readouterr()
        assert "Error: 404" in captured.out

    def test_request_exception(self, capsys):
        with patch("logger.requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            logger.download_full_log()

        captured = capsys.readouterr()
        assert "Error downloading log" in captured.out

    def test_timeout_exception(self, capsys):
        with patch("logger.requests.get", side_effect=requests.exceptions.Timeout):
            logger.download_full_log()

        captured = capsys.readouterr()
        assert "Error downloading log" in captured.out


class TestReadLogContinuously:
    """Tests for read_log_continuously — exits via exception to avoid infinite loop."""

    def _run_n_iterations(self, responses, capsys):
        """
        Drive read_log_continuously for len(responses) iterations, then raise
        StopIteration to break out of the infinite loop.
        """
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            if call_count >= len(responses):
                raise StopIteration("done")
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("logger.requests.get", side_effect=fake_get), \
             patch("logger.time.sleep"):
            with pytest.raises(StopIteration):
                logger.read_log_continuously()

        return capsys.readouterr()

    def _make_response(self, logs, status=200):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = {"logs": logs}
        return r

    def test_prints_new_lines(self, capsys):
        responses = [
            self._make_response(["line1", "line2"]),
            self._make_response(["line1", "line2", "line3"]),
        ]
        out = self._run_n_iterations(responses, capsys)
        assert "line1" in out.out
        assert "line2" in out.out
        assert "line3" in out.out

    def test_no_duplicate_lines_printed(self, capsys):
        same_logs = ["line1", "line2"]
        responses = [
            self._make_response(same_logs),
            self._make_response(same_logs),
        ]
        out = self._run_n_iterations(responses, capsys)
        # "line1" should appear exactly once
        assert out.out.count("line1") == 1

    def test_non_200_response_prints_nothing(self, capsys):
        responses = [self._make_response([], status=500)]
        out = self._run_n_iterations(responses, capsys)
        assert out.out == ""

    def test_connection_error_handled(self, capsys):
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.ConnectionError("no route")
            raise StopIteration("done")

        with patch("logger.requests.get", side_effect=fake_get), \
             patch("logger.time.sleep"):
            with pytest.raises(StopIteration):
                logger.read_log_continuously()

        captured = capsys.readouterr()
        assert "Error reading logs" in captured.out

    def test_sleep_called_on_success(self):
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise StopIteration
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"logs": []}
            return r

        with patch("logger.requests.get", side_effect=fake_get), \
             patch("logger.time.sleep") as mock_sleep:
            with pytest.raises(StopIteration):
                logger.read_log_continuously()

        mock_sleep.assert_any_call(1)

    def test_sleep_longer_on_error(self):
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise StopIteration
            raise Exception("boom")

        with patch("logger.requests.get", side_effect=fake_get), \
             patch("logger.time.sleep") as mock_sleep:
            with pytest.raises(StopIteration):
                logger.read_log_continuously()

        mock_sleep.assert_any_call(5)
