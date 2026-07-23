from __future__ import annotations

import unittest
from unittest import mock

import requests

from metis_data.acquisition.io import RetrySession


class AcquisitionIoTests(unittest.TestCase):
    def test_request_waits_for_explicit_github_rate_limit_reset(self) -> None:
        limited = requests.Response()
        limited.status_code = 403
        limited.raw = mock.Mock()
        limited.headers["X-RateLimit-Remaining"] = "0"
        limited.headers["X-RateLimit-Reset"] = "101"
        successful = requests.Response()
        successful.status_code = 200
        client = RetrySession(retries=1, timeout=10)
        with (
            mock.patch.object(
                client.session,
                "request",
                side_effect=[limited, successful],
            ) as request,
            mock.patch("metis_data.acquisition.io.time.time", return_value=100.0),
            mock.patch("metis_data.acquisition.io.time.sleep") as sleep,
        ):
            observed = client.request("GET", "https://api.github.com/repos/example/project")
        self.assertIs(observed, successful)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(3.0)


if __name__ == "__main__":
    unittest.main()
