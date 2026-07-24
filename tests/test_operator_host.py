from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from metis_data.config import load_profile
from metis_data.doctor import _operator_role_checks


class OperatorHostTests(unittest.TestCase):
    def test_login2_profile_accepts_actual_portage_login2_hostnames(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "METIS_LUSTRE_ROOT": "/lus/lustre1/vollmerc/metis-1.6",
                "METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT": "unlimited",
            },
        ):
            _, profile = load_profile(
                Path(__file__).resolve().parents[1] / "configs" / "metis16" / "login2.yaml"
            )
        for hostname in (
            "login2",
            "login2.example",
            "portage-login2",
            "portage-login2.head.cm.hpcrb.rdlabs.ext.hpe.com",
        ):
            with self.subTest(hostname=hostname), mock.patch(
                "metis_data.doctor.socket.gethostname", return_value=hostname
            ):
                checks = _operator_role_checks(profile, "acquisition")
                host_check = next(check for check in checks if check.name == "operator-host")
                self.assertEqual(host_check.status, "PASS", host_check.detail)


if __name__ == "__main__":
    unittest.main()
