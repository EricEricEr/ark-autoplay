"""使用合成用户名校验路径检查，样例不包含维护者真实路径。"""

from __future__ import annotations

import unittest

from scripts.check_private_paths import private_path_lines


class PrivatePathTests(unittest.TestCase):
    def test_real_profile_forward_and_backslashes(self) -> None:
        drive = "C"
        path = f"{drive}:/Users/SyntheticPerson/Desktop/tool"
        self.assertEqual(private_path_lines(path.encode()), [1])
        escaped = path.replace("/", "\\\\")
        self.assertEqual(private_path_lines(escaped.encode()), [1])

    def test_placeholders_and_variables_are_allowed(self) -> None:
        drive = "D"
        for username in ("xxx", "<user>", "local-user", "%USERNAME%"):
            path = f"{drive}:/Users/{username}/data"
            self.assertEqual(private_path_lines(path.encode()), [])
        self.assertEqual(private_path_lines(b"${ARK_ML_ROOT}/processed"), [])

    def test_reports_line_numbers_without_touching_binary(self) -> None:
        drive = "E"
        path = f"{drive}:/Users/SyntheticPerson/data"
        self.assertEqual(private_path_lines(("first\n" + path).encode()), [2])
        self.assertEqual(private_path_lines(b"\0" + path.encode()), [])


if __name__ == "__main__":
    unittest.main()
