"""隐私门禁回归：暂存区和新增历史检查，使用合成数据与独立临时仓库。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_privacy import (
    findings,
    install_hooks,
    outgoing_entries,
    private_filename,
    scan,
    staged_entries,
)


class PrivacyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ark-privacy-test-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "unicode-测试"
        self.repo.mkdir()
        self.git("init", "--quiet", "--initial-branch=main")

    def git(self, *args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(self.repo), *args])

    def commit(self, body: str, message: str) -> str:
        (self.repo / "note.txt").write_text(body, encoding="utf-8")
        self.git("add", "note.txt")
        self.git(
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        )
        return self.git("rev-parse", "HEAD").decode().strip()

    def scan_silently(
        self, entries: list, commits: list | None = None
    ) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = scan(self.repo, entries, commits)
        return code, output.getvalue()

    def test_private_config_and_raw_exports_are_blocked(self) -> None:
        for name in (
            "collector.local.yaml",
            ".secrets/cred.json",
            "roster_skland.json",
            "capture.mp4",
            "before.bundle",
            "manual_sessions/sample/session.json",
            "bridge/manual_sessions/sample/input_events.jsonl",
        ):
            self.assertTrue(private_filename(name))
        self.assertFalse(private_filename("example.yaml"))
        self.assertFalse(private_filename(".env.example"))

    def test_tokens_passwords_and_binary_are_found_without_echoing(self) -> None:
        token = "ghp_" + "x" * 36
        self.assertTrue(findings("code.py", token.encode()))
        key = "-" * 5 + "BEGIN " + "PRIVATE KEY" + "-" * 5
        self.assertTrue(findings("key.txt", key.encode()))
        example_value = "not-a-real-password"
        self.assertTrue(
            findings("config.json", json.dumps({"password": example_value}).encode())
        )
        self.assertFalse(
            findings("example.json", json.dumps({"token": "<TOKEN>"}).encode())
        )
        self.assertTrue(findings("opaque.bin", b"\0opaque"))

    def test_phone_detection_does_not_mistake_dependency_hashes(self) -> None:
        phone = "1" + "3" + "0" * 9
        self.assertTrue(findings("note.txt", phone.encode()))
        self.assertFalse(findings("uv.lock", ("ab" + phone + "cd").encode()))

    def test_staged_content_is_checked_even_after_worktree_is_cleaned(self) -> None:
        self.commit("safe", "baseline")
        token = "ghp_" + "x" * 36
        (self.repo / "note.txt").write_text(token)
        self.git("add", "note.txt")
        (self.repo / "note.txt").write_text("safe")
        code, output = self.scan_silently(staged_entries(self.repo))
        self.assertEqual(code, 1)
        self.assertNotIn(token, output)

    def test_intermediate_leak_blocks_push_even_when_tip_is_clean(self) -> None:
        base = self.commit("safe", "baseline")
        token = "ghp_" + "x" * 36
        self.commit(token, "intermediate")
        tip = self.commit("safe again", "clean tip")
        entries, commits = outgoing_entries(self.repo, base, tip)
        code, output = self.scan_silently(entries, commits)
        self.assertEqual(code, 1)
        self.assertEqual(len(commits), 2)
        self.assertNotIn(token, output)

    def test_accepted_old_history_does_not_block_clean_new_commits(self) -> None:
        drive = "C"
        self.commit(f"{drive}:/Users/SyntheticPerson/data", "old history")
        base = self.commit("clean baseline", "cleanup")
        tip = self.commit("clean new content", "new change")
        entries, commits = outgoing_entries(self.repo, base, tip)
        code, _ = self.scan_silently(entries, commits)
        self.assertEqual(code, 0)
        self.assertEqual(len(commits), 1)

    def test_hook_install_is_local_idempotent_and_preserves_foreign_hooks(self) -> None:
        hooks = self.repo / ".git/hooks"
        original = hooks / "pre-push"
        original.write_text("#!/bin/sh\n# synthetic existing hook\n")
        with self.assertRaises(ValueError):
            install_hooks(self.repo)
        self.assertIn("synthetic existing hook", original.read_text())
        self.assertFalse((hooks / "pre-commit").exists())
        original.unlink()  # 仅删除本测试自行创建的文件。
        with contextlib.redirect_stdout(io.StringIO()):
            install_hooks(self.repo)
            install_hooks(self.repo)
        self.assertIn("--staged", (hooks / "pre-commit").read_text())
        self.assertIn("--pre-push", original.read_text())

    def test_real_pre_push_hook_blocks_intermediate_commit_before_transfer(
        self,
    ) -> None:
        source = Path(__file__).resolve().parents[1] / "scripts"
        (self.repo / "scripts").mkdir()
        for name in ("check_privacy.py", "check_private_paths.py"):
            shutil.copyfile(source / name, self.repo / "scripts" / name)
        self.git("add", "scripts")
        base = self.commit("safe baseline", "baseline")
        remote = self.repo / "remote.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
        self.git("remote", "add", "origin", str(remote))
        self.git("push", "--quiet", "origin", "main")
        token = "ghp_" + "x" * 36
        bad = self.commit(token, "intermediate")
        self.commit("safe tip", "cleanup")
        with contextlib.redirect_stdout(io.StringIO()):
            install_hooks(self.repo)
        environment = dict(os.environ, PYTHONUTF8="0")
        pushed = subprocess.run(
            ["git", "-C", str(self.repo), "push", "origin", "main"],
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertNotEqual(pushed.returncode, 0)
        self.assertIn(b"access-token", pushed.stdout)
        self.assertNotIn(token.encode(), pushed.stdout + pushed.stderr)
        self.assertEqual(
            self.git("--git-dir", str(remote), "rev-parse", "main").decode().strip(),
            base,
        )
        absent = subprocess.run(
            ["git", "--git-dir", str(remote), "cat-file", "-e", bad],
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(absent.returncode, 0)


if __name__ == "__main__":
    unittest.main()
