"""提交/推送前隐私检查：暂存快照、新增提交、常见凭据与本机私有文件。

检查结果只输出文件、行号、类别，不回显匹配内容。历史保留不变；待推送检查
只检查本次新增提交和最终快照，不重新处置已经接受的旧历史。
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from .check_private_paths import git, private_path_lines, read_blobs, tree_entries
except ImportError:
    from check_private_paths import git, private_path_lines, read_blobs, tree_entries

TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
KEY = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")
PHONE = re.compile(r"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])")
ACCOUNT = re.compile(r"(?:\buid\s*[:=]\s*[\"']?|\bID)(\d{6,})", re.IGNORECASE)
LITERAL = re.compile(
    r"\b(password|passwd|api[_-]?key|api[_-]?secret|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|cred|token|sessdata|bili_jct)"
    r"[\"']?\s*[:=]\s*[\"']([^\"'\r\n]+)[\"']",
    re.IGNORECASE,
)
SAMPLE_VALUES = frozenset(
    {
        "...",
        "redacted",
        "change_me",
        "your_api_key",
        "your_token",
        "your_password",
    }
)
PRIVATE_NAMES = frozenset(
    {
        "skland_cred.json",
        "roster_skland.json",
        "roster_merged.json",
        "operbox.json",
        "user_overrides.json",
    }
)
PRIVATE_EXTENSIONS = frozenset(
    {
        ".bundle",
        ".har",
        ".p12",
        ".pfx",
        ".key",
        ".npz",
        ".parquet",
        ".pt",
        ".mp4",
        ".avi",
        ".png",
        ".jpg",
        ".jpeg",
        ".7z",
        ".zip",
    }
)


def private_filename(name: str) -> bool:
    """阻止强制 git add 的私有导出/本机配置，即使 .gitignore 已覆盖。"""
    p = Path(name)
    parts = {part.lower() for part in p.parts}
    filename = p.name.lower()
    return (
        bool(
            parts
            & {
                ".secrets",
                "episodes",
                "episodes_quarantine",
                "manual_sessions",
                "debug",
            }
        )
        or filename in PRIVATE_NAMES
        or filename.endswith((".local.yaml", ".local.json", ".fast-export", ".tar.gz"))
        or p.suffix.lower() in PRIVATE_EXTENSIONS
        or (
            filename.startswith(".env")
            and not filename.endswith((".example", ".sample"))
        )
    )


def findings(name: str, content: bytes) -> list[tuple[int, str]]:
    """检测常见泄露形态；仍需人工复核无法机械判断的身份和业务信息。"""
    result = {(line, "personal-profile-path") for line in private_path_lines(content)}
    if private_filename(name):
        result.add((1, "private-file-or-export"))
    if b"\0" in content:
        result.add((1, "binary-file-requires-review"))
        return sorted(result)
    text = content.decode("utf-8", errors="replace")
    for rule, kind in (
        (TOKEN, "access-token"),
        (KEY, "private-key"),
        (PHONE, "phone-number"),
        (ACCOUNT, "account-id"),
    ):
        for match in rule.finditer(text):
            result.add((text.count("\n", 0, match.start()) + 1, kind))
    for match in LITERAL.finditer(text):
        field, value = match.group(1).lower(), match.group(2).strip()
        placeholder = (
            value.lower() in SAMPLE_VALUES
            or value.startswith(("FAKE_", "SYN_", "SYNTHETIC_", "${", "%"))
            or (value.startswith("<") and value.endswith(">"))
        )
        short_token = field in {"cred", "token"} and len(value) < 12
        if not placeholder and not short_token:
            result.add((text.count("\n", 0, match.start()) + 1, "literal-credential"))
    return sorted(result)


def staged_entries(repo: Path) -> list[tuple[str, str]]:
    """读取真实 index，而非可能与暂存区不同的工作树。"""
    result = []
    for row in git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, oid, stage = header.split()
        if stage != b"0":
            raise ValueError("暂存区存在未解决冲突")
        if mode != b"160000":
            result.append((oid.decode(), name.decode("utf-8")))
    return result


def outgoing_entries(
    repo: Path, base: str, tip: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """检查每个新提交的完整快照，覆盖先添加后删除及重新引用旧 blob 的情况。"""
    if not base.strip("0"):
        probe = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "origin/main"],
            capture_output=True,
            check=False,
        )
        base = probe.stdout.decode().strip() if probe.returncode == 0 else ""
    selection = f"{base}..{tip}" if base else tip
    commits = git(repo, "rev-list", "--reverse", selection).decode().splitlines()
    entries = set(tree_entries(repo, tip))
    for commit in commits:
        entries.update(tree_entries(repo, commit))
    return sorted(entries), commits


def scan(
    repo: Path, entries: list[tuple[str, str]], commits: list[str] | None = None
) -> int:
    """只打印定位与类别，保持秘密值不进入检查日志。"""
    count = 0
    for name, body in read_blobs(repo, entries):
        for line, kind in findings(name, body):
            print(f"{name}:{line}: {kind} (content hidden)")
            count += 1
    for commit in commits or []:
        for line, kind in findings(
            "commit-metadata", git(repo, "cat-file", "commit", commit)
        ):
            print(f"commit {commit[:12]}:{line}: {kind} (content hidden)")
            count += 1
    print(
        f"Privacy check: {len(entries)} file versions, {len(commits or [])} new commits, {count} findings"
    )
    return 1 if count else 0


def install_hooks(repo: Path) -> None:
    """仅安装本仓库本地 hook；存在用户 hook 或自定义 hooksPath 时拒绝覆盖。"""
    custom = subprocess.run(
        ["git", "-C", str(repo), "config", "--get", "core.hooksPath"],
        capture_output=True,
        check=False,
    )
    if custom.returncode == 0:
        raise ValueError("已有 core.hooksPath，拒绝覆盖；请先检查并整合现有 hook")
    directory = (
        Path(git(repo, "rev-parse", "--absolute-git-dir").decode().strip()) / "hooks"
    )
    interpreter = shlex.quote(Path(sys.executable).as_posix())
    contents = {}
    for hook, mode in (("pre-commit", "--staged"), ("pre-push", "--pre-push")):
        contents[directory / hook] = (
            "#!/bin/sh\n# ark-autoplay privacy check; repository-local only\n"
            'root="$(git rev-parse --show-toplevel)" || exit 2\n'
            f'exec {interpreter} "$root/scripts/check_privacy.py" {mode}\n'
        )
    for path, body in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != body:
            raise ValueError("已有不同 hook，拒绝覆盖；请先检查并整合")
    directory.mkdir(parents=True, exist_ok=True)
    for path, body in contents.items():
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | 0o111)
    print("Installed repository-local pre-commit and pre-push privacy hooks")


def main(argv: list[str] | None = None) -> int:
    """默认检查 HEAD；本地 hook 在真正提交/发送对象前调用其他检查模式。"""
    parser = argparse.ArgumentParser(
        description="提交/推送前隐私检查（命中内容不回显）"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--outgoing", metavar="BASE")
    group.add_argument("--pre-push", action="store_true")
    group.add_argument("--install-hooks", action="store_true")
    parser.add_argument("--tip", default="HEAD")
    args = parser.parse_args(argv)
    try:
        repo = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                encoding="utf-8",
            ).strip()
        )
        if args.install_hooks:
            install_hooks(repo)
            return 0
        if args.pre_push:
            result = 0
            for row in sys.stdin:
                _local_ref, tip, _remote_ref, base = row.split()
                if not tip.strip("0"):
                    continue
                entries, commits = outgoing_entries(repo, base, tip)
                result = max(result, scan(repo, entries, commits))
            return result
        if args.outgoing:
            entries, commits = outgoing_entries(repo, args.outgoing, args.tip)
            return scan(repo, entries, commits)
        return scan(
            repo, staged_entries(repo) if args.staged else tree_entries(repo, args.tip)
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(
            f"Privacy check could not finish ({type(exc).__name__}); "
            "stop commit/push and inspect locally or fetch the baseline first"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
