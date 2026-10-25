"""检查已跟踪文件中的真实 Windows 用户目录，不输出命中内容或用户名。

默认检查 HEAD；--worktree 也检查未忽略的待提交文件；--history 检查所有本地
引用可达的历史 blob。占位符可以公开，真实机器配置应放进 *.local.yaml。
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

PROFILE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'`]+)", re.IGNORECASE)
PLACEHOLDERS = frozenset({
    "xxx", "user", "username", "example", "yourname", "local-user",
    "%username%", "${username}", "$user",
})


def private_path_lines(content: bytes) -> list[int]:
    """返回命中的行号；不把个人路径塞进日志或异常。"""
    if b"\0" in content:
        return []
    text = content.decode("utf-8", errors="replace")
    lines = set()
    for match in PROFILE.finditer(text):
        name = match.group(1)
        if name.lower() in PLACEHOLDERS or (name.startswith("<") and name.endswith(">")):
            continue
        lines.add(text.count("\n", 0, match.start()) + 1)
    return sorted(lines)


def git(repo: Path, *args: str) -> bytes:
    """用 argv 调 Git，避免拼接 shell 命令。"""
    return subprocess.check_output(["git", "-C", str(repo), *args])


def tree_entries(repo: Path, ref: str) -> list[tuple[str, str]]:
    """返回一个提交树中的 (blob oid, 路径)。"""
    entries = []
    for row in git(repo, "ls-tree", "-r", "-z", ref).split(b"\0"):
        if not row:
            continue
        head, name = row.split(b"\t", 1)
        _mode, kind, oid = head.split()
        if kind == b"blob":
            entries.append((oid.decode("ascii"), name.decode("utf-8")))
    return entries


def read_blobs(repo: Path, entries: Iterable[tuple[str, str]]) -> Iterator[tuple[str, bytes]]:
    """批量读取 blob，不检出文件或修改仓库。"""
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        for oid, name in entries:
            proc.stdin.write(oid.encode("ascii") + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise RuntimeError("无法读取待检查的 Git blob")
            body = proc.stdout.read(int(header[2]))
            if proc.stdout.read(1) != b"\n":
                raise RuntimeError("Git blob 数据边界异常")
            yield name, body
    finally:
        proc.stdin.close()
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError("Git blob 检查进程失败")


def history_entries(repo: Path) -> list[tuple[str, str]]:
    """每个历史 blob 只查一次，保留 Git 提供的代表路径。"""
    objects = git(repo, "rev-list", "--objects", "--all").splitlines()
    entries = []
    for row in objects:
        parts = row.split(b" ", 1)
        if len(parts) == 2:
            oid, name = parts
            if git(repo, "cat-file", "-t", oid.decode("ascii")).strip() == b"blob":
                entries.append((oid.decode("ascii"), name.decode("utf-8")))
    return entries


def main(argv: list[str] | None = None) -> int:
    """检查工作区、HEAD 或本地引用可达历史。"""
    parser = argparse.ArgumentParser(description="检查个人 Windows 用户路径（日志不回显内容）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--worktree", action="store_true")
    group.add_argument("--history", action="store_true")
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args(argv)
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True,
    ).strip())
    if args.worktree:
        names = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        files = ((name.decode("utf-8"), (repo / name.decode("utf-8")).read_bytes())
                 for name in names.split(b"\0") if name and (repo / name.decode("utf-8")).is_file())
    else:
        entries = history_entries(repo) if args.history else tree_entries(repo, args.ref)
        files = read_blobs(repo, entries)
    scanned, hits = 0, 0
    for name, content in files:
        scanned += 1
        for line in private_path_lines(content):
            print(f"{name}:{line}: private user-profile path (content hidden)")
            hits += 1
    print(f"Checked {scanned} files/blobs; private-path findings: {hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
