"""在项目构建环境生成便携式Windows采集软件，不打包账号、轨迹或游戏素材。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def main() -> int:
    """使用已获批准的PyInstaller，仅向指定的新构建目录写产物。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    args = parser.parse_args()
    bridge = Path(__file__).resolve().parents[1]
    collector_assets = bridge / "src/bridge/collector/resources"
    repo = bridge.parent
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "stage"
    stage.mkdir()
    node_version = subprocess.check_output(
        [str(args.node), "--version"], text=True
    ).strip()
    if (
        not node_version.startswith("v")
        or not node_version[1:].replace(".", "").isdigit()
    ):
        raise ValueError("Node版本格式异常")
    licenses = stage / "THIRD_PARTY_LICENSES"
    licenses.mkdir()
    url = f"https://raw.githubusercontent.com/nodejs/node/{node_version}/LICENSE"
    request = urllib.request.Request(url, headers={"User-Agent": "ark-collector-build"})
    with urllib.request.urlopen(request, timeout=60) as response:
        node_license = response.read()
    (licenses / "Node.txt").write_bytes(node_license)
    shutil.copy2(
        bridge / "tools/skland/lib/vendor/LICENSE", licenses / "QRCode-MIT.txt"
    )
    shutil.copy2(repo / "LICENSE", stage / "LICENSE.txt")
    shutil.copy2(bridge / "NOTICE.md", stage / "NOTICE.txt")
    if (bridge / "docs/collector-app.md").exists():
        shutil.copy2(bridge / "docs/collector-app.md", stage / "开始使用.txt")
    source_bundle = stage / "source"
    source_bundle.mkdir()
    source_project = source_bundle / "bridge"
    source_project.mkdir()
    shutil.copy2(repo / "LICENSE", source_bundle / "LICENSE")
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "NOTICE.md",
        "collector-build-requirements.txt",
    ):
        if (bridge / name).exists():
            shutil.copy2(bridge / name, source_project / name)
    shutil.copytree(
        bridge / "src/bridge",
        source_project / "src/bridge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(bridge / "tools/skland", source_project / "tools/skland")
    shutil.copytree(bridge / "docs", source_project / "docs")
    shutil.copy2(
        bridge / "tools/build_collector.py", source_project / "tools/build_collector.py"
    )
    shutil.copy2(
        bridge / "tools/collector_entry.py", source_project / "tools/collector_entry.py"
    )
    shutil.copy2(
        bridge / "collector-build-requirements.txt",
        source_project / "requirements-build.txt",
    )
    shutil.copytree(
        bridge / "configs",
        source_project / "configs",
        ignore=shutil.ignore_patterns(
            "instances",
            "jobs",
            "*.local.*",
            "jobs_main_v1.json",
            "loop_milestone_v1.json",
        ),
    )
    for package in ("numpy", "Pillow", "PyYAML"):
        distribution = importlib.metadata.distribution(package)
        for item in distribution.files or []:
            if item.name.casefold().startswith(("license", "copying")):
                license_path = distribution.locate_file(item)
                destination = licenses / package / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(license_path, destination)
    for name, source in (
        ("Python.txt", Path(sys.base_prefix) / "LICENSE.txt"),
        ("Tcl.txt", Path(sys.base_prefix) / "tcl/tcl8.6/license.terms"),
        ("Tk.txt", Path(sys.base_prefix) / "tcl/tk8.6/license.terms"),
    ):
        if source.exists():
            shutil.copy2(source, licenses / name)
    icon = stage / "collector.ico"
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (256, 256), "#16243b")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 238, 238), radius=42, fill="#2563eb")
    draw.line([(62, 159), (62, 194), (194, 194), (194, 159)], fill="white", width=16)
    draw.line([(128, 55), (128, 153)], fill="white", width=18)
    draw.polygon([(82, 119), (128, 164), (174, 119)], fill="white")
    image.save(
        icon, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    pyinstaller_args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--windowed",
        "--onedir",
        "--noupx",
        "--name",
        "ArkCollector",
        "--icon",
        str(icon),
        "--paths",
        str(bridge / "src"),
        "--collect-data",
        "bridge",
        "--collect-submodules",
        "bridge.collector",
        "--hidden-import",
        "ctypes.util",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "ark_core",
        "--exclude-module",
        "onnxruntime",
        "--add-data",
        f"{bridge / 'configs/nav_main.yaml'}{os.pathsep}bridge/collector/resources",
        "--add-data",
        f"{bridge / 'src/bridge/resources'}{os.pathsep}bridge/resources",
        "--add-data",
        f"{collector_assets}{os.pathsep}bridge/collector/resources",
        "--add-data",
        f"{bridge / 'tools/skland'}{os.pathsep}tools/skland",
        "--add-binary",
        f"{args.node}{os.pathsep}tools",
        "--distpath",
        str(output / "dist"),
        "--workpath",
        str(output / "build"),
        "--specpath",
        str(output / "spec"),
        str(bridge / "tools/collector_entry.py"),
    ]
    env = os.environ.copy()
    cache = output / "cache"
    temporary = output / "temporary"
    cache.mkdir()
    temporary.mkdir()
    env.update(
        PYINSTALLER_CONFIG_DIR=str(cache), TEMP=str(temporary), TMP=str(temporary)
    )
    env["PYTHONPATH"] = str(bridge / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(pyinstaller_args, check=True, env=env)
    app = output / "dist/ArkCollector"
    for name in ("THIRD_PARTY_LICENSES", "source"):
        shutil.copytree(stage / name, app / name)
    for name in ("LICENSE.txt", "NOTICE.txt", "开始使用.txt"):
        if (stage / name).exists():
            shutil.copy2(stage / name, app / name)
    manifest = {}
    private_findings = []
    # 检查构建者个人目录字面量，不扫描/复制本机配置、日志或名册。
    personal_patterns = [
        str(Path.home()).encode(),
        str(Path.home()).replace("\\", "/").encode(),
        str(Path.home()).encode("utf-16-le"),
    ]
    for file in app.rglob("*"):
        if file.is_file():
            content = file.read_bytes()
            if any(needle in content for needle in personal_patterns):
                private_findings.append(file.relative_to(app).as_posix())
            manifest[file.relative_to(app).as_posix()] = hashlib.sha256(
                content
            ).hexdigest()
    report = {
        "schema": "collector_build/1",
        "pyinstaller": importlib.metadata.version("pyinstaller"),
        "node_version": node_version,
        "privacy_findings": private_findings,
        "files": manifest,
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if private_findings:
        raise RuntimeError(f"构建产物包含个人路径，停止交付：{private_findings}")
    archive = output / "方舟数据采集助手-Windows-x64.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in app.rglob("*"):
            if file.is_file():
                bundle.write(file, Path("ArkCollector") / file.relative_to(app))
    print(
        json.dumps(
            {
                "exe": str(app / "ArkCollector.exe"),
                "zip": str(archive),
                "bytes": archive.stat().st_size,
                "privacy_findings": private_findings,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
