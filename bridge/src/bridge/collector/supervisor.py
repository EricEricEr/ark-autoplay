"""采集监督器：系统故障等待恢复，逐份失败继续，持久收据防止断电后重复开打。"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bridge.candidate_coverage import execution_key
from bridge.data_paths import data_path

from .resilience import NeedsUser, account_key, reconcile, unhealthy, writable_probe
from .storage import Settings, append_json, now, process_birth, read_json, write_json
from .task_package import load_queue


def run(spec_path: Path, runtime: Any) -> int:
    """隔离运行全部任务，恢复只重试未完成项，不按连续失败数停止。"""
    spec = read_json(spec_path)
    settings = Settings(**spec["settings"])
    if not settings.demo:
        data_path(settings.data_dir)
    run_dir, data = Path(spec["run_dir"]), Path(settings.data_dir)
    account = account_key(settings)
    ledger = data / (
        "demo_results.jsonl" if settings.demo else "collection_results.jsonl"
    )
    state_file = run_dir / "status.json"
    live_file = Path(spec.get("live_file", state_file))
    state: dict = {
        "pid": os.getpid(),
        "process_birth": process_birth(os.getpid()),
        "run_token": spec["run_token"],
        "status": "starting",
        "started_at": now(),
        "message": "正在核对未完成任务",
        "total": 0,
        "attempted": 0,
        "valid": 0,
        "review": 0,
        "demo": settings.demo,
    }
    lock, finished, storage_failed = (
        threading.Lock(),
        threading.Event(),
        threading.Event(),
    )
    storage_errors: list[Exception] = []
    child = None
    lease = None
    profile_lease = None
    group = runtime.ProcessTree()

    def update(**values):
        with lock:
            state.update(values)

    def publish():
        with lock:
            snapshot = {**state, "heartbeat_at": now(), "heartbeat_epoch": time.time()}
        write_json(state_file, snapshot)
        if live_file != state_file:
            write_json(live_file, snapshot)

    def pulse():
        while not finished.is_set():
            try:
                publish()
            except OSError as error:
                storage_errors.append(error)
                storage_failed.set()
                return
            finished.wait(1)

    def check_storage():
        if storage_failed.is_set():
            raise OSError(f"数据保存失败：{storage_errors[-1]}")
        writable_probe(data)

    def stopping():
        if storage_failed.is_set():
            return True
        return runtime.control(run_dir).startswith("stop")

    def wait_user(message):
        challenge = uuid.uuid4().hex
        update(status="needs_user", message=message, challenge=challenge)
        # 绑定本次提示，不依赖系统时钟，历史确认不能自动放行。
        while not stopping():
            value = read_json(run_dir / "control.json", {}) or {}
            if (
                value.get("command") == "resume_checked"
                and value.get("challenge") == challenge
            ):
                runtime.prepare_device(settings, update, stopping)
                if stopping():
                    return
                if settings.demo or home_visible(settings, runtime):
                    update(
                        status="recovering", message="已确认回到主页，正在恢复本份任务"
                    )
                    return
                update(
                    message="尚未识别到游戏主页，请完成登录、更新或弹窗处理后再次点击确认"
                )
                challenge = uuid.uuid4().hex
                update(challenge=challenge)
            time.sleep(0.2)

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    try:
        if spec.get("profile_home"):
            profile_lease = runtime.DeviceLease(f"profile:{spec['profile_home']}")
            if not profile_lease.acquire():
                raise NeedsUser("本档案已有任务，请回到原窗口查看")
        check_storage()
        if spec.get("conversion_only"):
            from .conversion import run_conversion

            run_conversion(data, run_dir, update, stopping, group, runtime)
            return 0
        # 稳定实例编号不随ADB端口变化；在任何启动或输入操作前占用设备。
        identity = (
            f"demo:{spec['run_token']}"
            if settings.demo
            else f"mumu:{Path(settings.mumu_manager).resolve()}:"
            f"{settings.instance_index}"
            if settings.instance_index
            else settings.device_addr
        )
        lease = runtime.DeviceLease(identity)
        if not lease.acquire():
            raise NeedsUser("所选模拟器已有采集任务，请回到原窗口查看进度")
        _, entries = load_queue(Path(settings.queue_file))
        previous = {
            row["execution_key"]: row
            for row in reconcile(data, account, demo=settings.demo)
            if row.get("collector_account") == account and row.get("execution_key")
        }
        if not spec.get("restart_all"):
            if spec.get("retry_failed"):
                entries = [
                    entry
                    for entry in entries
                    if execution_key(entry["source"]) in previous
                    and not previous[execution_key(entry["source"])].get(
                        "valid_collected_win"
                    )
                    and not previous[execution_key(entry["source"])].get("demo_passed")
                ]
            else:
                entries = [
                    entry
                    for entry in entries
                    if execution_key(entry["source"]) not in previous
                ]
        update(total=len(entries))
        roster = None
        if settings.roster_file and Path(settings.roster_file).is_file():
            import hashlib

            digest = hashlib.sha256(Path(settings.roster_file).read_bytes()).hexdigest()
            roster = data / "roster_snapshots" / f"{digest}.json"
            roster.parent.mkdir(parents=True, exist_ok=True)
            if not roster.exists():
                shutil.copy2(settings.roster_file, roster)
        for index, entry in enumerate(entries):
            completed = False
            while not completed and not stopping():
                while runtime.control(run_dir) == "pause_after" and not stopping():
                    update(status="paused", message="已暂停，点击继续即可恢复")
                    time.sleep(0.2)
                if stopping():
                    break
                check_storage()
                try:
                    runtime.prepare_device(settings, update, stopping)
                    if stopping():
                        break
                    runtime.start_game_if_needed(settings)
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                    check_storage()
                    if isinstance(error, NeedsUser) or not settings.automatic_recovery:
                        wait_user(str(error))
                    else:
                        update(
                            status="waiting_device",
                            message="设备未响应，已保留本份任务并重试连接",
                        )
                        time.sleep(1)
                    continue
                instance = (
                    run_dir / "demo.local.yaml"
                    if settings.demo
                    else runtime.instance_config(settings, data)
                )
                if roster:
                    entry["source"]["collector_roster_file"] = roster.relative_to(
                        data
                    ).as_posix()
                attempt = uuid.uuid4().hex
                entry["source"]["collector_attempt_id"] = attempt
                child_spec = run_dir / "jobs" / f"j{index}-{attempt[:8]}.json"
                result = child_spec.with_suffix(".result.json")
                health_file = live_file.parent / f"health-{attempt}.json"
                write_json(
                    child_spec,
                    {
                        "settings": asdict(settings),
                        "entry": entry,
                        "run_dir": str(run_dir),
                        "instance_file": str(instance),
                        "result_file": str(result),
                        "ordinal": index,
                        "health_file": str(health_file),
                        "attempt_id": attempt,
                    },
                )
                update(
                    status="running",
                    message="正在重放作业并保存数据",
                    current_stage=entry["maa_job"]["stage_name"],
                    current_job=str(entry["source"]["job_id"]),
                    position=index + 1,
                )
                reason = ""
                launched_at = last_probe = time.time()
                with child_spec.with_suffix(".log").open("ab", buffering=0) as log:
                    child = runtime.subprocess.Popen(
                        runtime.command_for("--episode", str(child_spec)),
                        cwd=data,
                        env=child_env(),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    group.assign(child)
                    stop_at = None
                    while child.poll() is None:
                        check_storage_error(storage_failed, storage_errors)
                        command = runtime.control(run_dir)
                        if time.time() - last_probe > 5:
                            check_storage()
                            last_probe = time.time()
                            reason = runtime.environment_failure(settings)
                            if reason:
                                runtime.stop_owned_process(child)
                                break
                        reason = unhealthy(health_file, launched_at)
                        if reason:
                            runtime.stop_owned_process(child)
                            break
                        notice = (
                            read_json(child_spec.with_suffix(".notice.json"), {}) or {}
                        )
                        if notice.get("at", 0) > time.time() - 10:
                            update(message=notice["message"])
                        if command in ("pause_after", "stop_after"):
                            update(message="当前局保存完成后暂停或停止")
                        if command == "stop_now":
                            stop_at = stop_at or time.monotonic()
                            if time.monotonic() - stop_at >= 12:
                                runtime.stop_owned_process(child)
                        time.sleep(0.2)
                try:
                    summary = read_json(result)
                except (OSError, ValueError):
                    summary = None
                if summary is None:
                    summary = {
                        "ok": False,
                        "battle_result": "unknown",
                        "fail_stage": "process_exit",
                        "error": reason or "采集子进程异常退出，原始数据已保留",
                        "worker_exit_code": child.returncode,
                    }
                    fatal = read_json(child_spec.with_suffix(".fatal.json"), {}) or {}
                    if fatal.get("error_type") in (
                        "ModuleNotFoundError",
                        "ImportError",
                    ):
                        summary.update(
                            needs_user=True,
                            retry_pending=True,
                            error="采集组件缺失，请先停止任务并换用完整软件包，再检查续跑。",
                            diagnostic_error=fatal.get("error"),
                        )
                summary.update(
                    execution_key=execution_key(entry["source"]),
                    job_id=entry["source"]["job_id"],
                    stage_code=entry["maa_job"]["stage_name"],
                    collector_account=account,
                    completed_at=now(),
                    attempt_id=attempt,
                    receipt_complete=True,
                )
                if summary.get("storage_error"):
                    raise OSError(summary.get("error", "采集数据无法写入"))
                if runtime.control(run_dir) == "stop_now":
                    summary.update(retry_pending=True, interrupted=True)
                if summary.get("needs_user"):
                    summary["retry_pending"] = True
                    write_json(result, summary)
                    wait_user(summary.get("error", "请处理模拟器并回到主页"))
                    continue
                if summary.get("retry_before_battle"):
                    summary["retry_pending"] = True
                    write_json(result, summary)
                    append_json(run_dir / "recovery_events.jsonl", summary)
                    update(
                        status="waiting_device",
                        message="连接中断，本份任务保留，正在恢复",
                    )
                    if not settings.automatic_recovery:
                        wait_user("请连接模拟器并回到主页后继续")
                    continue
                write_json(result, summary)
                if summary.get("retry_pending"):
                    break
                append_json(ledger, summary)
                valid = bool(
                    summary.get("demo_passed")
                    if settings.demo
                    else summary.get("valid_collected_win")
                )
                update(
                    attempted=state["attempted"] + 1,
                    valid=state["valid"] + valid,
                    review=state["review"] + (not valid),
                    last_result=summary,
                )
                completed = True
        if storage_failed.is_set():
            check_storage()
        if stopping():
            update(status="stopped", message="已停止，未完成任务和原件保留")
        else:
            update(
                status="finished",
                message="本轮任务已完成，未通过项保留在待复核列表",
                pending=0,
            )
        if (
            settings.convert_after
            and not settings.demo
            and state["status"] == "finished"
        ):
            from .conversion import run_conversion

            run_conversion(data, run_dir, update, stopping, group, runtime)
        return 0
    except NeedsUser as error:
        update(status="needs_user", message=str(error))
        return 1
    except KeyboardInterrupt:
        update(status="stopped", message="采集已中断，原始数据保留")
        return 0
    except Exception as error:
        update(
            status="storage_error" if isinstance(error, OSError) else "error",
            message=(
                "无法保存数据，采集已停止。请恢复磁盘空间或写入权限后点击‘检查并恢复’。"
                if isinstance(error, OSError)
                else str(error)
            ),
            error_type=type(error).__name__,
        )
        return 1
    finally:
        if child is not None and child.poll() is None:
            runtime.stop_owned_process(child)
        group.close()
        if lease:
            lease.close()
        if profile_lease:
            profile_lease.close()
        finished.set()
        heartbeat.join(timeout=6)
        update(finished_at=now())
        try:
            publish()
        except OSError:
            # 数据盘不可写时仍尝试向界面档案报告错误；不继续开打。
            with contextlib.suppress(OSError):
                write_json(
                    live_file,
                    {**state, "heartbeat_epoch": time.time(), "heartbeat_at": now()},
                )


def check_storage_error(failed, errors):
    """心跳线程写盘失败必须传回监督线程，不能仅停止进度更新。"""
    if failed.is_set():
        raise OSError(str(errors[-1]))


def child_env():
    """复用源码与打包版的同一套环境传递。"""
    from .service import child_environment

    return child_environment()


def home_visible(settings, runtime) -> bool:
    """人工处理后仅在真实主页证据成立时续跑，不盲点未知弹窗。"""
    import io

    from PIL import Image

    from bridge.battle_lifecycle import BattleLifecycleObserver

    frame = runtime.run_command(
        [settings.adb_path, "-s", settings.device_addr, "exec-out", "screencap", "-p"]
    )
    if frame.returncode or not frame.stdout:
        return False
    return (
        BattleLifecycleObserver(settings.maa_dir)
        .observe(Image.open(io.BytesIO(frame.stdout)))
        .home
    )
