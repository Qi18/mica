#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPECTED_GPU_COUNT = int(os.environ.get("EXPECTED_GPU_COUNT", "8"))
MIN_SHM_BYTES = int(os.environ.get("MIN_SHM_BYTES", str(800 * 1024**3)))
REQUIRE_SWANLAB = os.environ.get("REQUIRE_SWANLAB", "1") == "1"


def discover_executable(environment_name: str, preferred: str, fallback: str) -> Path:
    configured = os.environ.get(environment_name)
    if configured:
        return Path(configured)
    preferred_path = Path(preferred)
    if preferred_path.is_file():
        return preferred_path
    resolved = shutil.which(fallback)
    return Path(resolved) if resolved else preferred_path


MINIMIND_PYTHON = discover_executable("MINIMIND_PYTHON", "/data/venvs/minimind-lab/bin/python", "python3")
SWANLAB_BIN = discover_executable("SWANLAB_BIN", "/data/venvs/minimind-lab/bin/swanlab", "swanlab")


def fail(message: str) -> None:
    raise SystemExit(f"preflight_error={message}")


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> None:
    for command_name in ("nvidia-smi", "findmnt"):
        if shutil.which(command_name) is None:
            fail(f"missing command: {command_name}")
    if not MINIMIND_PYTHON.is_file():
        fail(f"MiniMind Python not found: {MINIMIND_PYTHON}")
    if not (ROOT_DIR / "model" / "model_mica.py").is_file():
        fail("run from the Mica repository")

    gpu_lines = run(["nvidia-smi", "-L"]).stdout.strip().splitlines()
    if len(gpu_lines) != EXPECTED_GPU_COUNT:
        fail(f"expected {EXPECTED_GPU_COUNT} GPUs, found {len(gpu_lines)}")

    process_query = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    compute_processes = process_query.stdout.strip()
    if compute_processes:
        print(compute_processes, file=sys.stderr)
        fail("GPU compute processes already exist")

    cuda_probe = run(
        [
            str(MINIMIND_PYTHON),
            "-c",
            (
                "import json,torch;"
                "assert torch.cuda.is_available();"
                f"assert torch.cuda.device_count()=={EXPECTED_GPU_COUNT};"
                "assert torch.cuda.is_bf16_supported();"
                "x=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4);"
                "y=x@x.T;torch.cuda.synchronize();"
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda_runtime':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'gpu_count':torch.cuda.device_count(),"
                "'gpu0':torch.cuda.get_device_name(0),"
                "'bf16_supported':torch.cuda.is_bf16_supported(),"
                "'compute_sum':float(y.sum().item())}))"
            ),
        ]
    )

    mount = run(["findmnt", "-T", "/data", "-n", "-o", "FSTYPE,OPTIONS"])
    mount_fields = mount.stdout.strip().split(maxsplit=1)
    if len(mount_fields) != 2 or "rw" not in mount_fields[1].split(","):
        fail("/data is not a read-write mount")

    probe_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".minimind-lab-preflight.", dir="/data", delete=False
        ) as probe:
            probe.write("ok\n")
            probe_path = probe.name
        if Path(probe_path).read_text(encoding="utf-8") != "ok\n":
            fail("CPFS write/read probe failed")
    finally:
        if probe_path:
            Path(probe_path).unlink(missing_ok=True)

    shm_bytes = shutil.disk_usage("/dev/shm").total
    if shm_bytes < MIN_SHM_BYTES:
        fail(f"/dev/shm too small: expected {MIN_SHM_BYTES}, found {shm_bytes}")

    if REQUIRE_SWANLAB:
        if not SWANLAB_BIN.is_file():
            fail(f"SwanLab CLI not found: {SWANLAB_BIN}")
        swanlab = run([str(SWANLAB_BIN), "verify"], check=False)
        if swanlab.returncode != 0:
            fail("SwanLab login is not valid")

    print(cuda_probe.stdout.strip())
    print(
        json.dumps(
            {
                "preflight": "pass",
                "gpu_count": len(gpu_lines),
                "cpfs_type": mount_fields[0],
                "shm_bytes": shm_bytes,
                "swanlab_required": REQUIRE_SWANLAB,
            }
        )
    )


if __name__ == "__main__":
    main()
