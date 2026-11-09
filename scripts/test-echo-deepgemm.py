#!/usr/bin/env python3
"""Host checks for the actual ECHO Ascend selection/staging implementation."""
from pathlib import Path
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
DG = ROOT / "deepgemm-ascend"
MOCK = DG / "tests/echo_mock"


def main():
    with tempfile.TemporaryDirectory(prefix="echo-deepgemm-test-") as tmp:
        binary = Path(tmp) / "echo_prefetch_test"
        subprocess.run([
            "g++", "-std=c++20", "-O1", "-g", "-pthread", "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer", "-I", str(MOCK), "-I", str(DG / "deep_gemm/include"),
            str(MOCK / "echo_prefetch_test.cpp"), "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True, timeout=90)
    subprocess.run([sys.executable, str(DG / "tests/test_echo_wrapper_cpu.py")], check=True)


if __name__ == "__main__":
    main()
