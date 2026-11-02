#!/usr/bin/env python3
"""Rebuild local Torch registration objects using prepared ECHO dependency objects."""

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-build", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_build.resolve()
    target = ROOT / "build/copysfa-prefetch-v2/binding"
    target.mkdir(parents=True, exist_ok=True)
    flags = {}
    for line in (
        (reference / "CMakeFiles/vllm_ascend_C.dir/flags.make").read_text().splitlines()
    ):
        if line.startswith("CXX_"):
            name, value = line.split(" = ", 1)
            flags[name] = shlex.split(value)
    compiler = [
        "/usr/bin/c++",
        *flags["CXX_DEFINES"],
        *flags["CXX_INCLUDES"],
        *flags["CXX_FLAGS"],
    ]
    replacements = {}
    for name in ("torch_binding.cpp", "torch_binding_meta.cpp"):
        source = ROOT / "vllm-ascend/csrc" / name
        obj = target / (name + ".o")
        if not obj.exists() or obj.stat().st_mtime < max(
            path.stat().st_mtime
            for path in (ROOT / "vllm-ascend/csrc").rglob("*")
            if path.is_file()
            and path.suffix in (".cpp", ".h")
            and "build" not in path.parts
        ):
            subprocess.run([*compiler, "-c", str(source), "-o", str(obj)], check=True)
        replacements[f"CMakeFiles/vllm_ascend_C.dir/csrc/{name}.o"] = str(obj)
    link = shlex.split(
        (reference / "CMakeFiles/vllm_ascend_C.dir/link.txt").read_text()
    )
    output = target / "vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
    for i, arg in enumerate(link):
        if i and link[i - 1] == "-o":
            link[i] = str(output)
        elif arg in replacements:
            link[i] = replacements[arg]
        elif arg.endswith((".o", ".so")) and not arg.startswith(("/", "-")):
            link[i] = str(reference / arg)
        elif arg.startswith("-Wl,--dependency-file="):
            link[i] = "-Wl,--dependency-file=" + str(target / "link.d")
        elif arg == "-flto=auto":
            link[i] = "-flto=4"
    link = [arg.replace("\\$ORIGIN", "$ORIGIN") for arg in link]
    subprocess.run(link, cwd=target, check=True)
    print(output)


if __name__ == "__main__":
    main()
