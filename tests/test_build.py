# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import runpy
import shutil
import subprocess
import sys
import sysconfig
import types

import pytest


@pytest.mark.parametrize(
    ("platform", "wheel_platform", "hip"),
    [
        ("win32", "win-amd64", None),
        ("win32", "win-arm64", None),
        ("linux", "linux-x86_64", None),
        ("linux", "linux-aarch64", "7.0"),
    ],
)
def test_native_extension_configuration(monkeypatch, platform, wheel_platform, hip):
    captured = {}
    setuptools = types.ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = object()
    cuda_lib = pathlib.Path("CUDA") / "lib" / "x64"
    torch_lib = pathlib.Path("torch") / "lib"

    def cuda_extension(**kwargs):
        return types.SimpleNamespace(
            **kwargs, library_dirs=[str(torch_lib), str(cuda_lib)]
        )

    cpp_extension.CUDAExtension = cuda_extension
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["setup.py", "bdist_wheel"])
    monkeypatch.setattr(sysconfig, "get_platform", lambda: wheel_platform)
    monkeypatch.chdir(pathlib.Path(__file__).resolve().parents[1])

    runpy.run_path("setup.py")

    extension = captured["ext_modules"][0]
    cxx = extension.extra_compile_args["cxx"]
    nvcc = extension.extra_compile_args["nvcc"]
    if platform == "win32":
        assert "/O2" in cxx and "/std:c++17" in cxx
        assert "-O3" not in cxx and "-std=c++17" not in cxx
        assert "/Zc:preprocessor" in cxx
        assert nvcc[-2:] == ["-Xcompiler", "/Zc:preprocessor"]
    else:
        assert cxx == ["-O3", "-std=c++17"]
        assert "-Xcompiler" not in nvcc
    assert ("--use_fast_math" in nvcc) == (hip is None)
    expected_cuda_lib = (
        cuda_lib.with_name("arm64") if wheel_platform == "win-arm64" else cuda_lib
    )
    assert extension.library_dirs == [str(torch_lib), str(expected_cuda_lib)]


@pytest.mark.parametrize(
    ("operating_system", "exit_code"),
    [("Linux", 0), ("Windows_NT", 0), ("Linux", 7)],
)
def test_release_helper(tmp_path, operating_system, exit_code):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("The release helper requires Bash")
    tools = tmp_path / "tools"
    tools.mkdir()
    msvc = tmp_path / "msvc"
    msvc.mkdir()
    for command in (tools / "link.exe", msvc / "link.exe", msvc / "cl.exe"):
        command.write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
        command.chmod(0o755)
    scripts = {
        "nvcc": "#!/usr/bin/env bash\necho 'release 13.4'\n",
        "uv": (
            "#!/usr/bin/env bash\n"
            'printf "arches=%s\\n" "$TORCH_CUDA_ARCH_LIST"\n'
            'printf "linker=<%s>\\n" "$(command -v link.exe)"\n'
            'printf "arg=<%s>\\n" "$@"\n'
            f"exit {exit_code}\n"
        ),
    }
    for name, contents in scripts.items():
        command = tools / name
        command.write_text(contents, encoding="utf-8", newline="\n")
        command.chmod(0o755)
    helper = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_release_wheel.sh"
    )
    env = dict(os.environ)
    env.update(
        PATH=os.pathsep.join((str(tools), str(msvc), env["PATH"])),
        TORCH_CUDA_ARCH_LIST="12.1+PTX",
        OS=operating_system,
    )
    result = subprocess.run(
        [bash, str(helper), "-o", "wheel output"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    assert "arches=12.1+PTX" in result.stdout
    if operating_system == "Windows_NT":
        assert "/msvc/link.exe>" in result.stdout.replace("\\", "/")
    assert (
        "arg=<--wheel>\narg=<--no-build-isolation>\narg=<-o>\narg=<wheel output>"
        in result.stdout
    )
