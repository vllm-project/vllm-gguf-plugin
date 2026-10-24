# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import sys

import tomllib
from setuptools import setup
from setuptools.command.egg_info import write_file
from setuptools.command.sdist import sdist


def _package_version() -> str:
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    version = project["tool"]["vllm_gguf_plugin"]["base_version"]
    suffix = os.environ.get("VLLM_GGUF_PLUGIN_LOCAL_VERSION_SUFFIX")
    if not suffix:
        return version
    normalized_suffix = suffix if suffix.startswith("+") else f"+{suffix}"
    return f"{version}{normalized_suffix}"


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    return not any(command in packaging_commands for command in sys.argv[1:])


UPSTREAM_ROOT = pathlib.Path("third_party/llama.cpp")
UPSTREAM_METADATA_FILE = pathlib.Path("vllm_gguf_plugin/llama_cpp_upstream.toml")
UPSTREAM_METADATA = tomllib.loads(UPSTREAM_METADATA_FILE.read_text())
UPSTREAM_COMMIT = UPSTREAM_METADATA["commit"]
UPSTREAM_CUDA_ROOT = UPSTREAM_ROOT / "ggml" / "src" / "ggml-cuda"
UPSTREAM_LICENSE = UPSTREAM_ROOT / "LICENSE"
DISTRIBUTED_UPSTREAM_LICENSE = pathlib.Path("LICENSES/llama.cpp-MIT.txt")
UPSTREAM_SOURCES = [
    UPSTREAM_ROOT / relative_path for relative_path in UPSTREAM_METADATA["sources"]
]
UPSTREAM_HEADERS = [
    UPSTREAM_ROOT / relative_path for relative_path in UPSTREAM_METADATA["headers"]
]
UPSTREAM_SDIST_FILES = [UPSTREAM_LICENSE, *UPSTREAM_SOURCES, *UPSTREAM_HEADERS]


def _check_upstream_checkout() -> None:
    if not UPSTREAM_ROOT.is_dir() or not all(
        path.is_file() for path in UPSTREAM_SDIST_FILES
    ):
        raise RuntimeError(
            "llama.cpp source files are missing. Initialize the pinned submodule with "
            "`git submodule update --init --recursive`, then rebuild."
        )
    if UPSTREAM_LICENSE.read_bytes() != DISTRIBUTED_UPSTREAM_LICENSE.read_bytes():
        raise RuntimeError(
            "The distributed llama.cpp license does not match the pinned submodule. "
            "Review the upstream license and update LICENSES/llama.cpp-MIT.txt."
        )

    # Keep the source closure complete as upstream adds quantized MMQ instances.
    # The metadata remains explicit so source archives are reproducible, while this
    # check prevents a new template instance from silently becoming unlinked.
    template_root = UPSTREAM_CUDA_ROOT / "template-instances"
    actual_template_instances = {
        path.relative_to(UPSTREAM_ROOT).as_posix()
        for pattern in ("mmq-instance-*.cu", "mmf-instance-ncols_*.cu")
        for path in template_root.glob(pattern)
    }
    listed_sources = {
        path.relative_to(UPSTREAM_ROOT).as_posix() for path in UPSTREAM_SOURCES
    }
    missing_template_instances = actual_template_instances - listed_sources
    extra_template_instances = {
        path
        for path in listed_sources
        if path.startswith("ggml/src/ggml-cuda/template-instances/")
        and path.endswith(".cu")
        and path.rsplit("/", 1)[-1].startswith(("mmq-instance-", "mmf-instance-ncols_"))
    } - actual_template_instances
    if missing_template_instances or extra_template_instances:
        raise RuntimeError(
            "llama.cpp template-instances source closure is out of sync; "
            f"missing={sorted(missing_template_instances)}, "
            f"extra={sorted(extra_template_instances)}"
        )

    # Source archives carry the selected files and metadata without Git state.
    if not (UPSTREAM_ROOT / ".git").exists():
        return
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Unable to verify the llama.cpp submodule revision."
        ) from error
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"llama.cpp submodule is at {revision}, expected {UPSTREAM_COMMIT}."
        )

    status = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if status.returncode == 1:
        raise RuntimeError("llama.cpp submodule has modified tracked files.")
    if status.returncode != 0:
        raise RuntimeError("Unable to verify that the llama.cpp submodule is clean.")


class _PinnedUpstreamSdist(sdist):
    def make_distribution(self) -> None:
        _check_upstream_checkout()
        selected = {path.as_posix() for path in UPSTREAM_SDIST_FILES}
        upstream_prefix = f"{UPSTREAM_ROOT.as_posix()}/"
        self.filelist.files[:] = [
            path
            for path in self.filelist.files
            if not path.startswith(upstream_prefix) or path in selected
        ]
        self.filelist.extend(sorted(selected))
        self.filelist.sort()
        self.filelist.remove_duplicates()
        egg_info = self.get_finalized_command("egg_info")
        write_file(os.path.join(egg_info.egg_info, "SOURCES.txt"), self.filelist.files)
        super().make_distribution()


setup_kwargs: dict = {
    "version": _package_version(),
    "cmdclass": {"sdist": _PinnedUpstreamSdist},
}

if _should_build_extension():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    is_rocm = getattr(torch.version, "hip", None) is not None
    legacy_only = is_rocm or os.environ.get("VLLM_GGUF_BUILD_LEGACY_ONLY", "0") == "1"

    if not legacy_only:
        _check_upstream_checkout()

    nvcc_args = [
        "-O3",
        "-std=c++17",
        # Exposes aoti_torch_get_current_cuda_stream in the AOTI shim.
        "-DUSE_CUDA",
    ]
    if not is_rocm:
        # hipcc (ROCm 7.x) rejects nvcc-only flags like --use_fast_math.
        nvcc_args.insert(2, "--use_fast_math")
        nvcc_args.extend(
            [
                # CUDAExtension defines these for PyTorch-owned kernels, but
                # llama.cpp CUDA templates require the native conversions.
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--extended-lambda",
                "-Xcompiler=-fvisibility=hidden",
                "-Xcompiler=-ffunction-sections",
            ]
        )
    if legacy_only:
        # Upstream CUDA entry points live in runtime_dense.cu and runtime_moe.cu.
        # Keep ROCm on the unchanged legacy implementation.
        nvcc_args.append("-DVLLM_GGUF_LEGACY_ONLY")

    cxx_args = [
        "-O3",
        "-std=c++17",
        "-fvisibility=hidden",
        "-ffunction-sections",
    ]
    if legacy_only:
        cxx_args.append("-DVLLM_GGUF_LEGACY_ONLY")

    sources = [
        "vllm_gguf_plugin/csrc/torch_bindings.cpp",
        "vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu",
    ]
    extra_link_args: list[str] = []
    if not legacy_only:
        sources.extend(
            [
                "vllm_gguf_plugin/csrc/upstream/runtime_dense.cu",
                "vllm_gguf_plugin/csrc/upstream/runtime_moe.cu",
                "vllm_gguf_plugin/csrc/upstream/runtime_dequant_blas.cu",
                "vllm_gguf_plugin/csrc/upstream/torch_context.cu",
                "vllm_gguf_plugin/csrc/upstream/runtime.cu",
                *(str(source) for source in UPSTREAM_SOURCES),
            ]
        )
        # Upstream headers take precedence; quoted legacy headers still
        # resolve next to gguf_kernel.cu. All paths must be absolute:
        # torch's ninja build runs from build/temp*, so relative -I flags
        # would silently fail to resolve (torch only absolute-izes
        # compiler.include_dirs, not the per-extension include_dirs).
        include_dirs = [
            str(path.resolve())
            for path in (
                UPSTREAM_ROOT / "ggml" / "include",
                UPSTREAM_ROOT / "ggml" / "src",
                UPSTREAM_CUDA_ROOT,
                pathlib.Path("vllm_gguf_plugin/csrc/upstream"),
                pathlib.Path("vllm_gguf_plugin/csrc"),
                pathlib.Path("vllm_gguf_plugin/csrc/gguf"),
            )
        ]
        extra_link_args = ["-Wl,--gc-sections", "-lcublas"]
    else:
        include_dirs = [
            str(path.resolve())
            for path in (
                pathlib.Path("vllm_gguf_plugin/csrc"),
                pathlib.Path("vllm_gguf_plugin/csrc/gguf"),
            )
        ]

    setup_kwargs.update(
        ext_modules=[
            CUDAExtension(
                name="vllm_gguf_plugin._C_gguf",
                sources=sources,
                include_dirs=include_dirs,
                py_limited_api=True,
                extra_compile_args={
                    "cxx": cxx_args,
                    "nvcc": nvcc_args,
                },
                extra_link_args=extra_link_args,
            )
        ],
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )
    setup_kwargs["cmdclass"]["build_ext"] = BuildExtension

setup(**setup_kwargs)
