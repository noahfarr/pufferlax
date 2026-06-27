import os
import subprocess
import sysconfig
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "pufferlax"
PUFFERLIB = ROOT / "vendor" / "pufferlib"
RAYLIB = "raylib-5.5_linux_amd64"


def cudart():
    site = Path(sysconfig.get_paths()["purelib"])
    home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    for lib in [site / "nvidia" / "cuda_runtime" / "lib", home / "lib64", home / "lib"]:
        sonames = sorted(path.name for path in lib.glob("libcudart.so*")) if lib.exists() else []
        if sonames:
            return str(lib), sonames[0]
    raise RuntimeError("libcudart not found (cudart wheel or CUDA_HOME)")


def build(env_name="craftax"):
    out = PACKAGE / f"ffi_{env_name}.so"
    build_dir = PACKAGE / "build"
    build_dir.mkdir(exist_ok=True)
    raylib = PUFFERLIB / RAYLIB
    cuda_lib, cuda_soname = cudart()

    env_object = build_dir / f"libstatic_{env_name}.o"
    archive = build_dir / f"libstatic_{env_name}.a"
    handler_object = build_dir / "ffi.o"

    subprocess.run(
        ["gcc", "-c", "-O2", "-DNDEBUG", "-mavx2", "-mfma",
         f"-I{PUFFERLIB}", f"-I{PUFFERLIB / 'src'}", f"-I{PUFFERLIB / 'ocean' / env_name}",
         f"-I{PUFFERLIB / 'vendor'}", f"-I{raylib / 'include'}",
         "-DPLATFORM_DESKTOP", "-fno-semantic-interposition", "-fvisibility=hidden",
         "-fPIC", "-fopenmp",
         str(PUFFERLIB / "ocean" / env_name / "binding.c"), "-o", str(env_object)],
        check=True,
    )
    subprocess.run(["ar", "rcs", str(archive), str(env_object)], check=True)
    subprocess.run(
        ["g++", "-c", "-O2", "-std=c++17", "-x", "c++", "-fPIC",
         f"-I{jax.ffi.include_dir()}", str(PACKAGE / "ffi.cu"), "-o", str(handler_object)],
        check=True,
    )
    subprocess.run(
        ["g++", "-shared", "-fPIC", "-fopenmp", str(handler_object), str(archive),
         str(raylib / "lib" / "libraylib.a"),
         f"-L{cuda_lib}", f"-l:{cuda_soname}", f"-Wl,-rpath,{cuda_lib}",
         "-lm", "-lpthread", "-O2", "-Bsymbolic-functions", "-o", str(out)],
        check=True,
    )
    return out


if __name__ == "__main__":
    import sys

    print("built", build(sys.argv[1] if len(sys.argv) > 1 else "craftax"))
