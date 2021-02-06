import torch
import glob
from os import path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"


def get_extension():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "ts_ops", "csrc")

    main_source = path.join(extensions_dir, "ops.cpp")
    sources = glob.glob(path.join(extensions_dir, "soft_nms", "*.cpp"))
    sources = [main_source] + sources

    include_dirs = [extensions_dir]
    extensions = [
        CppExtension(
            "ts_ops",
            sources,
            include_dirs=include_dirs
        )
    ]

    return extensions


setup(name="ts_ops",
      version="1.0",
      packages=("ts_ops",),
      description="soft-nms impl",
      install_requires=[],
      ext_modules=get_extension(),
      cmdclass={"build_ext": BuildExtension.with_options(
          no_python_abi_suffix=True)}
      )
