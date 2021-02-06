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
    sources = glob.glob(path.join(extensions_dir,  "*.cpp"))

    include_dirs = [extensions_dir]
    extensions = [
        CppExtension(
            "soft_nms",
            sources,
            include_dirs=include_dirs
        )
    ]

    return extensions


setup(name="ts_ops",
      version="0.1",
      packages=("ts_ops",),
      description="PyTorch implementation of the soft-nms algorithm",
      install_requires=[],
      ext_modules=get_extension(),
      cmdclass={"build_ext": BuildExtension.with_options(
          no_python_abi_suffix=True)}
      )
