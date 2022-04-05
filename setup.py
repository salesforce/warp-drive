# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rl-warp-drive",
    version="1.6",
    author="Tian Lan, Sunil Srinivasa, Stephan Zheng",
    author_email="stephan.zheng@salesforce.com",
    description="Framework for fast end-to-end "
    "multi-agent reinforcement learning on GPUs.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/salesforce/warp-drive",
    license="",
    packages=find_packages(),
    package_data={
        "example_envs": ["tag_continuous/*.cu", "tag_gridworld/*.cu", "dummy_env/*.cu"],
        "warp_drive": [
            "cuda_includes/*",
            "cuda_includes/core/*.*",
            "training/run_configs/default_configs.yaml",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.0",
)
