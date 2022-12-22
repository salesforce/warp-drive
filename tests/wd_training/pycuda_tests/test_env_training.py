# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os
import unittest
import warnings

import torch
import yaml

from warp_drive.training.example_training_script_pycuda import setup_trainer_and_train
from warp_drive.training.utils.device_child_process.child_process_base import ProcessWrapper
from warp_drive.training.utils.distributed_train.distributed_trainer_pycuda import (
    perform_distributed_training,
)
from warp_drive.utils.common import get_project_root

_ROOT_DIR = get_project_root()


def launch_process(func, kwargs):
    """
    Run a Python function on a separate process.
    """
    p = ProcessWrapper(target=func, kwargs=kwargs)
    p.start()
    p.join()
    if p.exception:
        raise p.exception


class MyTestCase(unittest.TestCase):
    """
    Test the end-to-end training loop
    """

    num_gpus = torch.cuda.device_count()

    def get_config(self, env_name):
        config_path = os.path.join(
            _ROOT_DIR, "warp_drive", "training", "run_configs", f"{env_name}.yaml"
        )
        if not os.path.exists(config_path):
            raise ValueError(
                "Invalid environment specified! The environment name should "
                "match the name of the yaml file in training/run_configs/."
            )

        with open(config_path, "r", encoding="utf8") as fp:
            run_config = yaml.safe_load(fp)
        return run_config

    def test_tag_gridworld_training(self):
        run_config = self.get_config("tag_gridworld")
        try:
            launch_process(
                setup_trainer_and_train,
                kwargs={"run_configuration": run_config, "verbose": False},
            )
        except AssertionError:
            self.fail("TagGridWorld environment training failed")

    def test_tag_continuous_training(self):
        run_config = self.get_config("tag_continuous")
        try:
            run_config["env"]["num_taggers"] = 2
            run_config["env"]["num_runners"] = 10
            launch_process(
                setup_trainer_and_train,
                kwargs={"run_configuration": run_config, "verbose": False},
            )
        except AssertionError:
            self.fail("TagContinuous environment training failed")

    def test_tag_gridworld_training_with_multiple_devices(self):
        if self.num_gpus <= 1:
            warnings.warn(
                "Only single GPU is detected, we skip trainer test for multiple devices"
            )
            return

        run_config = self.get_config("tag_gridworld")
        run_config["trainer"]["num_gpus"] = self.num_gpus
        try:
            perform_distributed_training(setup_trainer_and_train, run_config)
        except AssertionError:
            self.fail(
                f"TagGridWorld environment training failed for {self.num_gpus} GPUs"
            )
