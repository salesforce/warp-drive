# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import subprocess
import unittest


class MyTestCase(unittest.TestCase):
    """
    Test the end-to-end training loop
    """

    def test_tag_gridworld_training(self):
        try:
            subprocess.call(
                [
                    "python",
                    "warp_drive/training/example_training_script",
                    "-e",
                    "tag_gridworld",
                ]
            )
        except AssertionError:
            self.fail("TagGridWorld environment training failed")

    def test_tag_continuous_training(self):
        try:
            subprocess.call(
                [
                    "python",
                    "warp_drive/training/example_training_script",
                    "-e",
                    "tag_continuous",
                ]
            )
        except AssertionError:
            self.fail("TagContinuous environment training failed")
