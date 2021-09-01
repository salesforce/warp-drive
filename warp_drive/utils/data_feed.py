# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


class DataFeed(dict):
    """
    Helper class to build up the data dict for CUDADataManager.push_data_to_device(data)

    Example:
        data = DataFeed()
        data.add(name="X", data=[1,2,3], save_copy_and_apply_at_reset=True,
            log_data_across_episode=True)

    """

    def add_data(
        self,
        name: str,
        data,
        save_copy_and_apply_at_reset: bool = False,
        log_data_across_episode: bool = False,
        **kwargs
    ):
        """
        :param name: name of the data
        :param data: data in the form of list, array or scalar
        :param save_copy_and_apply_at_reset: the copy of the data will be saved
            inside GPU for the reset
        :param log_data_across_episode: an data buffer of episode length is
            reserved for logging data
        """
        d = {
            "data": data,
            "attributes": {
                "save_copy_and_apply_at_reset": save_copy_and_apply_at_reset,
                "log_data_across_episode": log_data_across_episode,
            },
        }
        for key, value in kwargs.items():
            d["attributes"][key] = value
        self[name] = d
