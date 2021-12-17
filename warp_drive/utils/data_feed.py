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
        :param log_data_across_episode: a data buffer of episode length is
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

    def add_data_list(self, data_list):
        """
        :param data_list: list of data configures either in dict or in tuple
        for example
        add_data_list([("x1", x1, True),
                       ("x2", x2, False, True),
                       {"name": "x3",
                        "data": x3,
                        "save_copy_and_apply_at_reset": False},
                      ]

        """
        assert isinstance(data_list, list)
        for d in data_list:
            assert len(d) >= 2, "name and data are strictly required"

            if isinstance(d, tuple):
                name = d[0]
                assert isinstance(name, str)
                data = d[1]
                save_copy_and_apply_at_reset = (
                    d[2] if (len(d) > 2 and isinstance(d[2], bool)) else False
                )
                log_data_across_episode = (
                    d[3] if (len(d) > 3 and isinstance(d[3], bool)) else False
                )
                self.add_data(
                    name, data, save_copy_and_apply_at_reset, log_data_across_episode
                )
            elif isinstance(d, dict):
                self.add_data(
                    name=d["name"],
                    data=d["data"],
                    save_copy_and_apply_at_reset=d.get(
                        "save_copy_and_apply_at_reset", False
                    ),
                    log_data_across_episode=d.get("log_data_across_episode", False),
                )
            else:
                raise Exception(
                    "Unknown type of data configure, only support tuple and dictionary"
                )
