# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


def _linear_interpolation(l_v, r_v, slope):
    """linear interpolation between l_v and r_v with a slope"""
    return l_v + slope * (r_v - l_v)


class ParamScheduler:
    """
    A scheduler for the adapting parameters such as
    learning rate and entropy coefficient.
    Available scheduler types are ["constant", "piecewise_linear"].
    """

    def __init__(self, schedule):
        """
        schedule: schedule for how to vary the parameter.
        Types of parameter schedules:
        - constant: a constant parameter throughout training
        - piecewise_linear: the schedule may be specified as a
            list of lists.
            Note: Each entry in the schedule must be a list with
            signature (timestep, parameter value),and the times need to
            be in an increasing order. The parameter values are
            linearly interpolated between boundaries.
            e.g. schedule = [[1000, 0.1], [2000, 0.05]] implements a schedule
               [0.1 if t <= 1000,
               0.05 if t > 2000,
               and linearly interpolated between 1000 and 2000 steps.]
               For instance, the value at 1500 steps will equal 0.075.
        """
        if isinstance(schedule, (int, float)):
            # The schedule corresponds to the param value itself.
            self.type = "constant"
        elif isinstance(schedule, list):
            self.type = "piecewise_linear"
            # The schedule itself is a list.
            # Each item in the schedule must be a list of [time, param_value].
            # Note: The times must be specified in an increasing (sorted) order.
            assert isinstance(schedule, list), (
                "Please specify " "the schedule as a list of tuples!"
            )
            for item in schedule:
                assert isinstance(item, list), (
                    "Each entry in the schedule must"
                    " be a list with signature "
                    "[time, param_value]."
                )
            times = [item[0] for item in schedule]
            assert times == sorted(times), (
                "All the times must be sorted in" " an increasing order!"
            )
        else:
            raise NotImplementedError
        self.schedule = schedule

    def get_param_value(self, timestep):
        """Obtain the parameter value at a desired timestep."""
        assert timestep >= 0
        if self.type == "constant":
            param_value = self.schedule
        elif self.type == "piecewise_linear":
            if timestep <= self.schedule[0][0]:
                param_value = self.schedule[0][1]
            elif timestep >= self.schedule[-1][0]:
                param_value = self.schedule[-1][1]
            else:
                for (l_t, l_v), (r_t, r_v) in zip(
                    self.schedule[:-1], self.schedule[1:]
                ):
                    if l_t <= timestep < r_t:
                        slope = float(timestep - l_t) / (r_t - l_t)
                        param_value = _linear_interpolation(l_v, r_v, slope)
        else:
            raise NotImplementedError
        return param_value
