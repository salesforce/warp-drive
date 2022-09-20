import os

import numba.cuda as numba_driver
import torch.distributed as dist


class NumbaSingleDeviceContext:
    """
    one single GPU hardware context available for both Numba and PyTorch
    """

    _context = None

    def init_context(self, device_id=None):
        if device_id is None:
            context = numba_driver.current_context()
            self._context = context
        else:
            context = numba_driver.select_device(device_id).get_primary_context()
            self._context = context

    @property
    def context(self):
        return self._context

    def clear_context(self):
        numba_driver.close()

    def __getattribute__(self, name):
        if name in "init_context":
            return object.__getattribute__(self, name)

        if name in "clear_context":
            return object.__getattribute__(self, name)

        if object.__getattribute__(self, "_context") is None:
            raise RuntimeError("Context hasn't been initialized yet")

        return object.__getattribute__(self, "_context").__getattribute__(name)
