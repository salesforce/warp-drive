import os

import pycuda.driver as cuda_driver
import torch.distributed as dist
from pycuda.tools import clear_context_caches, make_default_context


class PyCUDASingleDeviceContext:
    """
    one single GPU hardware context available for both PyCuda and PyTorch
    """

    _context = None

    def init_context(self, device_id=None):
        if device_id is None:
            context = make_default_context()
            self._context = context
        else:
            context = cuda_driver.Device(device_id).make_context()
            self._context = context

    @property
    def context(self):
        return self._context

    def __getattribute__(self, name):
        if name in "init_context":
            return object.__getattribute__(self, name)

        if object.__getattribute__(self, "_context") is None:
            raise RuntimeError("Context hasn't been initialized yet")

        return object.__getattribute__(self, "_context").__getattribute__(name)


def clear_context(context):
    context.pop()
    clear_context_caches()
