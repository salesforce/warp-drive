import os

import pycuda.driver as cuda_driver
import torch.distributed as dist
from pycuda.tools import clear_context_caches, make_default_context


class SingleDeviceContext:
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


def setup_torch_process_group(
    device_id, num_devices, master_addr="127.0.0.1", master_port="8888", backend="gloo"
):
    """Setup code comes directly from the docs:

    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(backend, rank=device_id, world_size=num_devices)


def clear_torch_process_group():
    dist.destroy_process_group()
