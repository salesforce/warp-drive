import logging

from warp_drive.training.utils.device_child_process.child_process_base import DeviceContextProcessWrapper
from warp_drive.training.utils.single_device_context import device_context_pycuda


class PyCUDADeviceContextProcessWrapper(DeviceContextProcessWrapper):
    """
    A worker process wrapper that will
    (1) open up a GPU context for both PyCUDA and PyTorch
    (2) under the current GPU context, call the kernel to run on this GPU,
        and get the data via DataManager under the current GPU context
    """

    def _init_context(self):
        self.cuda_context = device_context_pycuda.PyCUDASingleDeviceContext()
        self.cuda_context.init_context(device_id=self.device_id)

    def _clear_context(self):
        if self.cuda_context is not None:
            device_context_pycuda.clear_context(context=self.cuda_context.context)
