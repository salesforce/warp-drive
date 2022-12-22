import logging
import multiprocessing as mp

import torch

from warp_drive.training.utils.device_child_process import process_group_torch

mp.set_start_method("spawn", force=True)


class ProcessWrapper(mp.Process):
    """
    A process wrapper to catch exceptions when they occur.
    """

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._child_conn.send(None)
        except Exception as err:
            logging.error(err)
            self._child_conn.send(err)

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception


class DeviceContextProcessWrapper(ProcessWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_context = None
        self.device_id = kwargs["kwargs"].get("device_id", None)
        self.num_devices = kwargs["kwargs"].get("num_devices", 1)

    def _init_context(self):
        raise NotImplementedError

    def _clear_context(self):
        raise NotImplementedError

    def _init_torch_process_group(self):
        process_group_torch.setup_torch_process_group(
            device_id=self.device_id, num_devices=self.num_devices
        )

    @staticmethod
    def _clear_torch_process_group():
        process_group_torch.clear_torch_process_group()

    def assert_context_consistency(self):
        assert torch.cuda.current_device() == self.device_id, (
            f"PyCuda/Numba and Pytorch have seen different device rank: "
            f"We enforce that the entire GPU context "
            f"is consistent between PyCuda and Pytorch"
            f"PyCuda: {self.device_id}, Pytorch: {torch.cuda.current_device()}"
        )

    def run(self):
        print(f"Starting worker process: {self.device_id} ")

        try:
            # construct the Current GPU context
            self._init_context()
            self._init_torch_process_group()
            self.assert_context_consistency()
            mp.Process.run(self)
            self._child_conn.send(None)
        except Exception as err:
            logging.error(err)
            self._child_conn.send(err)
        finally:
            # clean up the context regardless the finish status
            self._clear_torch_process_group()
            self._clear_context()


event_messenger = mp.Event()
