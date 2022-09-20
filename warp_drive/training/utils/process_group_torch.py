import os

import torch.distributed as dist


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
