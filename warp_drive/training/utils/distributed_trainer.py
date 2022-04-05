import time

from warp_drive.training.utils.child_process import (
    DeviceContextProcessWrapper,
    event_messenger,
)


def perform_distributed_training(setup_trainer_and_train, config, results_dir=None):
    # Perform distributed training. Create a new process for each trainer.
    assert config["trainer"]["num_gpus"] > 1
    num_devices = config["trainer"]["num_gpus"]

    e = event_messenger

    procs = []

    if results_dir is None:
        # Use the current time as the name for the results directory.
        results_dir = f"{time.time():10.0f}"

    for device_id in range(num_devices):
        proc = DeviceContextProcessWrapper(
            target=setup_trainer_and_train,
            kwargs={
                "run_configuration": config,
                "device_id": device_id,
                "num_devices": num_devices,
                "event_messenger": e,
                "results_directory": results_dir,
                "verbose": (device_id == 0),
            },
        )
        procs.append(proc)

    for p in procs:
        p.start()

    for p in procs:
        p.join()
        if p.exception:
            print(p.exception)

    print("Exiting the Parent Process.")
