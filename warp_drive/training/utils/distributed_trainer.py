from warp_drive.training.utils.child_process import (
    DeviceContextProcessWrapper,
    event_messenger,
)


def perform_distributed_training(setup_trainer_and_train, config):
    # Perform distributed training. Create a new process for each trainer.
    assert config["trainer"]["num_gpus"] > 1
    num_devices = config["trainer"]["num_gpus"]

    e = event_messenger

    procs = []

    for device_id in range(num_devices):
        proc = DeviceContextProcessWrapper(
            target=setup_trainer_and_train,
            kwargs={
                "run_configuration": config,
                "device_id": device_id,
                "num_devices": num_devices,
                "event_messenger": e,
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
