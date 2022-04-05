"""
Automatic Vertical Scaling
Perform a binary search to figure out the max. values of the training parameters:
'num_envs' and 'train_batch_size' to use on a GPU.
The highest 'num_envs' is chosen that maximizes the GPU utilization
(i.e., uses up as many GPU blocks possible). The highest 'train_batch_size'
is then chosen in order to maximize the GPU memory usage.
These two parameters essentially determine the largest data batch size that
can be used towards training on a GPU.
Note: As the num_envs is increased further and further, the GPU eventually
runs out of blocks and the function run will throw a
'cuMemFree failed: an illegal memory access was encountered` error.
As the batch size is increased further and further (for a chosen num_envs),
the GPU runs out of memory, and the function run will throw a
`CUDA out of memory` error.
"""

import logging

from warp_drive.training.utils.child_process import ProcessWrapper


def best_param_search(low=1, margin=1, func=None):
    """
    Perform a binary search to determine the best parameter value.
    In this specific context, the best
    parameter is (the highest) value of the parameter (e.g. batch size)
    that can be used to run a func(tion)
    (e.g., training) successfully. Beyond a certain value,
    the function fails to run for reasons such as out-of-memory.
    param low: a starting low value to start searching from (defaults to 1).
    param margin: denotes the margin allowed when choosing the
        configuration parameter (and the optimal parameter).
    param func: the function that is required to be run with the
        configuration parameter.
    """
    assert low > 0
    assert margin > 0
    assert func is not None

    # Determine if the function succeeds to run at the starting (low) value.
    # If not, keep lowering the value of low until the run succeeds.
    try:
        print(f"Trying with a parameter value of {low}.")
        func(low)
        success = True
    except Exception as err:
        logging.error(err)
        success = False
        print("Run failed! The starting value of the parameter is itself too high!\n")

    while not success and low > 0:
        try:
            low = low // 2
            print(f"Trying with a parameter value of {low}.")
            func(low)
            success = True
        except Exception as err:
            logging.error(err)
            print("Run failed! Lowering the parameter value.\n")

    if not success:
        print("The function failed to run even at the lowest parameter value !")
        return None

    # Set coarse limits on low (function succeeds to run) and
    # high (function does not succeed running).
    while success:
        high = 2 * low
        try:
            print(f"Trying with a parameter value of {high}.")
            func(high)
            low = high
        except Exception as err:
            logging.error(err)
            success = False
            print("Run failed!\n")
            print(
                f"Low and high parameter values set to {low} and {high} respectively."
            )

    # Binary search to find the optimal value of low (within the margin).
    current_margin = high - low
    while current_margin > margin:
        mid = (low + high) // 2
        try:
            print(f"Trying with a parameter value of {mid}.")
            func(mid)
            low = mid
        except Exception as err:
            logging.error(err)
            high = mid
            print("Run failed!\n")
        print(f"Low and high parameter values set to {low} and {high} respectively.")
        current_margin = high - low

    print(f"Setting the parameter value to {low}\n")

    return low


def perform_auto_vertical_scaling(setup_trainer_and_train, config, num_iters=2):
    """
    Auto-scale the number of envs and batch size to maximize GPU utilization.
    param num_iters: number of iterations to use when performing automatic
    vertical scaling.
    """

    def launch_process(func, kwargs):
        """
        Run a Python function on a separate process.
        """
        p = ProcessWrapper(target=func, kwargs=kwargs)
        p.start()
        p.join()
        if p.exception:
            raise p.exception

    def set_num_envs_and_train(num_envs, run_config=config):
        run_config["trainer"]["num_envs"] = num_envs
        # Note that we also set the train batch size equal to
        # the number of environments, so that each block only
        # captures one timestep of the simulation.
        run_config["trainer"]["train_batch_size"] = num_envs
        # Set the appropriate number of episodes in order only
        # run for just `num_iters` iterations (i.e., train_batch_size = num_envs).
        run_config["trainer"]["num_episodes"] = (
            num_iters
            * run_config["trainer"]["train_batch_size"]
            / run_config["env"]["episode_length"]
        )
        # Performing training on a separate process
        launch_process(
            setup_trainer_and_train,
            kwargs={"run_configuration": config, "verbose": False},
        )

    def set_batch_size_per_env_and_train(train_batch_size_per_env, run_config=config):
        run_config["trainer"]["train_batch_size"] = (
            train_batch_size_per_env * config["trainer"]["num_envs"]
        )
        # Set the appropriate number of episodes in order only
        # run for just `num_iters` iterations (i.e., train_batch_size = num_envs).
        run_config["trainer"]["num_episodes"] = (
            num_iters
            * run_config["trainer"]["train_batch_size"]
            / run_config["env"]["episode_length"]
        )
        # Performing training on a separate process
        launch_process(
            setup_trainer_and_train,
            kwargs={"run_configuration": config, "verbose": False},
        )

    # Save some initial configs
    num_episodes = config["trainer"]["num_episodes"]
    use_wandb = config["saving"].get("use_wandb", False)
    # disable wandb
    config["saving"]["use_wandb"] = False

    # First, determine the maximum number of environments (i.e., GPU blocks)
    # that can be run in parallel before running out of thread memory.
    print("=" * 80)
    print("Determining the maximum number of environment replicas to run in parallel.")
    print("=" * 80)
    num_envs = config["trainer"]["num_envs"]
    max_envs = best_param_search(low=num_envs, func=set_num_envs_and_train)
    # Set the `num_envs` parameter to the max value found from above.
    config["trainer"]["num_envs"] = max_envs

    # Next, determine the maximum batch size that can be used
    # without running out of memory.
    print("=" * 80)
    print("Determining the maximum training batch size.")
    print("=" * 80)
    max_batch_size_per_env = best_param_search(func=set_batch_size_per_env_and_train)
    config["trainer"]["train_batch_size"] = (
        max_batch_size_per_env * config["trainer"]["num_envs"]
    )

    # Put back the original number of episodes and use_wandb settings.
    config["trainer"]["num_episodes"] = num_episodes
    config["saving"]["use_wandb"] = use_wandb

    return config
