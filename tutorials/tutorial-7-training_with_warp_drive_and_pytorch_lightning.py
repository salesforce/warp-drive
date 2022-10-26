# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fast Multi-agent Reinforcement Learning on a GPU using WarpDrive and Pytorch Lightning

# %% [markdown]
# Try this notebook on [Colab](http://colab.research.google.com/github/salesforce/warp-drive/blob/master/tutorials/tutorial-7-training_with_warp_drive_and_pytorch_lightning.ipynb)

# %% [markdown]
# # ⚠️ PLEASE NOTE:
# This notebook runs on a GPU runtime.\
# If running on Colab, choose Runtime > Change runtime type from the menu, then select `GPU` in the 'Hardware accelerator' dropdown menu.

# %%
import torch

assert torch.cuda.device_count() > 0, "This notebook needs a GPU to run!"

# %% [markdown]
# # Introduction

# %% [markdown]
# This tutorial shows how [WarpDrive](https://github.com/salesforce/warp-drive) can be used together with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
#
# WarpDrive is a flexible, lightweight, and easy-to-use RL framework that implements end-to-end deep multi-agent RL on a single GPU (Graphics Processing Unit). Using the extreme parallelization capability of GPUs, it enables [orders-of-magnitude faster RL](https://arxiv.org/abs/2108.13976) compared to common implementations that blend CPU simulations and GPU models. WarpDrive is extremely efficient as it runs simulations across multiple agents and multiple environment replicas in parallel and completely eliminates the back-and-forth data copying between the CPU and the GPU.
#
# Pytorch Lightning is a machine learning framework which [greatly reduces trainer boilerplate code](https://www.pytorchlightning.ai/), and improves training modularity and flexibility. It abstracts away most of the engineering pieces of code, so users can focus on research and building models, and iterate on experiments really fast. Pytorch Lightning also provides support for easily running the model on any hardware, performing distributed training, model checkpointing, performance profiling, logging and visualization.
#
# Below, we demonstrate how to use WarpDrive and PytorchLightning together to train a game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) where multiple *tagger* agents are trying to run after and tag multiple other *runner* agents. As such, the Warpdrive framework comprises several utility functions that help easily implement any (OpenAI-)*gym-style* RL environment, and furthermore, provides quality-of-life tools to train it end-to-end using just a few lines of code. You may familiarize yourself with WarpDrive with the help of these [tutorials](https://github.com/salesforce/warp-drive/tree/master/tutorials).
#
# We invite everyone to **contribute to WarpDrive**, including adding new multi-agent environments, proposing new features and reporting issues on our open source [repository](https://github.com/salesforce/warp-drive).

# %% [markdown]
# ### Dependencies

# %% [markdown]
# This notebook requires the `rl-warp-drive` as well as the `pytorch-lightning` packages.

# %%
# ! pip install -U rl_warp_drive

# ! pip install 'pytorch_lightning>=1.4'

# Also ,install ffmpeg for visualizing animations
# ! apt install ffmpeg --yes

# %%
import logging

from IPython.display import HTML
from pytorch_lightning import Trainer

from example_envs.tag_continuous.generate_rollout_animation import (
    generate_tag_env_rollout_animation,
)
from example_envs.tag_continuous.tag_continuous import TagContinuous
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.pytorch_lightning import (
    CUDACallback,
    PerfStatsCallback,
    WarpDriveModule,
)

# %%
# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

# %% [markdown]
# # Specify a set of run configurations for your experiments

# %% [markdown]
# The run configuration is a dictionary comprising the environment parameters, the trainer and the policy network settings, as well as configurations for saving.
#
# For our experiment, we consider an environment wherein $5$ taggers and $100$ runners play the game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) on a $20 \times 20$ plane. The game lasts $200$ timesteps. Each agent chooses it's own acceleration and turn actions at every timestep, and we use mechanics to determine how the agents move over the grid. When a tagger gets close to a runner, the runner is tagged, and is eliminated from the game. For the configuration below, the runners and taggers have the same unit skill levels, or top speeds.
#
# We train the agents using $50$ environments or simulations running in parallel. With WarpDrive, each simulation runs on sepate GPU blocks.
#
# There are two separate policy networks used for the tagger and runner agents. Each network is a fully-connected model with two layers each of $256$ dimensions. We use the Advantage Actor Critic (A2C) algorithm for training. WarpDrive also currently provides the option to use the Proximal Policy Optimization (PPO) algorithm instead.

# %%
run_config = dict(
    name="tag_continuous",
    # Environment settings.
    env=dict(
        num_taggers=5,  # number of taggers in the environment
        num_runners=100,  # number of runners in the environment
        grid_length=20.0,  # length of the (square) grid on which the game is played
        episode_length=200,  # episode length in timesteps
        max_acceleration=0.1,  # maximum acceleration
        min_acceleration=-0.1,  # minimum acceleration
        max_turn=2.35,  # 3*pi/4 radians
        min_turn=-2.35,  # -3*pi/4 radians
        num_acceleration_levels=10,  # number of discretized accelerate actions
        num_turn_levels=10,  # number of discretized turn actions
        skill_level_tagger=1.0,  # skill level for the tagger
        skill_level_runner=1.0,  # skill level for the runner
        use_full_observation=False,  # each agent only sees full or partial information
        runner_exits_game_after_tagged=True,  # flag to indicate if a runner stays in the game after getting tagged
        num_other_agents_observed=10,  # number of other agents each agent can see
        tag_reward_for_tagger=10.0,  # positive reward for the tagger upon tagging a runner
        tag_penalty_for_runner=-10.0,  # negative reward for the runner upon getting tagged
        end_of_game_reward_for_runner=1.0,  # reward at the end of the game for a runner that isn't tagged
        tagging_distance=0.02,  # margin between a tagger and runner to consider the runner as 'tagged'.
    ),
    # Trainer settings.
    trainer=dict(
        num_envs=50,  # number of environment replicas (number of GPU blocks used)
        train_batch_size=10000,  # total batch size used for training per iteration (across all the environments)
        num_episodes=500,  # total number of episodes to run the training for (can be arbitrarily high!)
    ),
    # Policy network settings.
    policy=dict(
        runner=dict(
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm="A2C",  # algorithm used to train the policy
            gamma=0.98,  # discount rate
            lr=0.005,  # learning rate
            model=dict(
                type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""
            ),  # policy model settings
        ),
        tagger=dict(
            to_train=True,
            algorithm="A2C",
            gamma=0.98,
            lr=0.002,
            model=dict(
                type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""
            ),
        ),
    ),
    # Checkpoint saving setting.
    saving=dict(
        metrics_log_freq=10,  # how often (in iterations) to print the metrics
        model_params_save_freq=5000,  # how often (in iterations) to save the model parameters
        basedir="/tmp",  # base folder used for saving
        name="continuous_tag",  # experiment name
        tag="example",  # experiment tag
    ),
)

# %% [markdown]
# # Instantiate the WarpDrive Module

# %% [markdown]
# In order to instantiate the WarpDrive module,
# we first use an environment wrapper to specify that the environment needs to
# be run on the GPU (via the `env_backend` flag).
# Also, agents in the environment can share policy models;
# so we specify a dictionary to map each policy network model to the list of agent ids using that model.

# %%
# Create a wrapped environment object via the EnvWrapper
# Ensure that env_backend is set to be "pycuda" (in order to run on the GPU)
# WarpDrive v2 also supports JIT numba backend,
# if you have installed Numba, you can set "numba" instead of "pycuda" too.
env_wrapper = EnvWrapper(
    TagContinuous(**run_config["env"]),
    num_envs=run_config["trainer"]["num_envs"],
    env_backend="pycuda",
)

# Agents can share policy models: this dictionary maps policy model names to agent ids.
policy_tag_to_agent_id_map = {
    "tagger": list(env_wrapper.env.taggers),
    "runner": list(env_wrapper.env.runners),
}

wd_module = WarpDriveModule(
    env_wrapper=env_wrapper,
    config=run_config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
    verbose=True,
)

# %% [markdown]
# # Visualizing an episode roll-out before training

# %% [markdown]
# We will use the `generate_tag_env_rollout_animation()` helper function in order to visualize an episode rollout. Internally, this function uses the WarpDrive module's `fetch_episode_states` to fetch the data arrays on the GPU for the duration of an entire episode. Specifically, we fetch the state arrays pertaining to agents' x and y locations on the plane and indicators on which agents are still active in the game, and will use these to visualize an episode roll-out. Note that this function may be invoked at any time during training, and it will use the state of the policy models at that time to sample actions and generate the visualization.
#
# The animation below shows a sample realization of the game episode before training, i.e., with randomly chosen agent actions. The $5$ taggers are marked in pink, while the $100$ blue agents are the runners. Both the taggers and runners move around randomly and about half the runners remain at the end of the episode.

# %%
anim = generate_tag_env_rollout_animation(wd_module, fps=25)
HTML(anim.to_html5_video())

# %% [markdown]
# # Create the Lightning Trainer

# %% [markdown]
# Next, we create the trainer for training the WarpDrive model. We add the `performance stats` callbacks to the trainer to view the throughput performance of WarpDrive.

# %%
log_freq = run_config["saving"]["metrics_log_freq"]

# Define callbacks.
cuda_callback = CUDACallback(module=wd_module)
perf_stats_callback = PerfStatsCallback(
    batch_size=wd_module.training_batch_size,
    num_iters=wd_module.num_iters,
    log_freq=log_freq,
)

# Instantiate the PytorchLightning trainer with the callbacks.
# # Also, set the number of gpus to 1, since this notebook uses just a single GPU.
num_gpus = 1
num_episodes = run_config["trainer"]["num_episodes"]
episode_length = run_config["env"]["episode_length"]
training_batch_size = run_config["trainer"]["train_batch_size"]
num_epochs = int(num_episodes * episode_length / training_batch_size)

# Set reload_dataloaders_every_n_epochs=1 to invoke
# train_dataloader() each epoch.
trainer = Trainer(
    accelerator="gpu",
    devices=num_gpus,
    callbacks=[cuda_callback, perf_stats_callback],
    max_epochs=num_epochs,
    reload_dataloaders_every_n_epochs=1,
)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# %% [markdown]
# # Train the WarpDrive Module

# %% [markdown]
# Finally, we invoke training.
#
# Note: please scroll up to the tensorboard cell to visualize the curves during training.

# %%
trainer.fit(wd_module)

# %% [markdown]
# ## Visualize an episode-rollout after training

# %%
anim = generate_tag_env_rollout_animation(wd_module, fps=25)
HTML(anim.to_html5_video())

# %% [markdown]
# Note: In the configuration above, we have set the trainer to only train on $50000$ rollout episodes, but you can increase the `num_episodes` configuration parameter to train further. As more training happens, the runners learn to escape the taggers, and the taggers learn to chase after the runner. Sometimes, the taggers also collaborate to team-tag runners. A good number of episodes to train on (for the configuration we have used) is $2$M or higher.

# %%
# Finally, close the WarpDrive module to clear up the CUDA memory heap
wd_module.graceful_close()

# %% [markdown]
# # Learn More about WarpDrive and explore our tutorials!

# %% [markdown]
# For your reference, all our tutorials are here:
# 1. [WarpDrive basics(intro and pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.a-warp_drive_basics.ipynb)
# 2. [WarpDrive basics(numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.b-warp_drive_basics.ipynb)
# 3. [WarpDrive sampler(pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.a-warp_drive_sampler.ipynb)
# 4. [WarpDrive sampler(numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.b-warp_drive_sampler.ipynb)
# 5. [WarpDrive resetter and logger](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)
# 6. [Create custom environments (pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)
# 7. [Create custom environments (numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.b-create_custom_environments_numba.md)
# 8. [Training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb)
# 9. [Scaling Up training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-6-scaling_up_training_with_warp_drive.md)
# 10. [Training with WarpDrive + Pytorch Lightning](https://github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-7-training_with_warp_drive_and_pytorch_lightning.ipynb)

# %%
