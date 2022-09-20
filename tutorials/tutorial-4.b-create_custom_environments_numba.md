Copyright (c) 2021, salesforce.com, inc. \
All rights reserved. \
SPDX-License-Identifier: BSD-3-Clause. \
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause.

# Introduction

In this tutorial, we will describe how to implement your own environment in Numba, and integrate it with WarpDrive for simulating the environment dynamics on the GPU.

In case you haven't familiarized yourself with WarpDrive, please see the other tutorials:

- [WarpDrive basics for Numba](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.b-warp_drive_basics.ipynb)
- [WarpDrive sampler for Numba](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.b-warp_drive_sampler.ipynb)
- [WarpDrive reset and log](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)

We follow the OpenAI [gym](https://gym.openai.com/) style. Each simulation should have `__init__`, `reset` and `step` methods. 

To use WarpDrive, you only need to implement the `step()` method in Numba. WarpDrive can automatically reinitialize the environment after it's done, i.e., at every `reset`, using the environment `Wrapper` class. This class takes your Numba `step()` function and manages the simulation flow on the GPU. 

You can then do RL! See the [next tutorial](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb) to learn how to perform end-to-end multi-agent RL on a single GPU with WarpDrive.

# Building Simulations in Numba


To make sure that everything works properly, one approach is to first implement your simulation logic in Python. You can then implement the same logic in Numba and check the simulation behaviors are the same. 

To help with this process, we provide an *environment consistency checker* method to do consistency tests between Python and Numba simulations. 

This workflow helps to familiarize yourself with Numba and works well for relatively simple simulations.

# Case Study: Building a Numba Version of Tag

Within the WarpDrive package, you can find the source code for the discrete and continuous versions of Tag.

- [Tag (GridWorld)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py)
- [Tag (Continuous)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py)

Tag is a simple multi-agent game involving 'taggers' and 'runners'. The taggers chase and try to tag the runners. Tagged runners leave the game. Runners try to get away from the taggers.

Next, we'll use the *continuous* version of Tag to explain some important elements of building Numba simulations. In fact, WarpDrive mostly shares the same APIs for data management and kernel function management, the only difference is writing custom environments itself. We will therefore refer back to the previous tutorial of [create_custom_environments_pycuda](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md) if the content is the same.

# Managing CUDA Simulations from Python using WarpDrive

We begin with the Python version of the continuous version [Tag](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py). The simulation follows the [gym](https://gym.openai.com/) format, implementing `reset` and `step` methods. We now detail all the steps necessary to transform the `step` function into [Numba code](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous_step_numba.py) that can be run on a GPU. Importantly, WarpDrive lets you call these Numba methods from Python, so you can design your own RL workflow entirely in Python.

## 1. Add data to be pushed to GPU via the *DataFeed* class

This shares the same content of the previous tutorial [create_custom_environments_pycuda](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)

## 2. Invoke the Numba version of *step* in Python

After all the relevant data is added to the data dictionary, we need to invoke the Numba kernel code for stepping through the environment. The syntax to do this is the same as PyCUDA except that 

```python
if self.env_backend = "numba:
    block=self.cuda_function_manager.block
    grid=self.cuda_function_manager.grid
    self.cuda_step[grid, step](
                self.cuda_data_manager.device_data("loc_x"),
                self.cuda_data_manager.device_data("loc_y"),
                self.cuda_data_manager.device_data("speed"),
                ...
    ) 
```

where you need to add all the keys of the data dictionary (in no particular order) as arguments to the step function. Also, remember to add the imperative `observations`, `sampled_actions` and `rewards` data, respectively as

```python
...
self.cuda_data_manager.device_data("observations"),
self.cuda_data_manager.device_data("sampled_actions"),
self.cuda_data_manager.device_data("rewards"),
...
```

It will also be very useful to add the following reserved keywords: `_done_`, `_timestep_` along with `n_agents`, `episode_length`, `block` and `grid`.
```python
...
self.cuda_data_manager.device_data("_done_"),
self.cuda_data_manager.device_data("_timestep_"),
self.cuda_data_manager.meta_info("n_agents"),
self.cuda_data_manager.meta_info("episode_length"),
```
Note that `n_agents` and `episode_length` are part of the meta information for the data manager, so they can be directly referenced from therein. In particular, the `block` and `grid` arguments are essential to have the CUDA implementation determine how many threads and blocks to activate and use for the environment simulation.

WarpDrive also supports feeding multiple arguments at once via the `CUDAFunctionFeed` class, see the [Tag (GridWorld)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py#L355) code for an example.

## 3. Write the *step()* method in Numba

The most laborious part of this exercise is actually writing out the step function in Numba. Importantly ,this function will need to be named `Numa<env.name>Step`, so that WarpDrive knows it represents the Numba version of the step function for the particular environment. The order of the arguments should naturally follow the order written out where the Numba kernel is invoked.

```python
import numba.cuda as numba_driver

@numba_driver.jit
def NumbaTagContinuousStep(
        loc_x_arr,
        loc_y_arr,
        speed_arr,
        ...
```


<table align="left">
<tr>
<th> Python </th>
<th> Python </th>
</tr>
<td>

```python
loc_x_curr_t = loc_x_prev_t + speed_curr_t * np.cos(dir_curr_t)
loc_y_curr_t = loc_y_prev_t + speed_curr_t * np.sin(dir_curr_t)
```

</td>
    
<td>
    
```python
kThisAgentId = numba_driver.threadIdx.x
kEnvId = numba_driver.blockIdx.x
if (kThisAgentId < kNumAgents) {

    loc_x_arr[kEnvId, kthisAgentId] += speed_arr[kEnvId, kthisAgentId] * cos(direction_arr[kEnvId, kthisAgentId])
    loc_y_arr[kEnvId, kthisAgentId] += speed_arr[kEnvId, kthisAgentId] * sin(direction_arr[kEnvId, kthisAgentId])
}
```
                              
</td>
    
</table>

## 4. Put together as an Environment class
This shares the same content of the previous tutorial [create_custom_environments_pycuda](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)


## 5. The EnvWrapper Class

This shares the same content of the previous tutorial [create_custom_environments_pycuda](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)

### Register the CUDA environment

Here we have some more details about how to use EnvWrapper to identify and build your environment automatically once the CUDA C step environment is ready.

#### 1. Default Environment

You shall register your default environment in `warp_drive/utils/numba_utils/misc` and the function `get_default_env_directory()`. There, you can simply provide the path to your Numba environment source code. Please remember that the register uses the environment name defined in your environment class as the key so EnvWrapper class can link it to the right environment. 

The **FULL_PATH_TO_YOUR_ENV_SRC** can be any PythonPath inside or outside WarpDrive. For example, you can develop your own Numba step function and environment in your codebase and register right here. 

```python
   envs = {
        "TagContinuous": "example_envs.tag_continuous.tag_continuous_step_numba",
        "YOUR_ENVIRONMENT": "PYTHON_PATH_TO_YOUR_ENV_SRC",
    }
    return envs.get(env_name, None)
```
Usually we do not suggest you use this "hard" way because it integrates your environment directly into WarpDrive. So far, we have our Tag games as benchmarks registered right there as we regard them as part of WarpDrive codebase.

#### 2. Customized Environment

You can register a customized environment by using **EnvironmentRegistrar**. Please note that the customized environment has the higher priority than the default environments, i.e., if two environments (one is registered as customized, the other is the default) take the same name, the customized environment will be loaded. However, it is recommended to not have any environment name conflict in any circumstance.

```python
from warp_drive.utils.env_registrar import EnvironmentRegistrar
import Your_Env_Class

env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(Your_Env_Class.name, "PYTHON_PATH_TO_YOUR_ENV_SRC", env_backend="numba")
env_wrapper = EnvWrapper(
    Your_Env_Class(**run_config["env"]), 
    num_envs=num_envs, 
    env_backend="numba", 
    env_registrar=env_registrar)
```

Now, inside the EnvWrapper, function managers will be able to feed the `self.num_env` and `self.num_agents` to the CUDA compiler at compile time to build and load a unique CUDA environment context for all the tasks.


## 6. Environment Consistency Checker

This shares the same content of the previous tutorial [create_custom_environments_pycuda](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)

## 7. Unittest WarpDrive

You can run `python warp_drive/utils/run_unittests_numba.py`


# Learn More and Explore our Tutorials!

And that's it for this tutorial. Good luck building your environments.
Once you are done building, see our next [tutorial](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb) on training your environment with WarpDrive and the subsequent [tutorial](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-6-scaling_up_training_with_warp_drive.md) on scaling up training.

For your reference, all our tutorials are here:
1. [WarpDrive basics(intro and pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.a-warp_drive_basics.ipynb)
2. [WarpDrive basics(numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.b-warp_drive_basics.ipynb)
3. [WarpDrive sampler(pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.a-warp_drive_sampler.ipynb)
4. [WarpDrive sampler(numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.b-warp_drive_sampler.ipynb)
5. [WarpDrive resetter and logger](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)
6. [Create custom environments (pycuda)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)
7. [Create custom environments (numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.b-create_custom_environments_numba.md)
8. [Training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb)
9. [Scaling Up training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-6-scaling_up_training_with_warp_drive.md)
10. [Training with WarpDrive + Pytorch Lightning](https://github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-7-training_with_warp_drive_and_pytorch_lightning.ipynb)

```python

```
