Copyright (c) 2021, salesforce.com, inc. \
All rights reserved. \
SPDX-License-Identifier: BSD-3-Clause. \
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause.

# Introduction

In this tutorial, we will describe how to implement your own environment in CUDA C, and integrate it with WarpDrive for simulating the environment dynamics on the GPU.

In case you haven't familiarized yourself with WarpDrive and its PyCUDA backend, please see the other tutorials:

- [WarpDrive basics for PyCUDA](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.a-warp_drive_basics.ipynb)
- [WarpDrive sampler for PyCUDA](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.a-warp_drive_sampler.ipynb)
- [WarpDrive reset and log](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)

We follow the OpenAI [gym](https://gym.openai.com/) style. Each simulation should have `__init__`, `reset` and `step` methods. 

To use WarpDrive, you only need to implement the `step()` method in CUDA C. WarpDrive can automatically reinitialize the environment after it's done, i.e., at every `reset`, using the environment `Wrapper` class. This class takes your CUDA C `step()` function and manages the simulation flow on the GPU. 

You can then do RL! See the [next tutorial](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb) to learn how to perform end-to-end multi-agent RL on a single GPU with WarpDrive.

# Building Simulations in CUDA C

CUDA C is an extension of C. See [this Nvidia blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) and the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) for more info and CUDA tutorials.

For our initial release of WarpDrive, we focus on relatively simple simulations. A key reason is that CUDA C can give you significantly faster simulations, but requires careful memory management, among other things. 

To make sure that everything works properly, one approach is to first implement your simulation logic in Python. You can then implement the same logic in CUDA C and check the simulation behaviors are the same. 

To help with this process, we provide an *environment consistency checker* method to do consistency tests between Python and CUDA C simulations. 

This workflow helps to familiarize yourself with CUDA C and works well for relatively simple simulations.

# Case Study: Building a CUDA Version of Tag

Within the WarpDrive package, you can find the source code for the discrete and continuous versions of Tag.

- [Tag (GridWorld)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py)
- [Tag (Continuous)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py)

Tag is a simple multi-agent game involving 'taggers' and 'runners'. The taggers chase and try to tag the runners. Tagged runners leave the game. Runners try to get away from the taggers.

Next, we'll use the *continuous* version of Tag to explain some important elements of building CUDA C simulations.

# Managing CUDA Simulations from Python using WarpDrive

We begin with the Python version of the continuous version [Tag](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py). The simulation follows the [gym](https://gym.openai.com/) format, implementing `reset` and `step` methods. We now detail all the steps necessary to transform the `step` function into [CUDA code](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous_step_pycuda.cu) that can be run on a GPU. Importantly, WarpDrive lets you call these CUDA methods from Python, so you can design your own RL workflow entirely in Python.

## 1. Add data to be pushed to GPU via the *DataFeed* class

First, we need to push all the data relevant to performing the reset() and step() functions on the GPU. In particular, there are two methods that need to be added to the environment 
```python
    def get_data_dictionary(self):
        data_dict = DataFeed()
        ...
        return data_dict
```
and 
```python
    def get_tensor_dictionary(self):
        data_dict = DataFeed()
        ...
        return data_dict
```
WarpDrive automatically handles pushing the data arrays provided within these methods to the GPU global memory. The data dictionary will be used to push data that will not require to be modified during training - once pushed into the GPU, this data will persist on the GPU, and not be modified. The tensor dictionary comprises data that is directly accessible by PyTorch, and is handy for data that needs to be modified during training. In each of the aforementioned data_dictionary methods, the return type needs to be a `DataFeed` class, which is essentially a dictionary, with additional attributes.

With the help of the DataFeed class, we can push arrays that are created when the environment is initialized, and needs to be re-initialized at every reset.

```python
data_dict = DataFeed()
for feature in ["loc_x", "loc_y", "speed", "direction", "acceleration"]:
    data_dict.add_data(
        name=feature,
        data=self.global_state[feature][0],
        save_copy_and_apply_at_reset=True,
    )
```

Importantly, notice the `save_copy_and_apply_at_reset` flag set to True. This instructs WarpDrive to make a copy of this data and automatically re-initialize the data array to that exact value at each reset.

We can also push environment configuration parameters, for example,

```python
data_dict.add_data(
    name="tag_reward_for_tagger", data=self.tag_reward_for_tagger
)
data_dict.add_data(
    name="distance_margin_for_reward", data=self.distance_margin_for_reward
)
```

and any auxiliary variables that will be useful for modeling the step function dynamics:
```python
data_dict.add_data(
    name="neighbor_distances",
    data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
    save_copy_and_apply_at_reset=True,
)
```

**Note**: for convenience, the data feed object also supports pushing multiple arrays at once via the `add_data_list()` API, see the [Tag (GridWorld)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py#L337) code for an example. 

An important point to note is that CUDA C always uses **32-bit precision**, so it's good to cast all the numpy arrays used in the Python simulation to 32-bit precision as well, before you push them.

## 2. Invoke the CUDA version of *step* in Python

After all the relevant data is added to the data dictionary, we need to invoke the CUDA C kernel code for stepping through the environment (when `self.use_cuda` is `True`). The syntax to do this is as follows

```python
if self.env_backend == "pycuda":
    self.cuda_step(
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
block=self.cuda_function_manager.block,
grid=self.cuda_function_manager.grid,
```
Note that `n_agents` and `episode_length` are part of the meta information for the data manager, so they can be directly referenced from therein. In particular, the `block` and `grid` arguments are essential to have the CUDA implementation determine how many threads and blocks to activate and use for the environment simulation.

WarpDrive also supports feeding multiple arguments at once via the `CUDAFunctionFeed` class, see the [Tag (GridWorld)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py#L355) code for an example.

## 3. Write the *step()* method in CUDA C

The most laborious part of this exercise is actually writing out the step function in CUDA C. Importantly ,this function will need to be named `Cuda<env.name>Step`, so that WarpDrive knows it represents the CUDA version of the step function for the particular environment. The order of the arguments should naturally follow the order written out where the CUDA C kernel is invoked.

```C
__global__ void CudaTagContinuousStep(
        float* loc_x_arr,
        float* loc_y_arr,
        float* speed_arr,
        ...
```

Note the keyword `__global__` used on the increment function. Global functions are also called "kernels" - they are functions you may call from the host. In our implementation of the CUDA C [code](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous_step.cu) for the tag environment, you will also notice there's also the keyword `__device__` (for example, `__device__ void CudaTagContinuousGenerateObservation()` and  `__device__ void CudaTagContinuousComputeReward()`) for functions that cannot be called from the host, but may only be called from other device or global functions.

Also, note the `void` return type - CUDA step functions don't need to return anything, but all the data arrays are modified in place.

While writing out the step code in CUDA C, the environment logic follows the same logic as in the Python step code. Remember that each thread only acts on a single agent, and for a single environment. The code excerpt below is a side-by-side comparison of Python and CUDA C code for updating the agents' x and y location co-ordinates.

On the CUDA C side, we can simplify and make the code mode readable by using constants such as `kThisAgentId` and `kEnvId` (we have used this [naming style guide](https://google.github.io/styleguide/cppguide.html#General_Naming_Rules)) to indicate the thread and block indices, respectively. As you may have noticed by now, since each thread only writes to a specific index of the data array, understanding array indexing is critical.

<table align="left">
<tr>
<th> Python </th>
<th> CUDA C </th>
</tr>
<td>

```python
loc_x_curr_t = loc_x_prev_t + speed_curr_t * np.cos(dir_curr_t)
loc_y_curr_t = loc_y_prev_t + speed_curr_t * np.sin(dir_curr_t)
```

</td>
    
<td>
    
```c
const int kThisAgentId = threadIdx.x;
const int kEnvId = blockIdx.x;
if (kThisAgentId < kNumAgents) {
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;

    loc_x_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
    loc_y_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);
}
```
                              
</td>
    
</table>

## 4. Put together as an Environment class

To use an existing Python Environment with
WarpDrive, one needs to add two augmentations. First,
a get data dictionary() method that returns a dictionary-like
DataFeed object with data arrays and parameters that should
be pushed to the GPU. Second, the step-function should call
the cuda step with the data arrays that the CUDA C step
function should have access to.

In general, we can use just a single (*dual-mode*) environment class that can run both the Python and the CUDA C modes of the environment code on a GPU. The `env_backend = "pycuda"` enables switching between those modes. Note that the environment class will need to subclass `CUDAEnvironmentContext`, which essentially adds attributes to the environment (such as the `cuda_data_manager` and `cuda_function_manager`) that are required for running on a GPU. This also means that the environment itself can be stepped through only on a GPU. Please refer to the [Tag (Continuous)](https://www.github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) for a detailed working example.

```python
"""
Dual mode Environment class ().
"""
class MyDualModeEnvironment(CUDAEnvironmentContext):
    
    ...
    
    def get_data_dictionary(self):
        data_dict = DataFeed()
        ...
        return data_dict 
    
    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        ...
        return tensor_dict
    
    def reset(self):
        # reset for CPU environment
        ...
    
    def step(self, actions=None):
        args = [YOUR_CUDA_STEP_ARGUMENTS]
        
        if self.use_cuda:
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
            return None
        else:
            ...
            return obs, rew, done, info
```

Alternatively, if you wish to run the Python environment in a CPU-only hardware (where WarpDrive cannot be installed), we suggest you treat the environment class as a base class (e.g. [Tag (GridWorld)](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py#L23)) and move the code augmentations needed for WarpDrive to a derived class (e.g. [Tag (GridWorld)](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld.py#L318)) Accordingly, you can run the Python environment without having to install WarpDrive, while the CUDA C mode of the environment will need to run only on a GPU (with WarpDrive installed).

```python
"""
The CUDA environment class is derived from the CPU environment class.
"""
class MyCPUEnvironment:
    
    ...
    
    def reset(self):
        # reset for CPU environment
        ... 
        
    def step(self):
        # step for CPU environment
        ...
        return obs, rew, done, info
    
    
class MyCUDAEnvironment(MyCPUEnvironment, CUDAEnvironmentContext):
    
    ...
    
    def get_data_dictionary(self):
        data_dict = DataFeed()
        ...
        return data_dict 
    
    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        ...
        return tensor_dict
    
    def step(self):
        # overwrite the CPU execution in the CPU base class
        self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
        return None
```

## 5. The EnvWrapper Class

Once the CUDA C environment implementation is complete, WarpDrive provides an environment wrapper class to help launch the simulation on the CPU or the GPU. This wrapper determines whether the simulation needs to be launched on the CPU or the GPU (via the `use_cuda` flag), and proceeds accordingly. If the environment runs on the CPU, the `reset` and `step` calls also occur on the CPU. If the environment runs on the GPU, only the first `reset` happens on the CPU, all the relevant data is copied over the GPU after, and the subsequent steps (and resets) all happen on the GPU. In the latter case, the environment wrapper also uses the `num_envs` argument to instantiate multiple replicas of the environment on the GPU.

Additionally, the environment wrapper handles all the tasks required to run the environment on the GPU:

- Determines the environment's observation and action spaces
- Initializes the CUDA data and function managers for the environment
- Registers the CUDA version of the step() function
- Pushes the data listed in the data dictionary and tensor dictionary attributes of the environment, and repeats them across the environment dimension, if necessary.
- Automatically resets each environment when it is done.

### Register the CUDA environment

Here we have some more details about how to use EnvWrapper to identify and build your environment automatically once the CUDA C step environment is ready.

#### 1. Default Environment

You shall register your default environment in `warp_drive/utils/pycuda_utils/misc` and the function `get_default_env_directory()`. There, you can simply provide the path to your CUDA environment source code. Please remember that the register uses the environment name defined in your environment class as the key so EnvWrapper class can link it to the right environment. 

The **FULL_PATH_TO_YOUR_ENV_SRC** can be any path inside or outside WarpDrive. For example, you can develop your own CUDA step function and environment in your codebase and register right here.

```python
   envs = {
       "TagGridWorld": f"{get_project_root()}/example_envs/tag_gridworld/tag_gridworld_step.cu",
       "TagContinuous": f"{get_project_root()}/example_envs/tag_continuous/tag_continuous_step_pycuda.cu",
       "YOUR_ENVIRONMENT": "FULL_PATH_TO_YOUR_ENV_CUDA_SRC",
   }
```
Usually we do not suggest you use this "hard" way because it integrates your environment directly into WarpDrive. So far, we have our Tag games as benchmarks registered right there as we regard them as part of WarpDrive codebase.

#### 2. Customized Environment

You can register a customized environment by using **EnvironmentRegistrar**. Please note that the customized environment has the higher priority than the default environments, i.e., if two environments (one is registered as customized, the other is the default) take the same name, the customized environment will be loaded. However, it is recommended to not have any environment name conflict in any circumstance.

```python
from warp_drive.utils.env_registrar import EnvironmentRegistrar
import Your_Env_Class

env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(Your_Env_Class.name, "FULL_PATH_TO_YOUR_ENV_CUDA_SRC", env_backend="pycuda")
env_wrapper = EnvWrapper(
    Your_Env_Class(**run_config["env"]), 
    num_envs=num_envs, 
    env_backend="numba", 
    env_registrar=env_registrar)
```

Now, inside the EnvWrapper, function managers will be able to feed the `self.num_env` and `self.num_agents` to the CUDA compiler at compile time to build and load a unique CUDA environment context for all the tasks.

## 6. Environment Consistency Checker

Given the environment is implemented in both Python and CUDA C (for running on the CPU and GPU, respectively), please use **EnvironmentCPUvsGPU** class to test the consistency of your implementation. The module will instantiate two separate environment objects (with the `use_cuda` flag set to True and False), step through `num_episodes` episodes (with the same actions) and determine if there are inconsistencies in terms of the generated states, rewards or done flags. 

Here is an example for the dual mode environment class. Please refer the to [Tag (Continuous) test](https://www.github.com/salesforce/warp-drive/blob/master/tests/example_envs/test_tag_continuous.py) for a detailed working example.

```python
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
import Your_Dual_Mode_Env_Class


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(Your_Dual_Mode_Env_Class.name, "FULL_PATH_TO_YOUR_ENV_CUDA_SRC")
env_configs = {
    "test1": {
        "num_agents": 4,
    }
}

testing_class = EnvironmentCPUvsGPU(
    dual_mode_env_class=Your_Dual_Mode_Env_Class,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step()
```

And the following is an example for the parent-child environment classes. It is actually the same as the dual mode, the only difference is that `EnvironmentCPUvsGPU` will take two corresponding classes. Please refer to the [Tag (GridWorld) test](https://www.github.com/salesforce/warp-drive/blob/master/tests/example_envs/test_tag_gridworld.py) for a detailed working example.

```python
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
import Your_CPU_Env_Class, Your_GPU_Env_Class


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(Your_GPU_Env_Class.name, "FULL_PATH_TO_YOUR_ENV_CUDA_SRC")
env_configs = {
    "test1": {
        "num_agents": 4,
    }
}

testing_class = EnvironmentCPUvsGPU(
    cpu_env_class=Your_CPU_Env_Class,
    cuda_env_class=Your_GPU_Env_Class,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step()
```

The `EnvironmentCPUvsGPU` class also takes in a few optional arguments that will need to be correctly set, if required.
- `use_gpu_testing_mode`: a flag to determine whether to simply load the cuda binaries (.cubin) or compile the cuda source code (.cu) each time to create a binary.` Defaults to False.
- `env_registrar`: the EnvironmentRegistrar object. It provides the customized env info (like src path) for the build.
- `env_wrapper`: allows for the user to provide their own EnvWrapper.
- `policy_tag_to_agent_id_map`: a dictionary mapping policy tag to agent ids.
- `create_separate_placeholders_for_each_policy`: a flag indicating whether there exist separate observations, actions and rewards placeholders, for each policy, as designed in the step function. The placeholders will be used in the step() function and during training. When there's only a single policy, this flag will be False. It can also be True when there are multiple policies, yet all the agents have the same obs and action space shapes, so we can share the same placeholder. Defaults to False.
- `obs_dim_corresponding_to_num_agents`: this is indicative of which dimension in the observation corresponds to the number of agents, as designed in the step function. It may be "first" or "last". In other words, observations may be shaped (num_agents, *feature_dim) or (*feature_dim, num_agents). This is required in order for WarpDrive to process the observations correctly. This is only relevant when a single obs key corresponds to multiple agents. Defaults to "first".


## 7. Unittest WarpDrive

The build and test can be done automatically by directly go to the CUDA source code folder and make 
`cd warp_drive/cuda_includes; make compile-test`

Or, you can run `python warp_drive/utils/run_unittests_pycuda.py`

# Important CUDA C Concepts

Writing CUDA programs requires basic knowledge of C and how CUDA C extends C. Here's a [quick reference](https://learnxinyminutes.com/docs/c/) to see the syntax of C. 

For many simulations, basic C concepts should get you very far. However, you could make very complex simulations -- the sky is the limit! 

Below, we'll discuss two important CUDA C concepts -- we're constantly planning to add more materials and tools in the future to facilitate developing CUDA simulations.

## Array Indexing

As described in the first [tutorial](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1-warp_drive_basics.ipynb#Array-indexing), CUDA stores arrays in a C-contiguous or a row-major fashion; 

In general, it helps to set up some indexing constants as you develop code, so you can reuse them across your code. For example, the index for a specific agent id `kThisAgentId` ($0 \leq \text{kThisAgentId} < \text{NumAgents}$) in the location arrays (shaped (`NumEnvs, NumAgents`)) would be
```C
const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
```
and this index can be reused across different contexts.

Note: to facilitate simulation development, we also created a `get_flattened_array_index` helper function to provide the flattened array index; please see [Tag (GridWorld)](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_gridworld/tag_gridworld_step.cu#L151) for a working example. 

## __syncthreads

Another keyword that is useful to understand in the context of multi-agent simulations is `__syncthreads()`. While all the agents can operate fully in parallel, there are often operations that may need to be performed sequentially by the agents or only by one of the agents. For such cases, we may use **__syncthreads()** command, a thread block-level synchronization barrier. All the threads will wait for all the threads in the block to reach that point, until processing further.

```C
        // Increment time ONCE -- only 1 thread can do this.
        if (kThisAgentId == 0) {
            env_timestep_arr[kEnvId] += 1;
        }

        // Wait here until timestep has been updated
        __syncthreads();
```

# Debugging and Checking Consistency

Once you are done building your environment, you may use the `env_cpu_gpu_consistency_checker` function in WarpDrive to ensure the Python and CUDA C versions of the environment are logically consistent with one another. The consistency tests run across two full episode lengths (to ensure consistent behavior even beyond the point when the environments are reset), and ensure that the observations, rewards, and done flags match one another. For catching syntax errors, the C compiler is pretty good at pointing out the exact error and the line number. Often, to figure out deeper issues with the code, `printf` is your best friend.

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
