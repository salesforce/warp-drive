# WarpDrive: Extremely Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU

WarpDrive is a flexible, lightweight, and easy-to-use open-source reinforcement learning (RL) 
framework that implements end-to-end multi-agent RL on a single or multiple GPUs (Graphics Processing Unit). 

Using the extreme parallelization capability of GPUs, WarpDrive enables orders-of-magnitude 
faster RL compared to CPU simulation + GPU model implementations. It is extremely efficient as it avoids back-and-forth data copying between the CPU and the GPU, 
and runs simulations across multiple agents and multiple environment replicas in parallel. 

We have some main updates since its initial open source,
- version 1.3: provides the auto scaling tools to achieve the optimal throughput per device.
- version 1.4: supports the distributed asynchronous training among multiple GPU devices.
- version 1.6: supports the aggregation of multiple GPU blocks for one environment replica. 
- version 2.0: supports the dual backends of both CUDA C and JIT compiled Numba. [(Our Blog article)](https://blog.salesforceairesearch.com/warpdrive-v2-numba-nvidia-gpu-simulations/)

Together, these allow the user to run thousands of concurrent multi-agent simulations and train 
on extremely large batches of experience, achieving over 100x throughput over CPU-based counterparts. 

We include several default multi-agent environments
based on the game of "Tag" for benchmarking and testing. In the "Tag" games, taggers are trying to run after
and tag the runners. They are fairly complicated games where thread synchronization, shared memory, high-dimensional indexing for thousands of interacting agents are involved. Several much more complex environments such as Covid-19 environment and climate change environment have been developed based on WarpDrive, you may see examples in [Real-World Problems and Collaborations](#real-world-problems-and-collaborations).

Below, we show multi-agent RL policies 
trained for different tagger:runner speed ratios using WarpDrive. 
These environments can **run** at **millions of steps per second**, 
and **train** in just a few **hours**, all on a single GPU!

<img src="https://blog.einstein.ai/content/images/2021/08/tagger2x-1.gif" width="250" height="250"/> <img src="https://blog.einstein.ai/content/images/2021/08/same_speed_50fps-1.gif" width="250" height="250"/> <img src="https://blog.einstein.ai/content/images/2021/08/runner2x-2.gif" width="250" height="250"/>

WarpDrive also provides tools to build and train 
multi-agent RL systems quickly with just a few lines of code.
Here is a short example to train tagger and runner agents:

```python
# Create a wrapped environment object via the EnvWrapper
# Ensure that env_backend is set to 'pycuda' or 'numba' (in order to run on the GPU)
env_wrapper = EnvWrapper(
    TagContinuous(**run_config["env"]),
    num_envs=run_config["trainer"]["num_envs"], 
    env_backend="pycuda"
)

# Agents can share policy models: this dictionary maps policy model names to agent ids.
policy_tag_to_agent_id_map = {
    "tagger": list(env_wrapper.env.taggers),
    "runner": list(env_wrapper.env.runners),
}

# Create the trainer object
trainer = Trainer(
    env_wrapper=env_wrapper,
    config=run_config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
)

# Perform training!
trainer.train()
```

Below, we compare the training speed on an N1 16-CPU
node versus a single A100 GPU (using WarpDrive), for the Tag environment with 100 runners and 5 taggers. With the same environment configuration and training parameters, WarpDrive on a GPU is about 10× faster. Both scenarios are with 60 environment replicas running in parallel. Using more environments on the CPU node is infeasible as data copying gets too expensive. With WarpDrive, it is possible to scale up the number of environment replicas at least 10-fold, for even faster training.

<img src="https://user-images.githubusercontent.com/7627238/144560725-83167c73-274e-4c5a-a6cf-4e06355895f0.png" width="400" height="400"/>


## Code Structure
WarpDrive provides a CUDA (or Numba) + Python framework and quality-of-life tools, so you can quickly build fast, flexible and massively distributed multi-agent RL systems. The following figure illustrates a bottoms-up overview of the design and components of WarpDrive. The user only needs to write a CUDA or Numba step function at the CUDA environment layer, while the rest is a pure Python interface. We have step-by-step tutorials for you to master the workflow.

<img src="https://user-images.githubusercontent.com/31748898/151683116-299943b9-4e70-4a7b-8feb-16a3a351ca91.png" width="780" height="580"/>

## Papers and Citing WarpDrive

Our paper published at *Journal of Machine Learning Research* (JMLR) [https://jmlr.org/papers/v23/22-0185.html](https://jmlr.org/papers/v23/22-0185.html). You can also find more details in our white paper: [https://arxiv.org/abs/2108.13976](https://arxiv.org/abs/2108.13976).

If you're using WarpDrive in your research or applications, please cite using this BibTeX:

```
@article{JMLR:v23:22-0185,
  author  = {Tian Lan and Sunil Srinivasa and Huan Wang and Stephan Zheng},
  title   = {WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {316},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v23/22-0185.html}
}

@misc{lan2021warpdrive,
  title={WarpDrive: Extremely Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU}, 
  author={Tian Lan and Sunil Srinivasa and Huan Wang and Caiming Xiong and Silvio Savarese and Stephan Zheng},
  year={2021},
  eprint={2108.13976},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Tutorials and Quick Start

#### Tutorials
Familiarize yourself with WarpDrive by running these tutorials on Colab or [NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/warp_drive)!

- [WarpDrive basics(Introdunction and PyCUDA)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.a-warp_drive_basics.ipynb)
- [WarpDrive basics(Numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.b-warp_drive_basics.ipynb)
- [WarpDrive sampler(PyCUDA)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.a-warp_drive_sampler.ipynb)
- [WarpDrive sampler(Numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2.b-warp_drive_sampler.ipynb)
- [WarpDrive resetter and logger](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)
- [Create custom environments (PyCUDA)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.a-create_custom_environments_pycuda.md)
- [Create custom environments (Numba)](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4.b-create_custom_environments_numba.md)
- [Training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb)
- [Scaling Up training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-6-scaling_up_training_with_warp_drive.md)
- [Training with WarpDrive + Pytorch Lightning](https://github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-7-training_with_warp_drive_and_pytorch_lightning.ipynb)

You may also run these [tutorials](www.github.com/salesforce/warp-drive/blob/master/tutorials) *locally*, but you will need a GPU machine with nvcc compiler installed 
and a compatible Nvidia GPU driver. You will also need [Jupyter](https://jupyter.org). 
See [https://jupyter.readthedocs.io/en/latest/install.html](https://jupyter.readthedocs.io/en/latest/install.html) for installation instructions

#### Example Training Script
We provide some example scripts for you to quickly start the end-to-end training.
For example, if you want to train tag_continuous environment (10 taggers and 100 runners) with 2 GPUs and CUDA C backend
```
python example_training_script_pycuda.py -e tag_continuous -n 2
```
or switch to JIT compiled Numba backend with 1 GPU
```
python example_training_script_numba.py -e tag_continuous
```
You can find full reference documentation [here](http://opensource.salesforce.com/warp-drive/).

## Real World Problems and Collaborations

- [AI Economist Covid Environment with WarpDrive](https://github.com/salesforce/ai-economist/blob/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb): We train two-level multi-agent economic simulations using [AI-Economist Foundation](https://github.com/salesforce/ai-economist) and train it using WarpDrive. We specifically consider the COVID-19 and economy simulation in this example.
- [Climate Change Cooperation Competition](https://mila-iqia.github.io/climate-cooperation-competition/) collaborated with [Mila](https://mila.quebec/en/). We provide the base version of the RICE (regional integrated climate environment) [simulation environment](https://github.com/mila-iqia/climate-cooperation-competition).
- [Pytorch Lightning Trainer with WarpDrive](https://github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-7-training_with_warp_drive_and_pytorch_lightning.ipynb): We provide a [tutorial example](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/warp-drive.html) and a [blog article](https://devblog.pytorchlightning.ai/turbocharge-multi-agent-reinforcement-learning-with-warpdrive-and-pytorch-lightning-6be9b00a3a43) of a multi-agent reinforcement learning training loop with WarpDrive and [Pytorch Lightning](https://www.pytorchlightning.ai/).
- [NVIDIA NGC Catalog and Quick Deployment to VertexAI](https://catalog.ngc.nvidia.com/): WarpDrive image is hosted by [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/partners/teams/salesforce/containers/warpdrive). The NGC catalog "hosts containers for the top AI and data science software, tuned, tested and optimized by NVIDIA". Our tutorials also enable the quick deployment to VertexAI supported by the NGC.  

## Installation Instructions

To get started, you'll need to have **Python 3.7+** and the **nvcc** compiler installed 
with a compatible Nvidia GPU CUDA driver. 

CUDA (which includes nvcc) can be installed by following Nvidia's instructions here: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Docker Image

V100 GPU: You can refer to the [example Dockerfile](https://github.com/salesforce/warp-drive/blob/master/Dockerfile) to configure your system. 

A100 GPU: Our latest image is published and maintained by NVIDIA NGC. We recommend you download the latest image from [NGC catalog](https://catalog.ngc.nvidia.com/orgs/partners/teams/salesforce/containers/warpdrive).

If you want to build your customized environment, we suggest you visit [Nvidia Docker Hub](https://hub.docker.com/r/nvidia/cuda) to download the CUDA and cuDNN images compatible with your system.
You should be able to use the command line utility to monitor the NVIDIA GPU devices in your system:
```pyfunctiontypecomment
nvidia-smi
```
and see something like this
```pyfunctiontypecomment
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    32W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
In this snapshot, you can see we are using a Tesla V100 GPU and CUDA version 11.0.

### Installing using Pip

You can install WarpDrive using the Python package manager:

```pyfunctiontypecomment
pip install rl_warp_drive
```

### Installing from Source

1. Clone this repository to your machine:

    ```
    git clone https://www.github.com/salesforce/warp-drive
    ```

2. *Optional, but recommended for first tries:* Create a new conda environment (named "warp_drive" below) and activate it:

    ```
    conda create --name warp_drive python=3.7 --yes
    conda activate warp_drive
    ```

3. Install as an editable Python package:

    ```pyfunctiontypecomment
    cd warp_drive
    pip install -e .
    ```

### Testing your Installation

You can call directly from Python command to test all modules and the end-to-end training workflow.
```
python warp_drive/utils/unittests/run_unittests_pycuda.py
python warp_drive/utils/unittests/run_unittests_numba.py
python warp_drive/utils/unittests/run_trainer_tests.py
```

## Learn More

For more information, please check out our [blog](https://blog.einstein.ai/warpdrive-fast-rl-on-a-gpu/), [white paper](https://arxiv.org/abs/2108.13976), and code [documentation](http://opensource.salesforce.com/warp-drive/).

If you're interested in extending this framework, or have questions, join the
AI Economist Slack channel using this 
[invite link](https://join.slack.com/t/aieconomist/shared_invite/zt-g71ajic7-XaMygwNIup~CCzaR1T0wgA).
