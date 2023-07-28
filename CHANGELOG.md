# Changelog
# Release 2.5 (2022-07-27)
- Introduce environment reset pool, so concurrent enviornment replicas can randomly reset themselves from the pool.

# Release 2.4 (2022-06-16)
- Introduce new device context management and autoinit_pycuda 
- Therefore, Torch (any version) will not conflict with PyCUDA in the GPU context 

# Release 2.3 (2022-03-22)
- Add ModelFactory class to manage custom models
- Add Xavier initialization for the model
- Improve trainer.fetch_episode_states() so it can fetch (s, a, r) and can replay with argmax.

# Release 2.2 (2022-12-20)
- Factorize the data loading for placeholders and batches (obs, actions and rewards) for the trainer.

# Release 2.1 (2022-10-26)
- v2 trainer integration with Pytorch Lightning

# Release 2.0 (2022-09-20)
Big release:
- WarpDrive:
  - Added data and function managers for both CUDA C and Numba.
  - Added core library (sampler and reset) for Numba.  
  - Dual environment backends, supporting both CUDA C and Numba.
  - Training pipeline compatible with both CUDA C and Numba.
  - Full backward compatibility with version 1.
- Environments
  - tag (continuous version) implemented in Numba.
  - tag (gridworld version) implemented in Numba.
 
# Release 1.7 (2022-09-08)
- Update PyCUDA version to 2022.1

# Release 1.6 (2022-04-05)
- Allow for envs to span multiple blocks, adding the capability to train simulations with thousands of agents.

# Release 1.5 (2022-03-01)
- Trainer integration with Pytorch Lightning (https://www.pytorchlightning.ai/).

# Release 1.4 (2022-01-29)
- Added multi-GPU support.

# Release 1.3 (2022-01-10)
- Auto-scaling to maximize the number of environment replicas and training batch size (on a single GPU).
- Added Python logging.

# Release 1.2.2 (2021-12-16)
- Added a trainer module to fetch environment states for an episode.

# Release 1.2.1 (2021-12-07)
- Add policy-specific training parameters.

# Release 1.2 (2021-12-02)
- Added a parameter scheduler.
- Option to push a list of data arrays to the GPU at once.
- Option to pass multiple arguments to the CUDA step function as a list.
- CUDA utility to help index multi-dimensional arrays.
- Log the episodic rewards.
- Save metrics during training.
 
# Release 1.1 (2021-09-27)
- Support to register custom environments.
- Support for 'Dict' observation spaces.

# Release 1.0 (2021-09-01)
- WarpDrive
  - data and function managers.
  - CUDA C core library.
  - environment wrapper.
  - Python (CPU) vs. CUDA C (GPU) simulation implementation consistency checker
  - training pipeline (with FC network, and A2C, PPO agents).
- Environments
  - tag (grid-world version).
  - tag (continuous version).

