# Changelog

# Release 1.2.2 (2021-12-16)
- Added a trainer module to fetch environment states for an episode

# Release 1.2.1 (2021-12-07)
- Add policy-specific training parameters

# Release 1.2 (2021-12-02)
- Added a parameter scheduler
- Option to push a list of data arrays to the GPU at once
- Option to pass multiple arguments to the CUDA step function as a list
- CUDA utility to help index multi-dimensional arrays
- Log the episodic rewards
- Save metrics during training
 
# Release 1.1 (2021-09-27)
- Support to register custom environments
- Support for 'Dict' observation spaces

# Release 1.0 (2021-09-01)
- WarpDrive
  - data and function managers
  - CUDA C core library
  - environment wrapper
  - Python (CPU) vs. CUDA C (GPU) simulation implementation consistency checker
  - training pipeline (with FC network, and A2C, PPO agents)
- Environments
  - tag (grid-world version)
  - tag (continuous version)

