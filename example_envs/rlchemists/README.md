# rlchemists

This is the source code folder for the research project published on Nature Communications ...

## structure
1. `en_array`: energy landscape mesh from DFT
2. `run_configs`: configs for different environments
3. `single_agent_one_atom`: src for one atom scenario
4. `single_agent_two_atom`: src for two atom scenario
5. `scripts`: example running scripts 

## installation
1. setup GPU environment and install `warpdrive` package as instructed
2. under the root directory of `rlchemists`, run `bash setenv.sh` to setup the Python path for this project
   
## run
We simply choose the environment and type to run a particular training, the supported ones are all included
in the `run_configs` folders, for example, `run_configs/single_agent_one_atom_diffusion2d` can be run by 
`python example_training_script_numba.py --env single_agent_one_atom --type diffusion2d`
