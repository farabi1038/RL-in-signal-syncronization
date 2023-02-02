# Deep Reinforcement Learning Setup for Foothill Dr., Salt Lake City, Utah #

### Install Miniconda ###
Follow instructions on the [Anaconda](https://docs.conda.io/en/latest/miniconda.html) website to install Miniconda for your operating system.

### Requirements ###
There are two requirements file: `conda_requirements.txt` and `requirements.txt`

To use the former:

```
conda create --name <env> python=<python_ver> --file conda_requirements.txt
```

While to use the latter:

```
pip install -r requirements.txt
```

### Executing ###
This project utilizes [Hydra](https://github.com/facebookresearch/hydra) as the argument management system. For more information, please visit the Github page and read through some examples and documentation.

This project contains several concepts to understand how everything works. In no particular order:

* [conf](https://bitbucket.org/kailiangisu/drl_foothill/src/master/conf/) 				| Configuration files that houses Hydra configuration files in *.yaml* format.
* [env](https://bitbucket.org/kailiangisu/drl_foothill/src/master/env/) 				| Folder consisting of python environment codes, along with State, Action, and Reward codes.
* [excel_files](https://bitbucket.org/kailiangisu/drl_foothill/src/master/excel_files/) | All excel files are placed inside this folder. These excel files are important to allow corridor-level traffic control to perform similar to real-world.
* [model](https://bitbucket.org/kailiangisu/drl_foothill/src/master/model/) 			| Folder consisting of Deep RL models.
* [sumo_files](https://bitbucket.org/kailiangisu/drl_foothill/src/master/sumo_files/) 	| Necessary sumo files to simulate a traffic environment. 
* [utils](https://bitbucket.org/kailiangisu/drl_foothill/src/master/utils/) 			| Utility functions are housed inside this folder. 



#### Training ####
To start training, run the following command-line:

```
python main.py
```

You will notice the following text in the command-line:

```
Training with the following config:
action:
  name: v1
env:
  local_resolution: 300
  name: foothill_v1
  render: false
  seed: 1337
  source: sumo_files/foothill.7am.sumocfg
model:
  batchsize: 32
  bound_mean: true
  clip_threshold: 5
  cseed: 1337
  entropy_coef: 0.0
  epochs: 10
  gpu: -1
  lr: 0.0003
  name: ppo_v1
  standardize_advantages: false
  update_interval: 128
  weight_decay: 0.0
reward:
  name: v1
state:
  name: v1
  size: 8
train:
  debug: false
  max_t: 10
  n_episodes: 100
```

These are the modifiable configs from Hydra. If you want to change any one of them, i.e. local_resolution of the environment:

```
python main.py env.local_resolution=900
```

would change the displayed local_resolution configuration under env to 900. 

```
. . .
env:
  local_resolution: 900
. . .
```

The environment engine is in `env_v1.py` located under `env` folder.

#### Testing ####
To test a trained agent, make sure all configurations are similar when training it. Then simply run the following command-line along with the saved agent path:

```
python main_test.py test.load=<trained_agent_path>
```

#### Benchmarking ####
To run benchmark a run with a specific DOT plan, add the split and plan *.csv* files in the command-line like so:

```
python main_benchmark.py benchmark.split=<insert split csv here> benchmark.plan=<insert plan csv here>
```
 

### File Structure ###

```
|-- conf
	|-- action
		|-- *.yaml
	|-- env
		|-- *.yaml
	|-- model
		|-- *.yaml
	|-- reward
		|-- *.yaml
	|-- state
		|-- *.yaml
	|-- train
		|-- *.yaml
	|-- config.yaml
|-- env
	|-- action
		|-- action_selector.py
		|-- action_v*.py
	|-- reward
		|-- reward_selector.py
		|-- reward_v*.py
	|-- state
		|-- state_selector.py
		|-- state_v*.py
	|-- env_selector.py
	|-- env_v1.py
|-- excel_files
	|-- *.csv
|-- model
	|-- model_selector.py
	|-- model_v*.py
|-- outputs
	|-- output_directories by date and time
|-- sumo_files
	|-- dynamic
		|-- ped.route.*.xml
		|-- ped.tls.*.xml
		|-- routes.sample.*.xml
	|-- static
		|-- det.gapout.add.xml
		|-- foothill.net.xml
		|-- metric.*.add.xml
		|-- metric.res.*.xml
		|-- osm.poly.xml
		|-- osm.view.xml
	|-- *.sumocfg
|-- utils
	|-- functions.py
	|-- tingting.py
	|-- trafficLight.py
|-- main.py
|-- training_module.py
```
