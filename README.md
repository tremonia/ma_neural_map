# ma_neural_map


## nm_baselines
contains neural map's customized baseline pkg,
i.e. learning algorithms and the model
 https://github.com/openai/baselines.git
master branch checked out on commit ea25b9e from 2020.01.31

### customized files:
- common/policies.py
- common/runners.py
- common/models.py
- ppo2/ppo2.py
- ppo2/runner.py
- ppo2/model.py
- a2c/a2c.py
- a2c/runner.py
- deepq/deepq.py (only baseline's pre-implemented models)


## nm_envs
contains all of neural map's environments:

- 2 Indicators, 2 Goals, 2 Dimension (as described in the paper)


## nm_docker
contains the docker file

basline pkg and neural_map_env are installed via "pip install -e"


## neural_map.py
- this is the project's main script that contains all parameters and the calls of the baseline pkg
- wrapped with sacred (@ex.config and @ex.automain)
