# Minecraft Building Assistance Game

The idea of this project is to train a Minecraft AI that can help a person build a house, even though it initially doesn't know what kind of house the person is trying to build. The idea is to develop a general framework for AI to communicate with people about their goals and then help them achieve those goals. The variety of building materials, styles, and layouts possible in Minecraft reflects the complexity of people's goals in the real world. [Here's a document](https://docs.google.com/document/d/1OFFqyfHH55g8XXDsWV9ZyTasMjVPRFjqCEPsNhp6d9Y/edit?usp=sharing) with some motivation and ideas for the project.

This repository contains the implementation of the Minecraft Building Assistance Game (MBAG) environment. The multiagent environment can run on its own or connect to running instances of Minecraft via [Project Malmo](https://github.com/microsoft/malmo) for visualization or for interaction with human players.

## Setup

This section describes how to set up your development environment.

### Installing dependencies

First, install Python (≥3.8 tested but lower might work). Then, run

    pip install -r requirements.txt

to install all other dependencies.

### Linting, testing, and type checking

The project uses various tools to maintain code quality. To install them all, run

    pip install flake8 black mypy pytest

Then, you can run the following commands:
 * `black mbag tests`: automatic formatting with [Black](https://black.readthedocs.io/en/stable/).
 * `flake8 mbag tests`: linting with [Flake8](https://flake8.pycqa.org/en/latest/).
 * `mypy mbag tests`: type checking with [MyPy](http://mypy-lang.org/).
    * *Note:* due to this [bug](https://github.com/ray-project/ray/issues/14431), on Linux you may have to run `touch /path/to/site-packages/ray/py.typed`, replacing `/path/to/site-packages` depending on where python is installed, to make type checking work.
 * `pytest`: run tests with [PyTest](https://docs.pytest.org/en/6.2.x/).

### Integration with Minecraft via Project Malmo

The MBAG environment is implemented using a highly simplified Minecraft simulator that can run much, much faster than Minecraft itself. However, it can also connect to running Minecraft instances for interaction with human players via [Project Malmo](https://github.com/microsoft/malmo). The following steps describe how to set up the Minecraft interface. *Note: it can be somewhat difficult to build Malmo; reach out to [Cassidy](mailto:cassidy_laidlaw@berkeley.edu) with any issues.*

 1. Follow the Project Malmo build instructions for [Linux](https://github.com/microsoft/malmo/blob/master/doc/build_linux.md), [macOS](https://github.com/microsoft/malmo/blob/master/doc/build_macosx.md), or [Windows](https://github.com/microsoft/malmo/blob/master/doc/build_windows.md). **Stop before you run `make install`.**
 2. Instead of running `make install`, just run `make MalmoPython`.
 3. Run the following command to install the Malmo python package:
    
        ln -s `pwd`/Malmo/src/PythonWrapper/MalmoPython.so `python -c 'import site; print(site.getsitepackages()[0])'`

Now, whenever you want to interface with Minecraft from the MBAG environment, follow these steps:

 1. `cd` to the directory where you installed Malmo and then `cd` to the `Minecraft` subdirectory.
 2. Make sure `JAVA_HOME` is set for Java 8, which you should have installed while building Malmo.
 3. Run `./launchClient.sh`, once for MBAG player (i.e., run one instance for one player, two instances for two players, etc.). This should open a window running Minecraft.
 4. Once Minecraft is running, use a seperate terminal tab to run MBAG with `use_malmo` set to `True` in the configuration and it should automatically connect to the Minecraft instances. For instance, running the following test should connect to a single Minecraft instance and do some basic block breaking and placing:

        pytest tests/test_evaluator.py -k test_malmo
 5. For debugging purposes, it can be useful to take control of the player while the MBAG is controlling it. To do so, press `Return` in the Minecraft window. Pressing `Return` again will give control back to MBAG.

## Package layout

This section describes the layout and contents of the various Python packages.

 * `mbag.environment`: contains modules implementing the core MBAG environment.
    * `.mbag_env`: contains `MbagEnv`, which implements the core environment with a [Gym](https://gym.openai.com/)-like interface.
    * `.types`: contains various useful type definitions and a wrapper class for actions.
    * `.blocks`: contains the `MinecraftBlocks` class, which provides various methods for interfacing with a 3d grid of Minecraft blocks.
    * `.malmo`: contains the `MalmoClient` class, which allows for easy interfacing with Minecraft through Project Malmo.
    * `.goals`: contains modules with various "goal generators," which provide ways of generating goal structures for an agent to build.
       * `.goal_generator`: defines the `GoalGenerator` abstract base class.
       * `.simple`: very basic goal generators (e.g., random blocks).
       * `.craftassist`: houses from the [CraftAssist house dataset](https://github.com/facebookresearch/craftassist#datasets).
       * `.grabcraft`: houses scraped from [GrabCraft](https://www.grabcraft.com/).
 * `mbag.agents`: defines agents which interact in the MBAG environment.
    * `.mbag_agent`: defines the `MbagAgent` abstract base class.
    * `.heuristic_agents`: defines basic heuristic-based agents.
 * `mbag.evaluation.evaluator`: defines the `MbagEvaluator` class, which allows one to test a set of agents in an MBAG environment.
 * `mbag.rllib`: modules and scripts for training MBAG policies using reinforcement learning with [RLlib](https://www.ray.io/rllib).

Tests are contained in the `tests` directory.

## Training agents with RL

*Coming soon!*
