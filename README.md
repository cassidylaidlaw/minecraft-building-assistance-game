# Minecraft Building Assistance Game

The idea of this project is to train a Minecraft AI that can help a person build a house, even though it initially doesn't know what kind of house the person is trying to build. The idea is to develop a general framework for AI to communicate with people about their goals and then help them achieve those goals. The variety of building materials, styles, and layouts possible in Minecraft reflects the complexity of people's goals in the real world. [Here's a document](https://docs.google.com/document/d/1OFFqyfHH55g8XXDsWV9ZyTasMjVPRFjqCEPsNhp6d9Y/edit?usp=sharing) with some motivation and ideas for the project.

This repository contains the implementation of the Minecraft Building Assistance Game (MBAG) environment. The multiagent environment can run on its own or connect to running instances of Minecraft via [Project Malmo](https://github.com/microsoft/malmo) for visualization or for interaction with human players.

## Setup

This section describes how to set up your development environment.

### Installing dependencies

First, install Python 3.8, 3.9, or 3.10. Then, run one of the following commmands:

  * Install just the environment: `pip install -e .`
  * Also install the RLlib dependencies for training and running assistants: `pip install -e .[rllib]`
  * Also install the Malmo interface to run assistants in Minecraft: `pip install -e .[rllib,malmo]`

### Linting, testing, and type checking

The project uses various tools to maintain code quality. To install them all, run

    pip install --upgrade -e .[dev]

Then, run linting and testing with the following commands:

    ./lint.sh
    pytest -m "not uses_malmo"

To run fewer tests, you can some or all of these additional filters:

    pytest -m "not uses_malmo and not slow"
    pytest -m "not uses_malmo and not uses_rllib"
    pytest -m "not uses_malmo and not uses_cuda"

## Running assistants in Minecraft

Playing with assistants in Minecraft and/or collecting data of humans playing takes two steps: first, starting Minecraft instances, and second, connecting to those instances to run an episode within the MBAG environment.

### Starting Minecraft

To start Minecraft instances, run the following command, assuming you've installed the Malmo dependencies (see [setup](#setup)):

    python -m malmo.minecraft launch --num_instances 2 --goal_visibility True False

Set the number of Minecraft instances to launch with the `--num_instances` option. You need at least two instances to play with an assistant (one for the human and one for the AI assistant); if you want to record video of the game, start an additional instance for a "spectator" player.

The `--goal_visibility` argument controls which instances show the goal house as a transparent blueprint within the game. Generally you should set this to True for the first instances and False for additional instances (e.g., `--goal_visibility True False False` for three instances).

### Running an MBAG episode

Once the Minecraft instances are running, you can use the following command to start an episode playing with an assistant:

    python -m mbag.scripts.evaluate with human_with_assistant assistant_checkpoint=data/example_assistant/checkpoint_000100

To play an episode without an assistant, run:

    python -m mbag.scripts.evaluate with human_alone assistant_checkpoint=data/example_assistant/checkpoint_000100

The `assistant_checkpoint` argument is still needed in this case to load the environment configuration. The episode will automatically terminate when the house is completed, but if you want to end it sooner, use Ctrl+C. At the end of the episode, the episode data will be saved and metrics will be printed out.
