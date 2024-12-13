from typing import List, Optional, TypedDict

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator, GoalGeneratorConfig


class GoalTransform(GoalGenerator):
    """
    GoalTransforms are transformations that can be applied to generated
    goals. A GoalTransform can reject certain goals, alter a goal, and so on. It
    takes as input a GoalGenerator and should call this to get input goals; then, it
    modify them or ask for more goals if necessary.
    """

    def __init__(self, config: GoalGeneratorConfig, goal_generator: GoalGenerator):
        super().__init__(config)
        self.goal_generator = goal_generator

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        return self.goal_generator.generate_goal(size)


class GoalTransformSpec(TypedDict, total=False):
    transform: str
    config: GoalGeneratorConfig


class TransformedGoalGeneratorConfig(TypedDict):
    goal_generator: str
    goal_generator_config: GoalGeneratorConfig
    transforms: List[GoalTransformSpec]


class TransformedGoalGenerator(GoalGenerator):
    """
    Starts with a base goal generator and then applies GoalTransforms to it.
    """

    config: TransformedGoalGeneratorConfig

    def __init__(self, config: GoalGeneratorConfig):
        super().__init__(config)

        from . import ALL_GOAL_GENERATORS, ALL_GOAL_TRANSFORMS

        base_goal_generator_class = ALL_GOAL_GENERATORS[self.config["goal_generator"]]
        self.base_goal_generator = base_goal_generator_class(
            self.config["goal_generator_config"]
        )
        goal_generator = self.base_goal_generator
        self.goal_transforms: List[GoalTransform] = []
        # Recursively applies each goal transform in self.config["goal_transforms"]
        # to the base generator. E.g. if self.config["goal_transforms"] is
        # [{"transform": "foo"}, {"transform": "bar"}], this sets the new goal
        # generator to BarTransform(FooTransform(base_generator)).
        for transform_spec in self.config["transforms"]:
            transform_class = ALL_GOAL_TRANSFORMS[transform_spec["transform"]]
            goal_generator = transform_class(
                transform_spec.get("config", {}), goal_generator
            )
            self.goal_transforms.append(goal_generator)

        self.goal_generator = goal_generator

    @property
    def num_remaining_goals(self) -> Optional[int]:
        return self.base_goal_generator.num_remaining_goals

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        return self.goal_generator.generate_goal(size)
