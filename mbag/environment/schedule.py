"""Schedule classes for time-dependent values.

These are mostly copied from ray.rllib.utils.schedules, but with minor changes:
- Set return values of the schedules to float.
- Removed definition of framework and TensorFlow implementations.
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Tuple


class Schedule(metaclass=ABCMeta):
    """Schedule classes implement various time-dependent scheduling schemas.

    - Constant behavior.
    - Linear decay.
    - Piecewise decay.
    - Exponential decay.

    Useful for backend-agnostic rate/weight changes for learning rates,
    exploration epsilons, beta parameters for prioritized replay, loss weights
    decay, etc.

    Each schedule can be called directly with the `t` (absolute time step)
    value and returns the value dependent on the Schedule and the passed time.
    """

    def value(self, t: int) -> float:
        """Generates the value given a timestep (based on schedule's logic).

        Args:
            t: The time step.

        Returns:
            The calculated value depending on the schedule and `t`.
        """
        return self._value(t)

    def __call__(self, t: int) -> float:
        """Simply calls self.value(t). Implemented to make Schedules callable."""
        return self.value(t)

    @abstractmethod
    def _value(self, t: int) -> float:
        """
        Returns the value based on a time step input.

        Args:
            t: The time step.

        Returns:
            The calculated value depending on the schedule and `t`.
        """
        raise NotImplementedError


class ConstantSchedule(Schedule):
    """A Schedule where the value remains constant over time."""

    def __init__(self, value: float):
        """Initializes a ConstantSchedule instance.

        Args:
            value: The constant value to return, independently of time.
        """
        super().__init__()
        self._v = value

    def _value(self, t: int) -> float:
        return self._v


def _linear_interpolation(left, right, alpha):
    return left + alpha * (right - left)


class PiecewiseSchedule(Schedule):
    """Implements a Piecewise Scheduler."""

    def __init__(
        self,
        endpoints: List[Tuple[int, float]],
        interpolation: Callable[
            [
                float,
                float,
                float,
            ],
            float,
        ] = _linear_interpolation,
        outside_value: Optional[float] = None,
    ):
        """Initializes a PiecewiseSchedule instance.

        Args:
            endpoints: A list of tuples
                `(t, value)` such that the output
                is an interpolation (given by the `interpolation` callable)
                between two values.
                E.g.
                t=400 and endpoints=[(0, 20.0),(500, 30.0)]
                output=20.0 + 0.8 * (30.0 - 20.0) = 28.0
                NOTE: All the values for time must be sorted in an increasing
                order.
            framework: The framework descriptor string.
            interpolation: A function that takes the left-value,
                the right-value and an alpha interpolation parameter
                (0.0=only left value, 1.0=only right value), which is the
                fraction of distance from left endpoint to right endpoint.
            outside_value: If t in call to `value` is
                outside of all the intervals in `endpoints` this value is
                returned. If None then an AssertionError is raised when outside
                value is requested.
        """
        super().__init__()

        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self.interpolation = interpolation
        self.outside_value = outside_value
        self.endpoints = [(int(e[0]), float(e[1])) for e in endpoints]

    def _value(self, t: int) -> float:
        # Find t in our list of endpoints.
        for (l_t, l), (r_t, r) in zip(self.endpoints[:-1], self.endpoints[1:]):
            # When found, return an interpolation (default: linear).
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return float(self.interpolation(l, r, alpha))

        # t does not belong to any of the pieces, return `self.outside_value`.
        assert self.outside_value is not None
        return self.outside_value
