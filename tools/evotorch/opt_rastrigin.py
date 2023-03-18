from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger
from evotorch import Problem, Solution
import torch
import math



######################################
# Non-vectorized version
######################################

class OffsetRastrigin(Problem):
    def __init__(self, d: int = 25, A: int = 10):

        super().__init__(
            objective_sense="min",
            solution_length=d,
            initial_bounds=(-1, 1),
        )

        # Store the A parameter for evaluation
        self._A = A
        # Generate a random offset with center 0 and standard deviation 1
        self._x_prime = self.make_gaussian(d, center=0.0, stdev=1.0)

    def _evaluate(self, solution: Solution):
        x = solution.values
        z = x - self._x_prime
        f = (self._A * self.solution_length) + torch.sum(
            z.pow(2.0) - self._A * torch.cos(2 * math.pi * z)
        )
        solution.set_evals(f)

prob = OffsetRastrigin(d=14, A=5)
searcher = SNES(prob, stdev_init=5)
logger = StdOutLogger(searcher)

# single step
searcher.step()

searcher.run(num_generations=10)




######################################
# Vectorized version
######################################

from evotorch import SolutionBatch


class VecOffsetRastrigin(Problem):
    def __init__(self, d: int = 25, A: int = 10):

        super().__init__(
            objective_sense="min",
            solution_length=d,
            initial_bounds=(-1, 1),
        )

        # Store the A parameter for evaluation
        self._A = A
        # Generate a random offset with center 0 and standard deviation 1
        self._x_prime = self.make_gaussian((1, d), center=0.0, stdev=1.0)

    # Overwrite the _evaluate_batch that takes a stacked list of soltuions
    def _evaluate_batch(self, solutions: SolutionBatch):
        xs = solutions.values
        zs = xs - self._x_prime
        fs = (self._A * self.solution_length) + torch.sum(
            zs.pow(2.0) - self._A * torch.cos(2 * math.pi * zs), dim=-1
        )
        solutions.set_evals(fs)
