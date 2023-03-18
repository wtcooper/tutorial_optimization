# https://docs.evotorch.ai/v0.3.0/user_guide/algorithm_usage/#accessing-the-status

from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
import time
import matplotlib
matplotlib.use('TkAgg')

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

######################################
# Non-vectorized environment - 1D vector for single agent
######################################
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))

problem_nonvec = Problem(
    objective_sense="min",
    objective_func=sphere,
    solution_length=5,
    # initial_bounds=(-1, 1),
    # initial_bounds=([-1,-1,-1,-1,-1], [1,1,1,1,1]), # expliciely set bounds for each
    bounds=([-1,-1,-1,-1,-1], [1,1,1,1,1]), # if leave initial_bounds blank and set bounds, initial bounds the same
    num_actors=1,
    # dtype=torch.float16,  # change the dtype
    # # device="cuda:0",  # change the device
    device="cuda",  # change the device
)

# Define the search algo
searcher = SNES(problem_nonvec, stdev_init=5)

stdout_logger = StdOutLogger(searcher)  # Status printed to the stdout
pandas_logger = PandasLogger(searcher)  # Status stored in a Pandas dataframe

# single step
# searcher.step()

# Run the algorithm for as many iterations as desired
start_time = time.time()
searcher.run(100)
run_time = time.time() - start_time
print(f"{run_time} s")

# Process the information accumulated by the loggers.
progress = pandas_logger.to_dataframe()
progress.mean_eval.plot()  # Display a graph of the evolutionary progress by using the pandas data frame



print([k for k in searcher.iter_status_keys()])
best_discovered_solution = searcher.status["pop_best"]
print(best_discovered_solution)


######################################
# Vectorized version - get the paramters as stacked batch for batch of agents
######################################
def vectorised_sphere(xs: torch.Tensor) -> torch.Tensor:
    return torch.sum(xs.pow(2.0), dim=-1)

problem_vec = Problem(
    objective_sense="min",
    objective_func=vectorised_sphere,
    vectorized=True,
    solution_length=5,
    initial_bounds=(-1, 1),
    num_actors=2,  # 2
    # dtype=torch.float16,  # change the dtype
    # device="cuda:0",  # change the device
)

# Define the search algo
searcher = SNES(problem_vec, stdev_init=5)

# single step
searcher.step()

# run through generations
searcher.run(num_generations=100)


######################################
# Vectorized custom problem on GPU
######################################

# - need SolutionBatch and push solutions to gpu .to(self.aux_device)


from evotorch import SolutionBatch

class VecSphere(Problem):
    def _evaluate_batch(self, solutions: SolutionBatch):
        xs = solutions.values.to(self.aux_device)
        fs = vectorised_sphere(xs)
        solutions.set_evals(fs.to(solutions.device))

problem = VecSphere(
    objective_sense="min",
    solution_length=10,
    initial_bounds=(-1, 1),
    num_actors=2,  # 2 ray actors
    num_gpus_per_actor=0.5,   # spread each ray actor to half of the 1 GPU
)

searcher = SNES(problem_nonvec, stdev_init=5)

# single step
searcher.step()

# run through generations
searcher.run(num_generations=100)


######################################
# accessing the state
######################################
print([k for k in searcher.iter_status_keys()])
best_discovered_solution = searcher.status["pop_best"]

# 'best', the best discovered solution so far.
# 'worst', the worst discovered solution so far.
# 'best_eval', the fitness of the best discovered solution so far.
# 'worst_eval', the fitness of the worst discovered solution so far.
# 'pop_best', the best solution in the population.
# 'pop_best_eval', the fitness of the best solution in the population.
# 'mean_eval', the mean fitness of the population.
# 'median_eval', the best solution in the population.
