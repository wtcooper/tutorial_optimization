from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from evotorch.neuroevolution import GymNE

# Declare the problem to solve
problem = GymNE(
    env_name="LunarLander-v2",  # Solve the Humanoid-v4 task
    # env_name="Humanoid-v4",  # Solve the Humanoid-v4 task
    network="Linear(obs_length, act_length)",  # Linear policy
    observation_normalization=True,  # Normalize the policy inputs
    decrease_rewards_by=5.0,  # Decrease each reward by 5.0
    num_actors="max",  # Use all available CPUs
    # num_actors=4,    # Explicit setting. Use 4 actors.
)

# Instantiate a PGPE algorithm to solve the problem
searcher = PGPE(
    problem,

    # Base population size
    popsize=200,

    # For each generation, sample more solutions until the
    # number of simulator interactions reaches this threshold
    num_interactions=int(200 * 1000 * 0.75),

    # Stop re-sampling solutions if the current population size
    # reaches or exceeds this number.
    popsize_max=3200,

    # Learning rates
    center_learning_rate=0.0075,
    stdev_learning_rate=0.1,

    # Radius of the initial search distribution
    radius_init=0.27,

    # Use the ClipUp optimizer with the specified maximum speed
    optimizer="clipup",
    optimizer_config={"max_speed": 0.15},
)

# Instantiate a standard output logger
_ = StdOutLogger(searcher)

# Run the algorithm for the specified amount of generations
searcher.run(500)

# Get the center point of the search distribution,
# obtain a policy out of that point, and visualize the
# agent using that policy.
center_solution = searcher.status["center"]
trained_policy = problem.make_net(center_solution)
problem.visualize(trained_policy)