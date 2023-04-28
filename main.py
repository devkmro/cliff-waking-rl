import numpy as np

from plot import plot_steps, plot_rewards, console_output, plot_path
from monte_carlo import monte_carlo


if __name__ == "__main__":
    class sim_init:
        def __init__(self, num_episodes, gamma, alpha, epsilon):
            self.num_episodes = num_episodes
            self.gamma = gamma
            self.alpha = alpha
            self.epsilon = epsilon

        def __str__(self):
            return f"# episodes: {self.num_episodes}, gamma: {self.gamma}, alpha: {self.alpha}, epsilon: {self.epsilon}"

    run_algorithms = {
        "Monte Carlo"
    }



    class sim_output:
        def __init__(self, rewards_cache, step_cache, env_cache, name_cache):
            self.reward_cache = rewards_cache  # list of rewards
            self.step_cache = step_cache  # list of steps
            self.env_cache = env_cache  # list of final paths
            self.name_cache = name_cache  # list of algorithm names

    sim_output = sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[]
    )

    # Run Monte Carlo
    if "Monte Carlo" in run_algorithms:
        sim_input = sim_init(num_episodes=10000, gamma=0.8, alpha=0.01, epsilon=0.1)
        q_table_mc, sim_output = monte_carlo(sim_input, sim_output)

    # Print console output
    console_output(
        sim_output,
        sim_input.num_episodes,
    )

    # Plot output
    plot_steps(sim_output)
    plot_rewards(sim_output)
    plot_path(sim_output)
