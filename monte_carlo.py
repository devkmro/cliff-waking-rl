import random

import numpy as np

from environment import (
    init_env,
    mark_path,
    check_game_over,
    get_state,
)
from qtable import init_q_table
from actions import (
    greedy_action,
    epsilon_greedy_action,
    move_agent,
    get_reward,
    compute_cum_rewards
)


STATE_DIM = 48
ACTION_DIM = 4


def update_q_table(q_table, state_action_trajectory, reward_trajectory, gamma, alpha):
    for t in range(len(reward_trajectory) - 1, 0, -1):
        reward = reward_trajectory[t]
        state, action = state_action_trajectory[t]

        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory) + reward
        q_table[action, state] += alpha * (cum_reward - q_table[action, state])


def monte_carlo(sim_input, sim_output) -> (np.array, list):
    """
    Monte Carlo: full-trajectory RL algorithm to train agent
    """
    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha = sim_input.alpha
    epsilon = sim_input.epsilon
    max_steps = sim_input.max_steps
    epsilon_delta = sim_input.epsilon_delta

    q_table = init_q_table()
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):
        # Initialize environment and agent position
        agent_pos, env, cliff_pos, goal_pos, game_over = init_env()

        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []
        state_action_trajectory = []
        current_epsilon = epsilon
        while not game_over:
            current_epsilon = 0 if episode == len(range(num_episodes)) - 1 else current_epsilon - epsilon_delta
            if steps_cache[episode] == 0:
                state = get_state(agent_pos)
                action = epsilon_greedy_action(state, q_table, current_epsilon)
            state = get_state(agent_pos)
            agent_pos = move_agent(agent_pos, action)
            env = mark_path(agent_pos, env)
            next_state = get_state(agent_pos)
            reward = get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward
            reward_trajectory.append(reward)
            state_action_trajectory.append([state, action])
            game_over = check_game_over(episode, next_state, cliff_pos, goal_pos, steps_cache[episode], max_steps)
            next_action = epsilon_greedy_action(next_state, q_table, current_epsilon)
            action = next_action
            steps_cache[episode] += 1
        update_q_table(q_table, state_action_trajectory, reward_trajectory, gamma, alpha)

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)
    sim_output.env_cache.append(env)  # array of np arrays
    sim_output.name_cache.append("Monte Carlo")
    return q_table, sim_output
