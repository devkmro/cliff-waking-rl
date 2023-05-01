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


def update_q_table(
    q_table: np.array,
    state: int,
    action: int,
    reward: int,
    next_state_value: float,
    gamma: float,
    alpha: float,
) -> np.array:
    """
    Update Q-table based on observed rewards and next state value
    For SARSA (on-policy):
    Q(S, A) <- Q(S, A) + [α * (r + (γ * Q(S', A'))) -  Q(S, A)]

    For Q-learning (off-policy):
    Q(S, A) <- Q(S, A) + [α * (r + (γ * max(Q(S', A*)))) -  Q(S, A)
    """
    # Compute new q-value
    new_q_value = q_table[action, state] + alpha * (
        reward + (gamma * next_state_value) - q_table[action, state]
    )

    # Replace old Q-value
    q_table[action, state] = new_q_value

    return q_table


def sarsa(sim_input, sim_output) -> (np.array, list):
    """
    SARSA: on-policy RL algorithm to train agent
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
        current_epsilon = epsilon
        while not game_over:
            current_epsilon = 0 if episode == len(range(num_episodes)) - 1 else current_epsilon - epsilon_delta
            if steps_cache[episode] == 0:
                # Get state corresponding to agent position
                state = get_state(agent_pos)

                # Select action using ε-greedy policy
                action = epsilon_greedy_action(state, q_table, current_epsilon)

            # Move agent to next position
            agent_pos = move_agent(agent_pos, action)

            # Mark visited path
            env = mark_path(agent_pos, env)

            # Determine next state
            next_state = get_state(agent_pos)

            # Compute and store reward
            reward = get_reward(next_state, cliff_pos, goal_pos)
            rewards_cache[episode] += reward

            # Check whether game is over
            game_over = check_game_over(episode, next_state, cliff_pos, goal_pos, steps_cache[episode], max_steps)

            # Select next action using ε-greedy policy
            next_action = epsilon_greedy_action(next_state, q_table, current_epsilon)

            # Determine Q-value next state (on-policy)
            next_state_value = q_table[next_action][next_state]

            # Update Q-table
            q_table = update_q_table(
                q_table, state, action, reward, next_state_value, gamma, alpha
            )

            # Update state and action
            state = next_state
            action = next_action

            steps_cache[episode] += 1

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)  # array of np arrays
    sim_output.name_cache.append("SARSA")

    return q_table, sim_output