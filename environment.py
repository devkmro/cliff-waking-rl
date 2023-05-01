import numpy as np

""" Tile layout (36=start, 47=goal, 37-46=cliff)
0	1	2	3	4	5	6	7	8	9	10	11
12	13	14	15	16	17	18	19	20	21	22	23
24	25	26	27	28	29	30	31	32	33	34	35
36	37	38	39	40	41	42	43	44	45	46	47
"""

def check_game_over(episode:int,
    state: int, cliff_pos: np.array, goal_pos: int, number_of_steps: int, max_steps: int
) -> bool:
    """
    Function returns reward in the given state
    """
    # Game over when reached goal, fell down cliff, or exceeded max_steps
    game_over = (
        True
        if (state == goal_pos   or   state in cliff_pos or number_of_steps == max_steps - 1)
        else False
    )
    if state == goal_pos and number_of_steps > 1:
        print("===== Goal reached (episode", episode ,") =====")

    return game_over


def get_initial_random_position():
    #state = np.random.choice(12)
    action = np.random.choice(4)

    return action, 0


def init_env(expoloring_starts=False) -> (tuple, np.array, np.array, int, bool):
    """Initialize environment and agent position"""
    agent_pos = (3, 0)  # Left-bottom corner (start)
    env = np.zeros((4, 12), dtype=int)
    env = mark_path(agent_pos, env)
    cliff_states = np.arange(37, 47)  # States for cliff tiles
    goal_state = 47  # State for right-bottom corner (destination)
    game_over = False

    return agent_pos, env, cliff_states, goal_state, game_over


def mark_path(agent: tuple, env: np.array) -> np.array:
    """
    Store path taken by agent
    Only needed for visualization
    """
    (posY, posX) = agent
    env[posY][posX] += 1

    return env


def env_to_text(env: np.array) -> str:
    """
    Convert environment to text format
    Needed for visualization in console
    """
    env = np.where(env >= 1, 1, env)

    env = np.array2string(env, precision=0, separator=" ", suppress_small=False)
    env = env.replace("[[", " |")
    env = env.replace("]]", "|")
    env = env.replace("[", "|")
    env = env.replace("]", "|")
    env = env.replace("1", "x")
    env = env.replace("0", " ")

    return env


def get_state(agent_pos: tuple) -> int:
    """
    Obtain state corresponding to agent position
    """
    x_dim = 12
    (pos_x, pos_y) = agent_pos
    state = x_dim * pos_x + pos_y

    return state
