# --- Simulation Parameters ---
TOPOLOGY_FILE = 'topologies/nsfnet.json'
NUM_AGENTS = 4
TRAFFIC_MODEL = 'gravity'
MAX_STEPS_PER_EPISODE = 100

# --- Training Parameters ---
NUM_EPISODES = 500
N_STEPS = 20

# --- IMPROVEMENT: Increased Learning Rate ---
# The previous rate was too conservative. Let's learn faster.
LEARNING_RATE = 3e-4 # Was 1e-4

GAMMA = 0.99

# --- Agent and Model Parameters ---
GNN_HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
ACTION_SPACE_SIZE = 10

# --- IMPROVEMENT: Amplified Reward Signal ---
# We make the penalties much larger to create a stronger learning signal.
# A small improvement in utilization will now result in a much bigger change in reward.
REWARD_WEIGHTS = {
    'utilization': -10.0,  # Was -1.0
    'latency': -1.0        # Was -0.1
}

# --- IMPROVEMENT: Increased Exploration ---
# We significantly increase the entropy bonus to encourage the agent
# to try new actions and escape the local minimum.
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.05      # Was 0.01

# --- Logging and Saving ---
LOG_INTERVAL = 10
MODEL_SAVE_PATH = 'results/marl_agent.pth' # Save to a new file
PLOT_SAVE_PATH = 'results/training_rewards.png'