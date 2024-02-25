# Import the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# Define some hyperparameters
BATCH_SIZE = 64 # the size of the minibatch for training
GAMMA = 0.99 # the discount factor for future rewards
EPS_START = 0.9 # the initial value of epsilon for the epsilon-greedy exploration
EPS_END = 0.05 # the final value of epsilon
EPS_DECAY = 200 # the number of steps to decay epsilon
TARGET_UPDATE = 10 # the frequency of updating the target network
MEMORY_SIZE = 10000 # the size of the replay memory

# Define the Q-network class
class QNetwork(nn.Module):
    # Initialize the network with some parameters
    def __init__(self, input_size, output_size, hidden_size):
        # input_size: the size of the input features
        # output_size: the size of the output actions
        # hidden_size: the size of the hidden layer

        # Call the parent class constructor
        super(QNetwork, self).__init__()

        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # Define the forward pass of the network
    def forward(self, x):
        # x: the input features

        # Apply the activation function (ReLU) to the first layer
        x = F.relu(self.fc1(x))

        # Return the output of the second layer
        return self.fc2(x)

# Create the environment
env = StockTradingEnv(data, initial_balance, commission, max_steps)

# Get the size of the input features and the output actions
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Create the Q-network and the target network
q_network = QNetwork(input_size, output_size, hidden_size).to(device)
target_network = QNetwork(input_size, output_size, hidden_size).to(device)

# Copy the weights from the Q-network to the target network
target_network.load_state_dict(q_network.state_dict())

# Set the target network to evaluation mode (no gradient computation)
target_network.eval()

# Create the optimizer for the Q-network
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Create the replay memory
memory = deque(maxlen=MEMORY_SIZE)

# Define a function to select an action using epsilon-greedy exploration
def select_action(state, epsilon):
    # state: the current state
    # epsilon: the exploration rate

    # Generate a random number
    rnd = random.random()

    # If the random number is less than epsilon, select a random action
    if rnd < epsilon:
        return torch.tensor([[random.randrange(output_size)]], device=device, dtype=torch.long)

    # Otherwise, select the action with the highest Q-value
    else:
        # Turn off gradient computation
        with torch.no_grad():
            # Return the action with the maximum Q-value
            return q_network(state).max(1)[1].view(1, 1)

# Define a function to optimize the Q-network
def optimize_model():
    # Check if the memory is large enough for a minibatch
    if len(memory) < BATCH_SIZE:
        return

    # Sample a minibatch of transitions from the memory
    transitions = random.sample(memory, BATCH_SIZE)

    # Transpose the minibatch to get the states, actions, rewards, next_states, and dones
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    # Convert the batches to tensors
    state_batch = torch.cat(state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(reward_batch)
    next_state_batch = torch.cat(next_state_batch)
    done_batch = torch.cat(done_batch)

    # Compute the Q-values for the current states and actions
    q_values = q_network(state_batch).gather(1, action_batch)

    # Compute the expected Q-values for the next states using the target network
    next_q_values = target_network(next_state_batch).max(1)[0].detach()

    # Compute the expected Q-values for the current states and actions
    expected_q_values = (next_q_values * GAMMA * (1 - done_batch)) + reward_batch

    # Compute the loss (mean squared error)
    loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

    # Zero the gradients
    optimizer.zero_grad()

    # Backpropagate the loss
    loss.backward()

    # Clip the gradients to avoid exploding gradients
    for param in q_network.parameters():
        param.grad.data.clamp_(-1, 1)

    # Update the network parameters
    optimizer.step()

# Initialize the episode counter
i_episode = 0

# Loop until the environment is solved
while True:
    # Increment the episode counter
    i_episode += 1

    # Reset the environment and get the initial state
    state = env.reset()

    # Convert the state to a tensor
    state = torch.tensor([state], device=device, dtype=torch.float)

    # Initialize the total reward
    total_reward = 0

    # Loop until the episode is done
    for t in count():
        # Calculate the exploration rate
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)

        # Select an action
        action = select_action(state, epsilon)

        # Execute the action and get the next state, reward, and done
        next_state, reward, done, _ = env.step(action.item())

        # Convert the next state, reward, and done to tensors
        next_state = torch.tensor([next_state], device=device, dtype=torch.float)
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        done = torch.tensor([done], device=device, dtype=torch.uint8)

        # Store the transition in the memory
        memory.append((state, action, reward, next_state, done))

        # Update the state
        state = next_state

        # Update the total reward
        total_reward += reward.item()

        # Optimize the Q-network
        optimize_model()

        # Check if the episode is done
        if done:
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Print the episode and the total reward
    print(f"Episode {i_episode}, Total reward {total_reward}")

    # Check if the environment is solved
    if total_reward >= env.spec.reward_threshold:
        print(f"Solved in {i_episode} episodes!")
        break
