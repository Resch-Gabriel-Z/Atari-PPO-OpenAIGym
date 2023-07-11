import numpy as np
import torch

from Hyperparameters import hyperparameters


def trajectory_collector(env, output_size, policy_network, device):
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    steps = 0
    while not done and steps < hyperparameters['max_steps_per_trajectory']:
        states.append(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device=device)

        action_probs = policy_network(state_tensor)
        action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(output_size, p=action_probs.flatten())
        steps += 1
        next_state, reward, done, *others = env.step(action)

        actions.append(action)
        rewards.append(reward)

        state = next_state
    return states, actions, rewards, action_probs
