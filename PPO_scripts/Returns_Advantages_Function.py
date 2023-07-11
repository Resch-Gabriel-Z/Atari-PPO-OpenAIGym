import torch


def compute_return_advantage(device,rewards, states, value_network):
    returns = []
    advantages = []
    next_value = 0

    for t in reversed(range(len(rewards))):
        next_value = rewards[t] + next_value
        returns.insert(0, next_value)

        state_tensor = torch.FloatTensor(states[t]).unsqueeze(0).to(device)
        value = value_network(state_tensor).item()
        advantages.insert(0, returns[0] - value)

    return returns, advantages
