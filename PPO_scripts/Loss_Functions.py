import torch
import torch.nn as nn


def ppo_policy_loss(states, actions, returns, policy_network, value_network, epsilon, old_action_probs):
    policy_loss_list = []
    for t in range(len(states)):
        state_tensor = torch.FloatTensor(states[t])
        action_tensor = torch.LongTensor([actions[t]])
        advantage = returns[t] - value_network(state_tensor).item()

        old_action_probs = torch.tensor(old_action_probs)
        old_log_prob = torch.log(old_action_probs.squeeze(0)[action_tensor])

        action_probs = policy_network(state_tensor)
        log_prob = torch.log(action_probs.squeeze(0)[action_tensor])

        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

        policy_loss = -torch.min(surr1, surr2)
        policy_loss_list.append(policy_loss)
    policy_loss = sum(policy_loss_list) / len(policy_loss_list)

    return policy_loss


def compute_v_loss(states, returns, value_network):
    loss_list = []
    value_loss = nn.MSELoss()

    for t in range(len(states)):
        state_tensor = torch.FloatTensor(states[t])

        target_value = torch.FloatTensor([returns[t]])
        predicted_value = value_network(state_tensor)

        loss = value_loss(predicted_value, target_value)
        loss_list.append(loss)
    value_loss = sum(loss_list) / len(loss_list)

    return value_loss
