from collections import namedtuple

import torch.optim as optim
from tqdm import tqdm

from Loss_Functions import compute_v_loss, ppo_policy_loss
from Neural_Network import Actor, Critic
from Returns_Advantages_Function import compute_return_advantage
from Trajectory_Collector import trajectory_collector

Rewards = namedtuple('Reward', ['reward'])


def ppo_clip(device, env, num_episodes, epsilon, alpha_policy, alpha_value, num_epochs):
    input_size = 1
    output_size = env.action_space.n

    frames = 0

    ppo_policy_network = Actor(input_size, output_size).to(device)
    ppo_value_network = Critic(input_size).to(device)

    optimizer_policy = optim.Adam(ppo_policy_network.parameters(), lr=alpha_policy)
    optimizer_value = optim.Adam(ppo_value_network.parameters(), lr=alpha_value)

    reward_tracker = []

    for episode in tqdm(range(num_episodes)):

        states, actions, rewards, old_action_probs, frame_plus = trajectory_collector(env, output_size,
                                                                                      ppo_policy_network, device)
        frames += frame_plus
        returns, advantages = compute_return_advantage(device, rewards, states, ppo_value_network)

        # Update the value network
        optimizer_value.zero_grad()
        for _ in range(num_epochs):
            loss = compute_v_loss(device, states, returns, ppo_value_network)
            loss.backward()

        optimizer_value.step()

        # Update the policy network
        optimizer_policy.zero_grad()

        for _ in range(num_epochs):
            policy_loss = ppo_policy_loss(states, actions, returns, ppo_policy_network, ppo_value_network, epsilon,
                                          old_action_probs, device)
            policy_loss.backward()

        optimizer_policy.step()
        reward_tracker.append(Rewards(reward=sum(rewards)))
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode + 1}\n'
              f'Reward: {sum(rewards)}\n'
              f'Frames: {frames}\n'
              f'{"~" * 40}\n')

    return ppo_policy_network, ppo_value_network, reward_tracker
