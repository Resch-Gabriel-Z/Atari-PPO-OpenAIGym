import torch.optim as optim
from collections import namedtuple
from Atari_Preprocessing import environment_maker
from Neural_Network import Actor, Critic
from Returns_Advantages_Function import compute_return_advantage
from Trajectory_Collector import trajectory_collector
from Loss_Functions import compute_v_loss, ppo_policy_loss
from Hyperparameters import hyperparameters

Rewards = namedtuple('Reward', ['reward'])

def ppo_clip(env, num_episodes, epsilon, alpha_policy, alpha_value, num_epochs):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    ppo_policy_network = Actor(input_size, output_size)
    ppo_value_network = Critic(input_size)

    optimizer_policy = optim.Adam(ppo_policy_network.parameters(), lr=alpha_policy)
    optimizer_value = optim.Adam(ppo_value_network.parameters(), lr=alpha_value)

    reward_tracker = []

    for episode in range(num_episodes):

        states, actions, rewards, old_action_probs = trajectory_collector(env, output_size, ppo_policy_network)

        returns, advantages = compute_return_advantage(rewards, states, ppo_value_network)

        # Update the value network
        optimizer_value.zero_grad()
        for _ in range(num_epochs):
            loss = compute_v_loss(states, returns, ppo_value_network)
            loss.backward()

        optimizer_value.step()

        # Update the policy network
        optimizer_policy.zero_grad()

        for _ in range(num_epochs):
            policy_loss = ppo_policy_loss(states, actions, returns, ppo_policy_network, ppo_value_network, epsilon,
                                          old_action_probs)
            policy_loss.backward()

        optimizer_policy.step()
        reward_tracker.append(Rewards(reward=sum(rewards)))
        print(episode, sum(rewards))

    return ppo_policy_network, ppo_value_network, reward_tracker