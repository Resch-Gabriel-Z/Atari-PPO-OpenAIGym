import pandas as pd
import torch

from Atari_Preprocessing import environment_maker
from Hyperparameters import hyperparameters
from PPO import ppo_clip


def save_final_model(path, name, model):
    """
    A simple method to save the parameters after we trained the model
    Args:
        path: the path to save the final model
        name: the name of the final model.
        model: the model itself
    """
    torch.save(model.state_dict(), f'{path}/{name}.pt')


# get device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.get_device_name(device))

ppo_hyperparameters = [hyperparameters['number_of_episodes'], hyperparameters['epsilon'], hyperparameters['lr_act'],
                       hyperparameters['lr_crit'], hyperparameters['num_epochs']]

# Create the environment
env = environment_maker('-')

# Create the meta data
game_name = '-'
path_to_final_model = '-'
path_to_media = '-'

ppo_policy_network, ppo_value_network, reward_tracker = ppo_clip(device, env, *ppo_hyperparameters)

# After training, save the models parameters
name_final_model = game_name + '_final'
save_final_model(name=name_final_model, path=path_to_final_model, model=ppo_policy_network)
df = pd.DataFrame({'cumulative rewards': reward_tracker})
df.to_csv(f'{path_to_media}{game_name}.csv')
