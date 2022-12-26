import os
import gym
import numpy as np
import torch.multiprocessing as mp

from hyperopt import hp, fmin, tpe, space_eval

from tune_utils import objective


def main():
    """Main driver function."""
    # download the ROMS if we don't have them already
    if not os.path.exists('./HC ROMS.zip') or \
            not os.path.exists('./ROMS.zip'):
        os.system('rm -rf *.zip *.rar ./data/*.pth')
        os.system(
            'wget http://www.atarimania.com/roms/Roms.rar && unrar x Roms.rar'
        )
    # feed them to stella
    os.system('python3 -m atari_py.import_roms .')

    # empty the results file
    with open('./results.txt', 'w') as _:
        pass

    # create an environment just to get the observation dimensions
    env = gym.make('MontezumaRevenge-v0')
    input_size = env.observation_space.shape
    output_size = env.action_space.n
    env.close()

    # define the search space here
    search_space = {
        'input_size': input_size,
        'output_size': output_size,
        'gae_lambda': hp.uniform('gae_lambda', 0.85, 0.99),
        'num_workers': 16,
        'train_iterations': 1_000,
        'num_steps': 1_000,
        'ensemble_k': hp.randint('ensemble_k', 2, 3),
        'reward_reducer_method': hp.choice('reward_reducer_method', [
            'min', 'max', 'sum', 'mean', 'normalization_mean'
        ]),
        'policy_clip': hp.uniform('policy_clip', 0.05, 0.4),
        'epochs': 4,
        'batch_size': 8,
        'learning_rate': hp.loguniform('learning_rate',
                                       np.log(0.1), np.log(0.4)),
        'entropy_coefficient': hp.loguniform('entropy_coefficient',
                                             np.log(0.2), np.log(0.4)),
        'extrinsic_gamma': hp.uniform('extrinsic_gamma', 0.9, 0.9999),
        'intrinsic_gamma': hp.uniform('intrinsic_gamma', 0.9, 0.999),
        'clip_grad_norm': hp.uniform('clip_grad_norm', 0.4, 1),
        'extrinsic_coefficient': hp.choice('extrinsic_coefficient',
                                           [0.5, 1, 2]),
        'intrinsic_coefficient': hp.choice('intrinsic_coefficient',
                                           [0.5, 1, 2]),
        'sticky_actions': True,
        'sticky_actions_probability': 0.25,
        'pre_observations_normalization_steps':
            hp.choice('pre_observations_normalization_steps', [50, 100]),
        'seed': hp.choice('seed', [3, 13, 69, 420, 3407, 80085])
    }

    # tune
    best = fmin(
        objective,
        search_space,
        algo=tpe.suggest,
        max_evals=3
    )

    # print some results
    print(f'\nBest hyperparameters found are: {best}')
    print(f'Which correspond to: {space_eval(search_space, best)}\n')

    # remove downloaded files from current directory
    #os.system('rm -rf *.zip *.rar')


if __name__ == '__main__':
    print()
    main()
