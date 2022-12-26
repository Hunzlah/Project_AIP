"""
Install the atari-py package using the command:
    $ pip install atari-py
"""
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from tqdm import tqdm

from rnd_agent import RNDAgent
from environment import MontezumasRevengeEnvironment
from ensemble_utils import MeanRewardReducer
from utils import RunningMeanStd, RewardForwardFilter, make_train_data, \
    preprocess


# download the ROMS if we don't have them already
if not os.path.exists('./data/HC ROMS.zip') or \
        not os.path.exists('./data/ROMS.zip'):
    os.system('rm -rf *.zip *.rar ./data/*.pth')
    os.system(
        'wget http://www.atarimania.com/roms/Roms.rar && unrar x Roms.rar'
    )

# feed them to stella
os.system('python3 -m atari_py.import_roms .')

# now we can see how it works
env = gym.make('MontezumaRevenge-v0')
env.seed(13)
env.action_space.seed(13)
done = True
for step in range(25):
    if done:
        env.reset()
    _, _, done, _ = env.step(env.action_space.sample())

input_size = env.observation_space.shape
output_size = env.action_space.n
print(f'Environment Observation Space:   {input_size}')
print(f'Environment Action Space:        {output_size}')

env.close()

# define the hyperparameters here
rescale_width = 84
rescale_height = 84
history_size = 4
train_iterations = 25000
device_type = 'cuda'  # change between 'cpu' and 'cuda'
use_gae = True  # whether to use generalized advantage estimation
gae_lambda = 0.95
num_workers = 32
num_steps = 250
ensemble_k = 3  # number of (target, predictor) networks to use for the ensemble
reward_reducer = MeanRewardReducer()  # method used to combine intrinsic rewards
policy_clip = 0.2
epochs = 4
minibatch_size = 4
learning_rate = 1e-4
entropy_coefficient = 1e-3
extrinsic_gamma = 0.999
intrinsic_gamma = 0.99
clip_grad_norm = 0.5
extrinsic_coefficient = 2.0
intrinsic_coefficient = 1.0
sticky_actions = True
sticky_action_probability = 0.25
pre_observations_normalization_steps = 20
save_models = True
seed = 13

hyperparameters = {
    'input_size': input_size,
    'output_size': output_size,
    'width': rescale_width,
    'height': rescale_height,
    'history_size': history_size,
    'train_iterations': train_iterations,
    'device': torch.device(device_type),
    'use_gae': use_gae,
    'gae_lambda': gae_lambda,
    'num_workers': num_workers,
    'num_steps': num_steps,
    'ensemble_k': ensemble_k,
    'reward_reducer': reward_reducer,
    'policy_clip': policy_clip,
    'epochs': epochs,
    'effective_batch_size': int(num_steps * num_workers / minibatch_size),
    'learning_rate': learning_rate,
    'entropy_coefficient': entropy_coefficient,
    'extrinsic_gamma': extrinsic_gamma,
    'intrinsic_gamma': intrinsic_gamma,
    'clip_grad_norm': clip_grad_norm,
    'extrinsic_coefficient': extrinsic_coefficient,
    'intrinsic_coefficient': intrinsic_coefficient,
    'sticky_actions': sticky_actions,
    'sticky_actions_probability': sticky_action_probability,
    'pre_observations_normalization_steps':
        pre_observations_normalization_steps,
    'model_save_path': './data/model',
    'rnd_predictor_save_path': './data/rnd_predictor',
    'rnd_target_save_path': './data/rnd_target',
    'save_models_after_training': save_models,
    'seed': seed
}

# running statistics
reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
discounted_reward = RewardForwardFilter(hyperparameters['intrinsic_gamma'])

# agent
rnd_agent = RNDAgent(
    input_size=hyperparameters['input_size'],
    output_size=hyperparameters['output_size'],
    num_env=hyperparameters['num_workers'],
    num_step=hyperparameters['num_steps'],
    gamma=hyperparameters['extrinsic_gamma'],
    device=hyperparameters['device'],
    ensemble_k=hyperparameters['ensemble_k'],
    reward_reducer=hyperparameters['reward_reducer'],
    lam=hyperparameters['gae_lambda'],
    learning_rate=hyperparameters['learning_rate'],
    ent_coef=hyperparameters['entropy_coefficient'],
    clip_grad_norm=hyperparameters['clip_grad_norm'],
    epoch=hyperparameters['epochs'],
    batch_size=hyperparameters['effective_batch_size'],
    ppo_eps=hyperparameters['policy_clip'],
    use_gae=True
)

# parallel environments
envs = []
parent_connections = []
child_connections = []
for idx in range(hyperparameters['num_workers']):
    parent_conn, child_conn = mp.Pipe()
    parent_connections.append(parent_conn)
    child_connections.append(child_conn)
    worker = MontezumasRevengeEnvironment(
        env_idx=idx,
        child_conn=child_conn,
        history_size=hyperparameters['history_size'],
        height=hyperparameters['height'],
        width=hyperparameters['width'],
        sticky_action=hyperparameters['sticky_actions'],
        stick_action_prob=hyperparameters['sticky_actions_probability'],
        seed=hyperparameters['seed']
    )
    worker.start()
    envs.append(worker)
states = np.zeros([hyperparameters['num_workers'], 4, 84, 84])

# pre-observation normalization
total_steps = hyperparameters['num_steps'] * \
              hyperparameters['pre_observations_normalization_steps']
update_frequency = hyperparameters['num_steps'] * hyperparameters['num_workers']
next_observations = []

for step in tqdm(range(total_steps)):
    # sample a random action for each environment
    actions = np.random.randint(
        0,
        output_size,
        size=(hyperparameters['num_workers'],)
    )

    # play it
    for parent_conn, action in zip(parent_connections, actions):
        parent_conn.send(action)

    # receive step information
    for parent_conn in parent_connections:
        next_state, reward, done, rewardTemp = parent_conn.recv()
        next_observations.append(next_state[3, :, :].reshape([1, 84, 84]))

    # update running mean std
    if len(next_observations) % update_frequency == 0:
        next_observations = np.stack(next_observations)
        obs_rms.update(next_observations)
        next_observations = []

# main training loop
global_step = 0
for iteration in range(hyperparameters['train_iterations']):
    print(f'Iteration {iteration + 1}/{hyperparameters["train_iterations"]}.')

    # iteration data
    total_state, total_reward, total_done, total_next_state, total_action = \
        [], [], [], [], []
    total_int_reward, total_next_obs, total_ext_values = [], [], []
    total_policy_np, total_int_values, total_policy = [], [], []

    global_step += (hyperparameters['num_workers'] *
                    hyperparameters['num_steps'])

    # Step 1. n-step rollout
    for _ in range(hyperparameters['num_steps']):

        # choose an action for each environment and send it to the
        # corresponding worker
        actions, value_ext, value_int, policy = rnd_agent.get_action(
            np.float32(states) / 255.
        )
        for parent_conn, action in zip(parent_connections, actions):
            parent_conn.send(action)

        # for each worker, receive the responses
        next_states, rewards, dones, next_obs = [], [], [], []
        for parent_conn in parent_connections:
            next_state, reward, done, rewardTemp = parent_conn.recv()
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            next_obs.append(next_state[3, :, :].reshape([1, 84, 84]))

        # stack the data
        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        next_obs = np.stack(next_obs)

        # total reward = intrinsic reward + extrinsic Reward
        clipped_normalized_obs = ((next_obs - obs_rms.mean) /
                                  np.sqrt(obs_rms.var)).clip(-5, 5)
        intrinsic_reward = np.hstack(
            rnd_agent.compute_intrinsic_reward(clipped_normalized_obs)
        )

        # append the data to the iteration data
        total_next_obs.append(next_obs)
        total_int_reward.append(intrinsic_reward)
        total_state.append(states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        total_policy.append(policy)
        total_policy_np.append(policy.cpu().numpy())

        # the next states become the current states
        states = next_states[:, :, :, :]

    # calculate last next value
    _, value_ext, value_int, _ = rnd_agent.get_action(np.float32(states) / 255.)
    total_ext_values.append(value_ext)
    total_int_values.append(value_int)

    # reshape the data
    total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).\
        reshape([-1, 4, 84, 84])
    total_reward = np.stack(total_reward).transpose().clip(-1, 1)
    total_action = np.stack(total_action).transpose().reshape([-1])
    total_done = np.stack(total_done).transpose()
    total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).\
        reshape([-1, 1, 84, 84])
    total_ext_values = np.stack(total_ext_values).transpose()
    total_int_values = np.stack(total_int_values).transpose()
    total_logging_policy = np.vstack(total_policy_np)

    # Step 2. calculate intrinsic reward
    # running mean intrinsic reward
    total_int_reward = np.stack(total_int_reward).transpose()
    total_reward_per_env = np.array(
        [
            discounted_reward.update(reward_per_step)
            for reward_per_step in total_int_reward.T
        ]
    )
    mean, std, count = \
        np.mean(total_reward_per_env), \
        np.std(total_reward_per_env), \
        len(total_reward_per_env)
    reward_rms.update_from_moments(mean, std ** 2, count)

    # normalize intrinsic reward
    total_int_reward /= np.sqrt(reward_rms.var)

    # Step 3. make target and advantage
    # extrinsic reward calculate
    ext_target, ext_adv = make_train_data(
        total_reward,
        total_done,
        total_ext_values,
        hyperparameters['extrinsic_gamma'],
        hyperparameters['num_steps'],
        hyperparameters['num_workers'],
        hyperparameters['use_gae'],
        hyperparameters['gae_lambda']
    )

    # intrinsic reward calculate
    # None Episodic
    int_target, int_adv = make_train_data(
        total_int_reward,
        np.zeros_like(total_int_reward),
        total_int_values,
        hyperparameters['intrinsic_gamma'],
        hyperparameters['num_steps'],
        hyperparameters['num_workers'],
        hyperparameters['use_gae'],
        hyperparameters['gae_lambda']
    )

    # add ext adv and int adv
    total_adv = int_adv * hyperparameters['intrinsic_coefficient'] + \
        ext_adv * hyperparameters['extrinsic_coefficient']

    # Step 4. update obs normalize param
    obs_rms.update(total_next_obs)

    # Step 5. Training
    rnd_agent.train_model(
        s_batch=np.float32(total_state) / 255.,
        target_ext_batch=ext_target,
        target_int_batch=int_target,
        y_batch=total_action,
        adv_batch=total_adv,
        next_obs_batch=((total_next_obs - obs_rms.mean) /
                        np.sqrt(obs_rms.var)).clip(-5, 5),
        old_policy=total_policy
    )

    print('Current Global Step: {}'.format(global_step))
    print()

# save models if specified
if hyperparameters['save_models_after_training']:
    print('\nSaving models...')
    torch.save(rnd_agent.model.state_dict(),
               f"./{hyperparameters['model_save_path']}.pth")
    for idx, predictor in enumerate(rnd_agent.rnd.predictors):
        torch.save(predictor.state_dict(),
                   f"./{hyperparameters['rnd_predictor_save_path']}_{idx}.pth")
    for idx, target in enumerate(rnd_agent.rnd.targets):
        torch.save(target.state_dict(),
                   f"./{hyperparameters['rnd_target_save_path']}_{idx}.pth")

# kill the parallel envs
for parent_conn in parent_connections:
    parent_conn.send(-1)
[worker.join() for worker in envs]

# create a new environment
env = gym.make('MontezumaRevenge-v0')
env.reset()
history = np.zeros([hyperparameters['history_size'], hyperparameters['height'],
                    hyperparameters['width']])

# During evaluation, we mostly care about the behavior of the intrinsic reward
rall = 0
done = False
extrinsic_reward_list = []
intrinsic_reward_list = []
max_steps = 70000

# loop until done or max steps exceeded
while not done and max_steps > 0:

    # compute action from agent, and send it to the environment
    actions, value_ext, value_int, policy = rnd_agent.get_action(
        np.float32(states) / 255.
    )
    obs, reward, done, info = env.step(actions[0])

    # reset environment if done
    if done:
        env.reset()

    # preprocess
    rewards, dones = [], []
    history[:3, :, :] = history[1:, :, :]
    history[3, :, :] = preprocess(obs, hyperparameters['height'],
                                  hyperparameters['width'])
    next_state = history[:, :, :]
    extrinsic_reward_list.append(reward)
    rall += reward
    next_states = next_state.reshape([1, 4, 84, 84])
    next_obs = next_state[3, :, :].reshape([1, 1, 84, 84])

    # compute intrinsic rewards
    intrinsic_reward = rnd_agent.compute_intrinsic_reward(next_obs)
    intrinsic_reward_list.append(intrinsic_reward)
    states = next_states[:, :, :, :]

    # decrement number of maximum steps
    max_steps -= 1

# close the environment when finished
env.close()

# print total extrinsic reward
print(f'Total extrinsic reward: {rall}')

# plot rewards
intrinsic_reward_list = (intrinsic_reward_list -
                         np.mean(intrinsic_reward_list)) / \
                        np.std(intrinsic_reward_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

ax1.plot(extrinsic_reward_list, label='Extrinsic Rewards', color='b')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Extrinsic Rewards')
ax1.set_title('Extrinsic Rewards per Step in one episode')
ax1.legend()

ax2.plot(intrinsic_reward_list, label='Intrinsic Rewards', color='b')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Intrinsic Rewards')
ax2.set_title('Intrinsic Rewards per Step in one episode')
ax2.legend()

plt.show()

# remove files from current directory
os.system('rm -rf *.zip *.rar')
