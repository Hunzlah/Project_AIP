import torch
import random
import numpy as np
import torch.multiprocessing as mp

from rnd_agent import RNDAgent
from environment import MontezumasRevengeEnvironment
from ensemble_utils import get_reward_reducer_from_str
from utils import RunningMeanStd, RewardForwardFilter, make_train_data


global_num_evals = 0
global_best_args = None
global_best_max_reward_in_one_episode = float('-inf')


def train_agent(agent, args):
    """Trains an agent in the given environment and returns the maximum reward
        it got in a single episode."""
    # variable to be returned: Maximum reward the agent got within 1 episode
    max_reward_in_one_episode = float('-inf')

    # running statistics
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args['intrinsic_gamma'])

    # parallel environments
    envs = []
    parent_connections = []
    child_connections = []
    for idx in range(args['num_workers']):
        parent_conn, child_conn = mp.Pipe()
        parent_connections.append(parent_conn)
        child_connections.append(child_conn)
        worker = MontezumasRevengeEnvironment(
            env_idx=idx,
            child_conn=child_conn,
            history_size=4,
            height=84,
            width=84,
            sticky_action=args['sticky_actions'],
            stick_action_prob=args['sticky_actions_probability'],
            seed=args['seed']
        )
        worker.start()
        envs.append(worker)
    states = np.zeros([args['num_workers'], 4, 84, 84])

    # pre-observation normalization
    total_steps = args['num_steps'] * \
        args['pre_observations_normalization_steps']
    update_frequency = args['num_steps'] * args['num_workers']
    next_observations = []
    for step in range(total_steps):
        # sample random actions (one for each environment)
        actions = np.random.randint(
            0,
            args['output_size'],
            size=(args['num_workers'],)
        )
        # play it
        for parent_conn, action in zip(parent_connections, actions):
            parent_conn.send(action)
        # receive step information
        for parent_conn in parent_connections:
            next_state, _, _, _ = parent_conn.recv()
            next_observations.append(next_state[3, :, :].reshape([1, 84, 84]))
        # update running mean std
        if len(next_observations) % update_frequency == 0:
            obs_rms.update(np.stack(next_observations))
            next_observations = []

    # main training loop
    global_step = 0
    for iteration in range(args['train_iterations']):
        print(f'Iteration {iteration + 1}/{args["train_iterations"]}.')

        # iteration data
        total_state, total_reward, total_done,  = [], [], []
        total_next_state, total_action = [], []
        total_int_reward, total_next_obs, total_ext_values = [], [], []
        total_policy_np, total_int_values, total_policy = [], [], []
        global_step += (args['num_workers'] * args['num_steps'])

        # Step 1. n-step rollout
        for _ in range(args['num_steps']):

            # choose an action for each environment and send it to the
            # corresponding worker
            _obs = np.float32(states) / 255
            actions, value_ext, value_int, policy = agent.get_action(_obs)
            for parent_conn, action in zip(parent_connections, actions):
                parent_conn.send(action)

            # for each worker, receive the responses
            next_states, rewards, dones, next_obs = [], [], [], []
            for parent_conn in parent_connections:
                next_state, reward, done, episode_reward = parent_conn.recv()
                max_reward_in_one_episode = max(max_reward_in_one_episode,
                                                episode_reward)
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
                agent.compute_intrinsic_reward(clipped_normalized_obs)
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
        _, value_ext, value_int, _ = agent.get_action(
            np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        # reshape the data
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]). \
            reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]). \
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
        mean = np.mean(total_reward_per_env)
        std = np.std(total_reward_per_env)
        count = len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)
        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(
            total_reward,
            total_done,
            total_ext_values,
            args['extrinsic_gamma'],
            args['num_steps'],
            args['num_workers'],
            True,
            args['gae_lambda']
        )

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(
            total_int_reward,
            np.zeros_like(total_int_reward),
            total_int_values,
            args['intrinsic_gamma'],
            args['num_steps'],
            args['num_workers'],
            True,
            args['gae_lambda']
        )

        # add ext adv and int adv
        total_adv = int_adv * args['intrinsic_coefficient'] + \
            ext_adv * args['extrinsic_coefficient']

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)

        # Step 5. Training
        _obs = np.float32(total_state) / 255
        _next_obs = ((total_next_obs - obs_rms.mean) /
                     np.sqrt(obs_rms.var)).clip(-5, 5)
        agent.train_model(
            s_batch=_obs,
            target_ext_batch=ext_target,
            target_int_batch=int_target,
            y_batch=total_action,
            adv_batch=total_adv,
            next_obs_batch=_next_obs,
            old_policy=total_policy
        )

    # kill the parallel envs
    for parent_conn in parent_connections:
        parent_conn.send(-1)
    [worker.join() for worker in envs]

    # return the maximum extrinsic reward the agent got in a single episode
    return max_reward_in_one_episode


def objective(args):
    """Trains an agent and returns the total number of times he
        completed the maze."""
    # fix random seed
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    random.seed(args['seed'])

    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # agent
    effective_batch_size = int(args['num_steps'] * args['num_workers'] /
                               args['batch_size'])
    reward_reducer = get_reward_reducer_from_str(args['reward_reducer_method'],
                                                 args['ensemble_k'])
    rnd_agent = RNDAgent(
        input_size=args['input_size'],
        output_size=args['output_size'],
        num_env=args['num_workers'],
        num_step=args['num_steps'],
        gamma=args['extrinsic_gamma'],
        device=device,
        ensemble_k=args['ensemble_k'],
        reward_reducer=reward_reducer,
        lam=args['gae_lambda'],
        learning_rate=args['learning_rate'],
        ent_coef=args['entropy_coefficient'],
        clip_grad_norm=args['clip_grad_norm'],
        epoch=args['epochs'],
        batch_size=effective_batch_size,
        ppo_eps=args['policy_clip'],
        use_gae=True
    )

    # return the negative of the max reward for one episode
    max_reward_in_one_episode = train_agent(rnd_agent, args)

    # compare it to the global (over all hyperopt evaluations)
    global global_best_max_reward_in_one_episode
    global global_best_args
    if max_reward_in_one_episode > global_best_max_reward_in_one_episode:
        global_best_max_reward_in_one_episode = max_reward_in_one_episode
        global_best_args = args

    # write the results in a file
    global global_num_evals
    with open('./results.txt', 'a') as fp:
        fp.write(f'Evaluation: {global_num_evals}\n\n')
        fp.write(f'Max reward achieved in one episode: '
                 f'{max_reward_in_one_episode}.\n')
        fp.write(f'{args}\n')
        fp.write('\n')
        fp.write(f'Max reward achieved in one episode, over all evaluations: '
                 f'{global_best_max_reward_in_one_episode}\n')
        fp.write(f'{global_best_args}\n')
        fp.write('\n\n\n')
    global_num_evals += 1

    # return the negative value because we are trying to minimize
    return -max_reward_in_one_episode
