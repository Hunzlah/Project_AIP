import cv2
import gym
import numpy as np
import torch.multiprocessing as mp

from collections import deque


class MontezumasRevengeEnvironment(mp.Process):
    """Montezumas Revenge environment."""

    def __init__(
            self,
            env_idx,
            child_conn,
            history_size=4,
            height=84,
            width=84,
            sticky_action=True,
            stick_action_prob=0.25,
            seed=0
    ):
        super(MontezumasRevengeEnvironment, self).__init__()

        # define the environment here
        self.env = gym.make('MontezumaRevenge-v0')
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        # define some important variables
        self.daemon = True
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.episode_reward = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.stick_action_prob = stick_action_prob

        self.history_size = history_size
        self.history = np.zeros([history_size, height, width])
        self.height = height
        self.width = width

        self.reset()

    def run(self):
        """Invoked by the mp.Process.start() method."""

        # Run the environment until the action received is -1
        while True:

            action = self.child_conn.recv()
            if action == -1:
                self.env.close()
                return

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.stick_action_prob:
                    action = self.last_action

            # store the last action taken by the agent
            self.last_action = action

            # step in the environment
            obs, reward, done, info = self.env.step(action)
            self.rall += reward
            self.episode_reward += reward
            temp_episode_reward = self.episode_reward

            # update history frame
            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.preprocess(obs)
            # update number of steps performed
            self.steps += 1

            # if the episode is finished, reset the env
            if done:
                self.recent_rlist.append(self.rall)
                self.history = self.reset()

            # send the new data to the parent process
            next_obs = self.history[:, :, :]
            self.child_conn.send([next_obs, reward, done, temp_episode_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.episode_reward = 0
        self.rall = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def preprocess(self, x):
        # grayscaling and rezising
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (self.height, self.width))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.preprocess(s)
