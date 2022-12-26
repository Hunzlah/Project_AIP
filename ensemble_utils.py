import numpy as np

from abc import ABC, abstractmethod

from utils import RunningMeanStd


class RewardReducer(ABC):
    """Abstract Base Class for Reward Reducing in a RND Ensemble."""

    def __init__(self):
        pass

    @abstractmethod
    def reduce(self, reward):
        pass


class MinRewardReducer(RewardReducer):
    """Reduces a family of rewards by taking the minimum of the values along the
        second dimension."""

    def __init__(self):
        super(MinRewardReducer, self).__init__()

    def reduce(self, reward):
        """Given a reward of shape [Batch, Ensemble_K], it takes the minimum for
            each batch item, retuning an array of shape [Batch,]."""
        return reward.min(axis=1)


class MaxRewardReducer(RewardReducer):
    """Reduces a family of rewards by taking the maximum of the values along the
        second dimension."""

    def __init__(self):
        super(MaxRewardReducer, self).__init__()

    def reduce(self, reward):
        """Given a reward of shape [Batch, Ensemble_K], it takes the maximum for
            each batch item, retuning an array of shape [Batch,]."""
        return reward.max(axis=1)


class SumRewardReducer(RewardReducer):
    """Reduces a family of rewards by summing up the values along the
        second dimension."""

    def __init__(self):
        super(SumRewardReducer, self).__init__()

    def reduce(self, reward):
        """Given a reward of shape [Batch, Ensemble_K], it sums the elements for
            each batch item, retuning an array of shape [Batch,]."""
        return reward.sum(axis=1)


class MeanRewardReducer(RewardReducer):
    """Reduces a family of rewards by taking the average of the values
        along the second dimension."""

    def __init__(self):
        super(MeanRewardReducer, self).__init__()

    def reduce(self, reward):
        """Given a reward of shape [Batch, Ensemble_K], it averages the elements
            for each batch item, retuning an array of shape [Batch,]."""
        return reward.mean(axis=1)


class MeanNormalizerRewardReducer(RewardReducer):
    """Reduces a family of rewards by taking the average of the values along the
        second dimension."""

    def __init__(self, shape):
        super(MeanNormalizerRewardReducer, self).__init__()
        self.rms = RunningMeanStd(shape=shape)

    def reduce(self, reward):
        """Given a reward array of shape [Batch, Ensemble_K], it normalized
            (with running statistics) and then averages the elements for each
            batch item, retuning an array of shape [Batch,]."""
        self.rms.update(reward)
        normalized_rewards = (reward - self.rms.mean) / np.sqrt(self.rms.var)
        return normalized_rewards.mean(axis=1)


def get_reward_reducer_from_str(reward_reducer_as_str, ensemble_k):
    """Returns a RewardReducer that corresponds to the given string."""
    if reward_reducer_as_str == 'min':
        return MinRewardReducer()
    elif reward_reducer_as_str == 'max':
        return MaxRewardReducer()
    elif reward_reducer_as_str == 'sum':
        return SumRewardReducer()
    elif reward_reducer_as_str == 'mean':
        return MeanRewardReducer()
    elif reward_reducer_as_str == 'normalization_mean':
        return MeanNormalizerRewardReducer(shape=(ensemble_k,))
    else:
        raise ValueError(f'Unknown reward reducer: {reward_reducer_as_str}.')
