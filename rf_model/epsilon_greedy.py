import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def perform(self, q_value, action_space: list = None):
        prob = np.random.sample()  # get probability of taking random action
        if prob <= self.epsilon:  # take random action
            if action_space is None:  # all action are available
                return np.random.randint(len(q_value))
            return np.random.choice(action_space)
        else:  # take greedy action
            if action_space is None:
                return np.argmax(q_value)
            return max([[q_value[a], a] for a in action_space], key=lambda x: x[0])[1]

    def decay(self, decay_value, lower_bound):
        """
        Adjust the epsilon value by the formula: epsilon = max(decayValue * epsilon, lowerBound).
        :param decay_value: Value ratio adjustment (0, 1).
        :param lower_bound: Minimum epsilon value.
        :return: None
        """
        self.epsilon = max(self.epsilon * decay_value, lower_bound)
