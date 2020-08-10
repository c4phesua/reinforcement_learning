from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, discount_factor: float, epsilon: float, e_min: int, e_max: int):
        """
        self.qNetwork and self.qTargetNetwork need to be installed.

        :param discount_factor: It’s used to balance immediate and future reward.
        Typically this value can range anywhere from 0.8 to 0.99.

        :param epsilon: the probability of choosing to explore action.
        :param e_min: Minimum amount of experience to start training.
        :param e_max: Maximum amount of experience.
        """
        pass

    @abstractmethod
    def observe(self, state, action_space: list = None):
        """
        Observe state from environment and return a action

        :param state: Current situation returned by the environment.
        :param action_space: All the possible moves that the agent can take.
        :return: Action that have the max value from q table value.

        Note: If actionSpace is None then all action are possible
        """
        pass

    @abstractmethod
    def observe_on_training(self, state, action_space: list = None) -> int:
        """
        Observe state from environment and return a action by epsilon greedy policy.
        The state and action will be stored in the memory buffer to be used for Experience Replay

        :param state: Current situation returned by the environment.
        :param action_space: All the possible moves that the agent can take.
        :return: Action that have the max value from q table value.
        Note: If actionSpace is None then all action are possible
        """
        pass

    @abstractmethod
    def take_reward(self, reward, next_state, done):
        """
        After used observeOnTraining method, environment will return reward, nextState and done information
        we will use this method to put that information into the Experience Replay

        :param reward: immediate reward returned by the environment
        :param next_state: Next situation returned by the environment.
        :param done: describes whether the environment situation has terminated or not
        :return: None
        """
        pass

    @abstractmethod
    def train_network(self, sample_size: int, batch_size: int, epochs: int, verbose: int = 2, cer_mode: bool = False):
        """
        :param sample_size: number of samples taken from Experience Replay.
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified,
        `batch_size` will default to 32. Do not specify the `batch_size` if your data is in the form of datasets,
        generators, or `keras.utils.Sequence` instances (since they generate batches).
        :param epochs: Integer. Number of epochs to train the model.
        An epoch is an iteration over the entire `x` and `y` data provided.
        Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch".
        The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index
        `epochs` is reached.
        :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended
        when not running interactively (eg, in a production environment).
        :param cer_mode: Turn on or off cer (Combined Experience Replay). Default is False.
        :return: None
        """
        pass

    @abstractmethod
    def update_target_network(self):
        """
        Update Q target Network by weight of Q network
        :return: None
        """
        pass
