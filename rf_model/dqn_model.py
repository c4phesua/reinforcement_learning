from rf_model.baseModel import BaseModel
from addon.epsilon_greedy import EpsilonGreedy
from addon.experience_replay import ExperienceReplay
from tensorflow.keras import Sequential
import numpy as np


class DQN(BaseModel):
    def __init__(self, discount_factor: float, epsilon: float, e_min: int, e_max: int):
        super().__init__(discount_factor, epsilon, e_min, e_max)
        self.gamma = discount_factor
        self.epsilon_greedy = EpsilonGreedy(epsilon)
        self.e_min = e_min
        self.exp_replay = ExperienceReplay(e_max)
        self.training_network = Sequential()
        self.target_network = Sequential()
        self.cache = list()

    def observe(self, state, action_space: list = None):
        q_value = self.training_network.predict(np.array([state])).ravel()
        if action_space is not None:
            return max([[q_value[a], a] for a in action_space], key=lambda x: x[0])[1]
        return np.argmax(q_value)

    def observe_on_training(self, state, action_space: list = None) -> int:
        q_value = self.training_network.predict(np.array([state])).ravel()
        action = self.epsilon_greedy.perform(q_value, action_space)
        self.cache.extend([state, action])
        return action

    def take_reward(self, reward, next_state, done):
        self.cache.extend([reward, next_state, done])
        self.exp_replay.add_experience(self.cache.copy())
        self.cache.clear()

    def train_network(self, sample_size: int, batch_size: int, epochs: int, verbose: int = 2, cer_mode: bool = False):
        if self.exp_replay.get_size() >= self.e_min:
            # state_samples, action_samples, reward_samples, next_state_samples, done_samples
            s_batch, a_batch, r_batch, ns_batch, done_batch = self.exp_replay.sample_experience(sample_size, cer_mode)
            states, q_values = self.replay(s_batch, a_batch, r_batch, ns_batch, done_batch)
            history = self.training_network.fit(states, q_values, epochs=epochs, batch_size=batch_size, verbose=verbose)
            return history.history['loss']

    def replay(self, states, actions, rewards, next_states, terminals):
        q_values = self.target_network.predict(np.array(states))  # get q value at state t by target network
        nq_values = self.target_network.predict(np.array(next_states))  # get q value at state t+1 by target network
        for i in range(len(states)):
            a = actions[i]
            done = terminals[i]
            r = rewards[i]
            if done:
                q_values[i][a] = r
            else:
                q_values[i][a] = r + self.gamma * np.max(nq_values[i])
        return states, q_values

    def update_target_network(self):
        self.target_network.set_weights(self.training_network.get_weights())
