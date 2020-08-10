import numpy as np
import random


class ExperienceReplay:
    def __init__(self, e_max: int):
        if e_max <= 0:
            raise ValueError('Invalid value for memory size')
        self.e_max = e_max
        self.memory = list()
        self.index = 0

    def add_experience(self, sample: list):
        if len(sample) != 5:
            raise Exception('Invalid sample')
        if len(self.memory) < self.e_max:
            self.memory.append(sample)
        else:
            self.memory[self.index] = sample
        self.index = (self.index + 1) % self.e_max

    def sample_experience(self, sample_size: int, cer_mode: bool):
        samples = random.sample(self.memory, sample_size)
        if cer_mode:
            samples[-1] = self.memory[self.index - 1]
        # state_samples, action_samples, reward_samples, next_state_samples, done_samples
        s_batch, a_batch, r_batch, ns_batch, done_batch = map(np.array, zip(*samples))
        return s_batch, a_batch, r_batch, ns_batch, done_batch

    def get_size(self):
        return len(self.memory)
