import sys
import numpy as np

# Building the AI


class Policy:

    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + self.hp.noise * delta).dot(input)
        else:
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.hp.learning_rate / (self.hp.nb_best_directions * sigma_r) * step

    def save(self):
        np.save(self.hp.weights_save_dir, self.theta)

    def load(self):
        self.theta = np.load(self.hp.weights_save_dir)





