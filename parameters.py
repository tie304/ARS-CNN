# Setting the Hyper Parameters
import os


class Hp:
    def __init__(self):
        self.nb_episodes = 10000
        self.episode_length = 10000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'MountainCarContinuous-v0'
        self.conv_input = False
        self.render_eval = False
        self.save_freq = 1
        self.results_save_dir = os.path.join("results", f"results-{self.env_name}.npy")
        self.weights_save_dir = os.path.join("weights", f"weights-{self.env_name}.npy")




