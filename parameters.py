# Setting the Hyper Parameters
import os
import torch


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
        self.env_name = 'CarRacing-v0'
        self.conv_input = True # uses convolutions to extract from images
        if self.conv_input:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu or cup
            print('using CNN, device is: ',self.device)
        self.render_eval = True # renders during evaluation (gym env only)
        self.render_train = False # renders during training (gym env only)
        self.save_freq = 1 # episodes per save time
        self.train_from_previous_weights = False
        self.results_save_dir = os.path.join("results", f"results-{self.env_name}.npy")
        self.weights_save_dir = os.path.join("weights", f"weights-{self.env_name}.npy")
        self.normalizer_weights_save = os.path.join("weights", f"normalizer-{self.env_name}.npy")




