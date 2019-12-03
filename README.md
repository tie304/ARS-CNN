# ARS-CNN
Augmented Random Search Algorithm Framework. Supports OpenAI gym and custom enviroments. Built around the work of [Simple random search provides a competitive approach
to reinforcement learning](https://arxiv.org/pdf/1803.07055.pdf) this framework allows for the use of CNN's for feature inputs to the ARS algorithm. It also extends use by allowing user to easily specify their own custom enviroment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine or on the cloud. Pytorch CPU is installed by default if you want to use CUDA please see [Pytorch](https://pytorch.org/) getting started for your enviroment.
Also note that all gym enviroments are not installed by default. Please see respective installation guides.

### Install Dependencies:

```
pip3 install -r requirements.txt
```

## Training:
To train your model on an enviroment first configure parameters.py. If you want to use gym enviroments specify env_name. Run with --train flag. During training results are stored in the ./results directory keeping track of basic metrics such as avg reward evaluation, episode number, total steps, unix timestamp

```
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
```


```
python3 main.py --train
```

## Observing:
To evaluate your model on an enviroment run with --evaluate n flag. This will run n episodes of your model.

Run 5 episodes
```
 python3 main.py --evaluate 5
```
