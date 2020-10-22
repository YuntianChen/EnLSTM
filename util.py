# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import json
import logging
import torch
import shutil


def save_list(filename, li):
    with open(filename, 'w') as f:
        for data in li:
            f.write(str(data) + ',\n')
    f.close()
    
def save_txt(filename, data):
    with open(filename, 'a') as f:
        f.write(data)
    f.close()


def list_to_csv(li):
    result = ''
    for i in li:
        result += '{}, '.format(i)
    return result


def regeneralize(result, mean, std):  # 输入的result是个array
    mean_matrix = mean
    std_matrix = std
    result_real_scale = (result * std_matrix) + mean_matrix
    return result_real_scale


def get_file_list(prefix, folder):
    result = []
    file_list = os.listdir(folder)
    for file in file_list:
        if file.endswith(prefix):
            result.append(os.path.join(folder, file))
    return result


def save_var(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def shrink(data, size):
    i = 0
    while i+size < data.shape[1]:
        yield data[:, i:i+size, :].mean(dim=1)
        i += size

# =======================================
class Record:
    
    def __init__(self):
        self.data = []
        self.mean = []
        self.count = 0
        self.reset()

    def reset(self):
        self.data = []
        self.mean = []
        self.count = 0

    def update(self, val, n=1):
        self.data.append(val)
        if type(val) == list:
            self.mean.append(np.mean(val))
        else:
            self.mean.append(val)
        self.count += n
    
    def get_latest(self, mean=True):
        if mean:
            return self.mean[-1]
        else:
            return self.data[-1]

    def delta(self):
        return abs(self.mean[-1] - self.mean[-2])

    def bigger(self):
        return self.mean[-1] - self.mean[-2] > 0

    def check(self, val):
        return (self.mean[-1] - np.mean(val)) < 0

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


