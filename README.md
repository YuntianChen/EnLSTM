# EnLSTM
Ensemble long short-term memory is a gradient-free neural network that combines ensemble neural network and long short-term memory.

This is part of implmentation of the paper [Ensemble long short-term memory (EnLSTM) network](https://arxiv.org/abs/2004.13562) by Yuantian Chen, Dongxiao Zhang, and Yuanqi Cheng.

It uses the ENN algorithm(see [Ensemble Neural Networks (ENN): A gradient-free stochastic method](https://www.sciencedirect.com/science/article/pii/S0893608018303319)) to train an artifical neural network instead of back propogation.

## Data Preperation and Model Definition

### Dataset Decription

The dataset used in this example contains data files formated as .csv files with header of feature names.

### Loading dataset

 ```python
 text = TextDataset()
```

### Neural Network Definition

```python
class netLSTM_withbn(nn.Module):

    def __init__(self):
        super(netLSTM_withbn, self).__init__()
        self.lstm = nn.LSTM(config.input_dim,
                            config.hid_dim,
                            config.num_layer,
                            batch_first=True,
                            dropout=config.drop_out)

        self.fc2 = nn.Linear(config.hid_dim,
                             int(config.hid_dim / 2))
        self.fc3 = nn.Linear(int(config.hid_dim / 2),
                             config.output_dim)
        self.bn = nn.BatchNorm1d(int(config.hid_dim / 2))

    def forward(self, x, hs=None, use_gpu=config.use_gpu):
        batch_size = x.size(0)
        if hs is None:
            h = Variable(t.zeros(config.num_layer,
                                 batch_size,
                                 config.hid_dim))
            c = Variable(t.zeros(config.num_layer,
                                 batch_size,
                                 config.hid_dim))
            hs = (h, c)
        if use_gpu:
            hs = (hs[0].cuda(), hs[1].cuda())
        out, hs_0 = self.lstm(x, hs)
        out = out.contiguous()
        out = out.view(-1, config.hid_dim)
        out = F.relu(self.bn(self.fc2(out)))
        out = self.fc3(out)
        return out, hs_0
```

---

## Requirements

The program is written in Python, and uses pytorch, scipy. A GPU is necessary, the ENN algorithm can only be running when CUDA is available.

- Install PyTorch(pytorch.org)
- Install CUDA
- `pip install -r requirements.txt`

## Training

- The training parameters like learning rate can be adjusted in configuration.py

```python
# training parameters
self.ne = 100
self.T = 1
self.batch_size = 32
self.num_workers = 1
self.epoch = 3
self.GAMMA = 10
```

- To train a model, install the required module and run train.py, the result will be saved in the experiment folder.

```bash
python train.py
```



# Model_files
The folders in the 'Model_files' are the supporting materials for the EnLSTM, including experiment results and model files for each experiment. There are five experiment folders and one Pytorch code in the 'Model_files'. Each experiment file includes 14 trained neural network model files and three MSE loss files.

Regarding the trained neural network model files, which contain all the weights and bias in the EnLSTM. These files can be directly loaded in the ‘evaluate.py’ program and generate the corresponding models in Pytorch. People can use these models to predict and generate well logs. The file name contains 4 digits. The first digit indicates the group of experiments (in order to avoid the impact of randomness, all experiments are repeated five times), and the last two digits indicate the ID of the test well in the leave-one-out method. For example, model ‘1011’ means that the model is trained in the first group of experiment where the 11st well is taken as the test well.

Regarding the MSE loss, it contains the MSE loss of three epochs in a group of experiments. For example, exp1_epoch2 represents the MSE loss of the third epoch in the first experiment (epoch is calculated from 0). Each data file contains a matrix with size of 14*13. The 14 rows correspond to the 14 wells in the dataset, and the 13 columns are the 12 well logs and the average MSE. Specifically, the columns are Young’s modulus E_x and E_y, cohesion C, uniaxial compressive strength UCS, density ρ, tensile strength TS, brittleness index BI_x and BI_y, Poisson’s ratio ν_x and ν_y, neutron porosity NPR, total organic carbon TOC and average MSE from left to right.

Regarding the Pytorch code, it is named as “evaluate.py”. The model files can be directly loaded by this program and generate the corresponding models in Pytorch.

