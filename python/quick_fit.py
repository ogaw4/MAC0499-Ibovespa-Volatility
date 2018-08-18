import os
import numpy as np
import time
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import argparse
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
  """
  Recurrent LSTM network
  """
  def __init__(self, features, hidden_size, num_layers = 2, output_size = 1):
    super(LSTM, self).__init__()

    self.lstm = nn.LSTM(features, hidden_size, num_layers)
    self.lin = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    """
    Computes forward pass
    """
    x, states = self.lstm(x) 
    seq_len, batch, hidden = x.shape
    x = x.view(seq_len * batch, hidden)
    x = self.lin(x)
    x = x.view(seq_len, batch, -1)
    return x



parser = argparse.ArgumentParser(description='Run the thing.')
parser.add_argument('time', metavar='N', type=int, nargs='+',
                   help='prediction lag, 1, 5, 10 or 21')

args = parser.parse_args()
N = args.time[0]

print("loading raw data for predicting " + str(N) + " days")

raw_file = "input_vol" + str(N) + "d.csv"

raw_data = np.loadtxt(open(raw_file, "rb"), delimiter = ",", skiprows = 1)

# Test: 2306 ~ end (2017-08-16 ~ 2018-03-29)
# Validation: 2065 ~ 2305 (2016-08-15 ~ 2017-08-15)
# Train: 1 ~ 2064 (2008-01-31 ~ 2016-08-12)

raw_x = raw_data[:2064, :-1]
raw_y = raw_data[:2064, -1]

raw_x_valid = raw_data[2065:2305, :-1]
raw_y_valid = raw_data[2065:2305, -1]

raw_x_test = raw_data[:, :-1]
raw_y_test = raw_data[:, -1]

print(" Raw x shape")
print(" " + str(raw_x.shape))

print(" Raw y shape")
print(" " + str(raw_y.shape))

print(" Raw x valid shape")
print(" " + str(raw_x_valid.shape))

print(" Raw y valid shape")
print(" " + str(raw_y_valid.shape))

print(" Raw x test shape")
print(" " + str(raw_x_test.shape))

train_x = torch.from_numpy(raw_x.reshape(-1, 1, 8))
train_y = torch.from_numpy(raw_y.reshape(-1, 1, 1))

valid_x = torch.from_numpy(raw_x_valid.reshape(-1, 1, 8))
valid_y = torch.from_numpy(raw_y_valid.reshape(-1, 1, 1))

test_x = torch.from_numpy(raw_x_test.reshape(-1, 1, 8))

model = LSTM(8, 16).double()

print("loading best valid loss")
model.load_state_dict(torch.load("lstm_state.pkl"))
model = model.eval() 
pred_test = model(test_x).view(-1).data.numpy()

print("pred test shape is " + str(pred_test.shape))

with open('results.csv', "w") as file:
    file.write("pred\n")
    for i in range(pred_test.shape[0]):
        file.write("{}\n".format(pred_test[i]))