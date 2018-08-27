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



def train_model(model, train_x, train_y, valid_x, valid_y, state_path = "lstm_state.pkl", epochs = 1000):
  """
  Train a model for financial data
  """

  best_valid_loss = float("inf")
  criterion = nn.SmoothL1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

  train_loss = []
  valid_loss = []
  for epoch in range(epochs):
    pred = model(train_x)
    loss = criterion(pred, train_y)
    v_pred = model(valid_x)
    v_loss = criterion(v_pred, valid_y)
    valid_loss.append(float(v_loss))
    train_loss.append(float(loss))
    if (float(v_loss) < best_valid_loss):
      msg = "\ntrain_loss = {:.3f} | valid_loss = {:.3f} | epoch = {:.1f}\n".format(float(loss),float(v_loss), float(epoch))
      torch.save(model.state_dict(), state_path)
      best_valid_loss = float(v_loss)
      print(msg, end="")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return train_loss, valid_loss


parser = argparse.ArgumentParser(description='Run the thing.')
parser.add_argument('time', metavar='N', type=int, nargs='+',
                   help='prediction lag, 1, 5, 10 or 21')
parser.add_argument('--cleaned', dest='cleaned', action='store_true')
parser.set_defaults(cleaned=False)

args = parser.parse_args()
N = args.time[0]

print("loading raw data for predicting " + str(N) + " days")

if (args.cleaned):
  print("using cleaned series")
  raw_file = "../input_files/input_vol" + str(N) + "d_clean.csv"
  state_file = "lstm_state_" + str(N) + "d_clean.pkl"
else:
  raw_file = "../input_files/input_vol" + str(N) + "d.csv"
  state_file = "lstm_state_" + str(N) + "d.pkl"


raw_data = np.loadtxt(open(raw_file, "rb"), delimiter = ",", skiprows = 1)

# Test: (nrows - 30) ~ end (last 30 days of sample)
# Validation: (nrows - 160) ~ (nrows - 30) (around 6 months of validation)
# Train: 1 ~ (nrows - 160)  (the rest of sample)

test_start = len(raw_data) - 30
validation_start = len(raw_data) - 160

raw_x = raw_data[:(validation_start - 1), :-1]
raw_y = raw_data[:(validation_start - 1), -1]

raw_x_valid = raw_data[validation_start:(test_start - 1), :-1]
raw_y_valid = raw_data[validation_start:(test_start - 1), -1]

print(" Raw x shape")
print(" " + str(raw_x.shape))

print(" Raw y shape")
print(" " + str(raw_y.shape))

print(" Raw x valid shape")
print(" " + str(raw_x_valid.shape))

print(" Raw y valid shape")
print(" " + str(raw_y_valid.shape))

train_x = torch.from_numpy(raw_x.reshape(-1, 1, 8))
train_y = torch.from_numpy(raw_y.reshape(-1, 1, 1))

valid_x = torch.from_numpy(raw_x_valid.reshape(-1, 1, 8))
valid_y = torch.from_numpy(raw_y_valid.reshape(-1, 1, 1))


model = LSTM(8, 16).double()

print("starting training")
t_loss, v_loss = train_model(model, train_x, train_y, valid_x, valid_y, state_file)
