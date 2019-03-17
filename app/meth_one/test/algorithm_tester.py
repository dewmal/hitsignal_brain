import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import settings
from app.meth_one.processors import create_panda_data_frame_2min_ready

prediction_step = 5
algo_max_window_size = 26
window_size = 30
DATA_LENGTH = 1000
BATCH_SIZE = 1024
N_STEPS = window_size
N_INPUTS = 2
N_NEURONS = 128
N_OUTPUTS = 1
N_EPHOCS = 100

consider_value = 'close'

names = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
df = pd.read_csv(f"{settings.DATA_FOLDER}/EURUSD/HISTDATA_COM_MT_EURUSD_M1201902/DAT_MT_EURUSD_M1_201902.csv",
                 names=names, parse_dates=[['date', 'time']])

df = df.set_index('date_time')
# df = df.head(DATA_LENGTH)

# Predict
if len(df) < algo_max_window_size + window_size:
    print(f"Predict min size {algo_max_window_size + window_size}")

# Train
if len(df) < algo_max_window_size + window_size + prediction_step:
    print(f"Train min size {algo_max_window_size + window_size + prediction_step}")

data_frame = pd.DataFrame()
data_frame[consider_value] = df[consider_value]

data_frame = create_panda_data_frame_2min_ready(df=data_frame, value_name=consider_value, sma_1=6, sma_2=14, sma_3=26,
                                                macd_1=12,
                                                macd_2=26, macd_3=9,
                                                )
data_frame['prediction'] = data_frame[consider_value].shift(-prediction_step)
data_frame['price_change'] = data_frame['prediction'] - data_frame[consider_value]
data_frame['price_change_pip'] = data_frame['price_change'] / 0.0001  # 1pip equal = 0.0001 point change

data_frame.dropna(inplace=True)

print(data_frame.head(1))


def draw_diag(data_frame):
    fig, ax = plt.subplots()
    data_frame['mcad_12_26'].plot(label="mcad_12_26")
    data_frame['mcad_12_26_signal'].plot(label="mcad_12_26_signal")
    # ax.legend(['mcad_12_26', 'mcad_12_26_signal'])
    leg = ax.legend()
    fig2, ax2 = plt.subplots()
    data_frame['close'].plot(label="close")
    data_frame['sma_close_6'].plot(label="sma_close_6")
    data_frame['sma_close_14'].plot(label="sma_close_14")
    data_frame['sma_close_26'].plot(label="sma_close_26")
    leg = ax2.legend()
    # ax2.legend(['close', 'sma6', 'sma14', 'sma26'])
    plt.show()


class XYDataSet(Dataset):
    def __init__(self, df, window_size, train=True):
        self.train = train
        self.window_size = window_size
        self.df = df

    def __getitem__(self, idx):
        data_window = data_frame.iloc[idx:idx + window_size, 4:6]  # At idx data window

        if self.train:
            result_data_window = data_frame.iloc[idx + window_size, 8:9]  # Ai idx prediction must be
            return data_window.values.astype(np.float), result_data_window.values.astype(np.float)
        else:
            return data_window.values.astype(np.float)

    def __len__(self):
        return len(self.df) - window_size


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class StrategyModel(nn.Module):

    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(StrategyModel, self).__init__()
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.batch_size = batch_size

        self.rnn = nn.RNN(self.n_inputs, self.n_neurons)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons)).to(device)

    def forward(self, input):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        # X = input.permute(1,0,2)
        self.batch_size = input.shape[0]
        X = torch.reshape(input, (self.n_steps, self.batch_size, self.n_inputs))

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()

        lstm_out, self.hidden = self.rnn(X, self.hidden)

        out = self.FC(self.hidden)

        return out.view(-1, self.n_outputs)  # batch_size X n_output


_dataset = XYDataSet(data_frame, window_size, train=True)
dataloader = DataLoader(_dataset, batch_size=BATCH_SIZE,
                        shuffle=True)

model = StrategyModel(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()

    # TRAINING ROUND

    for i, data in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # reset hidden states
        model.hidden = model.init_hidden()

        # prepare the inputs
        x_vals, y_vals = data
        x_vals = x_vals.to(device).float()
        y_vals = y_vals.to(device).float()

        # print(x_vals)

        y_pred = model(x_vals)

        # forward + backward + optimize
        outputs = model(x_vals)

        # print(y_pred, y_vals)
        # print(outputs.shape, y_vals.shape)

        loss = criterion(outputs, y_vals)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()

    model.eval()
    if epoch % 1 == 0:
        print('Epoch:  %d | Loss: %.10f | Train Accuracy: %.10f'
              % (epoch, train_running_loss / i, train_acc / i))
