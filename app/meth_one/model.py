# Model your
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPredictModel(nn.Module):

    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, device='cpu'):
        super(LSTMPredictModel, self).__init__()
        self.device = device
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.batch_size = batch_size

        self.rnn = nn.LSTM(self.n_inputs, self.n_neurons)
        self.do1 = nn.Dropout(p=0.025)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons)).to(self.device)

    def forward(self, input):
        self.batch_size = input.shape[0]
        X = torch.reshape(input, (self.n_steps, self.batch_size, self.n_inputs))

        self.batch_size = X.size(1)

        lstm_out, self.hidden = self.rnn(X)
        out = self.do1(lstm_out)
        out = lstm_out[-1].view(self.batch_size, -1)
        out = torch.sigmoid(self.FC(out))
        out = out.view(-1, self.n_outputs)  # batch_size X n_output
        # print(out.shape)
        return out, lstm_out, self.hidden


class CNNForDetect(nn.Module):

    def __init__(self, in_channels, out_channels, window_size):
        super(CNNForDetect, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(window_size * out_channels // 2, window_size)

        # 64 input features, 10 output features for our 10 defined classes

        self.fc2 = torch.nn.Linear(window_size, out_channels)

    def forward(self, x):
        # print("-----")
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        # print(x.shape)
        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension

        x = torch.reshape(x, (x.shape[0], self.out_channels // 2, self.window_size))
        x = x.view(-1, x.shape[2] * x.shape[1])
        # print(x.shape)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        # print("-----")
        # print(x.shape)
        return (x)


class StrategyModel(nn.Module):

    def __init__(self, batch_size, n_steps, price_n_inputs, sma_n_inputs, mcad_n_inputs, n_neurons, n_outputs,
                 device='cpu'):
        super(StrategyModel, self).__init__()
        self.price_n_inputs = price_n_inputs

        self.device = device
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.sma_n_inputs = sma_n_inputs
        self.mcad_n_inputs = mcad_n_inputs
        self.n_neurons = n_neurons
        self.batch_size = batch_size

        self.price_rnn = LSTMPredictModel(batch_size, n_steps, price_n_inputs, n_neurons, n_outputs * 10, device=device)
        self.sma_rnn = LSTMPredictModel(batch_size, n_steps, sma_n_inputs, n_neurons, n_outputs * 10, device=device)
        self.mcad_rnn = LSTMPredictModel(batch_size, n_steps, mcad_n_inputs, n_neurons, n_outputs * 10, device=device)

        self.price_cnn = CNNForDetect(price_n_inputs, n_outputs * 10, window_size=n_steps)
        self.sma_cnn = CNNForDetect(sma_n_inputs, n_outputs * 10, window_size=n_steps)
        self.mcad_cnn = CNNForDetect(mcad_n_inputs, n_outputs * 10, window_size=n_steps)

        self.l_price_sma = nn.Linear(self.n_outputs * 10, self.n_outputs)

        self.bl1 = nn.Bilinear(self.n_outputs * 10, self.n_outputs * 10, self.n_outputs * 10)
        self.bl2 = nn.Bilinear(self.n_outputs * 10, self.n_outputs * 10, self.n_outputs * 10)
        self.bl3 = nn.Bilinear(self.n_outputs * 10, self.n_outputs * 10, self.n_outputs * 10)
        self.bl4 = nn.Bilinear(self.n_outputs * 10, self.n_outputs * 10, self.n_outputs * 10)

        self.ln1 = nn.LayerNorm(self.n_outputs * 10, eps=0.00025)

        self.l1 = nn.Linear(self.n_outputs * 10, self.n_outputs * 5)
        self.l1_2 = nn.Linear(self.n_outputs * 10, self.n_outputs)
        self.l2 = nn.Linear(self.n_outputs * 5, self.n_outputs * 3)
        self.l3 = nn.Linear(self.n_outputs * 3, self.n_outputs * 2)
        self.l4 = nn.Linear(self.n_outputs * 2, self.n_outputs)

    def forward(self, inputs):
        # print(inputs.shape)
        price = inputs[:, :, 0:1]
        sma_input = inputs[:, :, 1:4]
        mcad_input = inputs[:, :, 4:]

        # CNN Output
        x_price_cnn = self.price_cnn(
            torch.reshape(price, (price.shape[0], price.shape[2], price.shape[1])))
        x_sma_cnn = self.sma_cnn(torch.reshape(sma_input, (sma_input.shape[0], sma_input.shape[2], sma_input.shape[1])))
        x_mcad_cnn = self.mcad_cnn(
            torch.reshape(mcad_input, (mcad_input.shape[0], mcad_input.shape[2], mcad_input.shape[1])))

        # x_price_cnn = self.ln1(x_price_cnn)

        # LSTM Output
        x_price, _x_price_lstm, _ = self.price_rnn(price)
        x_sma, _x_sma_lstm, _ = self.sma_rnn(sma_input)
        x_mcad, _x_mcad_lstm, _ = self.mcad_rnn(mcad_input)

        price_x = self.bl1(x_price_cnn, x_price)  # torch.add(x_price_cnn, 1, x_price)
        sma_x = self.bl1(x_sma_cnn, x_sma)  # torch.add(x_price_cnn, 1, x_price)
        mcad_x = self.bl1(x_mcad_cnn, x_mcad)  # torch.add(x_price_cnn, 1, x_price)

        # sma_x = torch.addcmul(sma_x, 0.5, mcad_x, sma_x)

        x = price_x + sma_x

        price_x = self.l_price_sma(x)

        x = torch.add(price_x, 0.01, mcad_x)
        x = self.ln1(x)
        x1 = torch.sigmoid(self.l1((x)))
        x11 = self.l1_2(x)

        x2 = torch.sigmoid(self.l2((x1)))
        x3 = torch.sigmoid(self.l3((x2)))
        x3 = x3 + x11 / 2
        x4 = torch.sigmoid(self.l4((x3)))
        # print(x.shape)
        return x4


def build_model(device, settings, load_from_file=True):
    model = StrategyModel(settings.BATCH_SIZE, settings.N_STEPS, 1, 3, 2, settings.N_NEURONS,
                          settings.N_OUTPUTS, device=device)

    if settings.model_path is not None and load_from_file:
        model = torch.load(settings.model_path)

        # print(chkpt)
        # model.load_state_dict()
    model = model.to(device)
    print(model)
    return model
