# Model your
import torch
import torch.nn as nn
import torch.functional as F


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
        self.l1 = nn.Linear(self.n_outputs * 10, self.n_outputs)

        self.l_price = nn.Linear(self.price_n_inputs, n_outputs * 10)
        self.l_price_activation = nn.LeakyReLU()
        self.l_sma = nn.Linear(self.sma_n_inputs, n_outputs * 10)
        self.l_sma_activation = nn.LeakyReLU()
        self.l_mcad = nn.Linear(self.mcad_n_inputs, n_outputs * 10)
        self.l_mcda_activation = nn.LeakyReLU()

    def forward(self, inputs):
        price, sma_input, mcad_input = inputs[:, :, 0:1], inputs[:, :, 1:4], inputs[:, :, 4:6]
        sma_input = sma_input
        mcad_input = mcad_input

        # print(sma_input.shape, mcad_input.shape)
        # print(sma_input,mcad_input)
        # x_price, _x_price_lstm, _x_price_lstm_hidden = self.price_rnn(price)
        # x_sma, _x_sma_lstm, _x_sma_lstm_hidden = self.sma_rnn(sma_input)
        # x_mcad, _x_mcad_lstm, _x_macd_lstm_hidden = self.mcad_rnn(mcad_input)

        x_price = self.l_price(price[:, -1, :])
        x_price = self.l_price_activation(x_price)
        x_sma = self.l_sma(sma_input[:, -1, :])
        x_sma = self.l_sma_activation(x_sma)
        x_mcad = self.l_mcad(mcad_input[:, -1, :])
        x_mcad = self.l_mcda_activation(x_mcad)
        #
        # print(_x_price_lstm.shape, _x_sma_lstm.shape, _x_mcad_lstm.shape)
        # print(_x_price_lstm_hidden[0].shape, _x_sma_lstm_hidden[0].shape, _x_macd_lstm_hidden[0].shape)
        # print(_x_price_lstm_hidden[1].shape, _x_sma_lstm_hidden[1].shape, _x_macd_lstm_hidden[1].shape)

        x = (x_sma + x_mcad + x_price) / 3
        x = torch.tanh(x)
        # x = torch.sigmoid(x)
        x = self.l1(x)
        # print(x.shape)
        return x
