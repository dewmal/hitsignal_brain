# Model your
import torch
import torch.nn as nn


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

    def forward(self, inputs):
        print(inputs.shape)
        price = inputs[:, :, 0:1]
        sma_input = inputs[:, :, 1:4]
        mcad_input = inputs[:, :, 4:]
        print(inputs.shape, price.shape, sma_input.shape, mcad_input.shape)
        # print(inputs, price, sma_input, mcad_input)

        # print(sma_input.shape, mcad_input.shape)
        # print(sma_input,mcad_input)
        x_price, _x_price_lstm, _ = self.price_rnn(price)
        x_sma, _x_sma_lstm, _ = self.sma_rnn(sma_input)
        x_mcad, _x_mcad_lstm, _ = self.mcad_rnn(mcad_input)
        # print(_x_price_lstm.shape, _x_sma_lstm.shape, _x_mcad_lstm.shape)

        x = (x_sma + x_mcad + x_price) / 3
        x = torch.tanh(x)
        # x = torch.sigmoid(x)
        x = self.l1(x)
        return x


def build_model(device, settings, model_path=None):
    model = StrategyModel(settings.BATCH_SIZE, settings.N_STEPS, 1, 3, 2, settings.N_NEURONS,
                          settings.N_OUTPUTS, device=device).to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    return model
