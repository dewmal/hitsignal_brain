import matplotlib.pyplot as plt
import pandas as pd
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import MeanSquaredError, Loss
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import settings
from app.meth_one.model import StrategyModel
from app.meth_one.processors import create_panda_data_frame_2min_ready

prediction_step = 3
algo_max_window_size = 26
window_size = 50
DATA_LENGTH = 1000000
BATCH_SIZE = 128
N_STEPS = window_size
N_INPUTS = 2
N_NEURONS = 256
N_OUTPUTS = 2
N_EPHOCS = 100
VALIDATION_SIZE = 0.2
LOG_X = False
consider_value = 'close'

names = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
df = pd.read_csv(f"{settings.DATA_FOLDER}/EURUSD/HISTDATA_COM_MT_EURUSD_M1201902/DAT_MT_EURUSD_M1_201902.csv",
                 names=names, parse_dates=[['date', 'time']])

df = df.set_index('date_time')
df = df.head(DATA_LENGTH)

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
# data_frame['price_change'] = data_frame['prediction'] - data_frame[consider_value]
data_frame['price_change_pip'] = (data_frame['prediction'] - data_frame[
    consider_value]) / 0.0001  # 1pip equal = 0.0001 point change

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
        data_window = data_frame.iloc[idx:idx + window_size, :6]  # At idx data window

        if self.train:
            result_data_window = data_frame.iloc[idx + window_size, 6:]  # Ai idx prediction must be
            return data_window.values, result_data_window.values
        else:
            return data_window.values

    def __len__(self):
        return len(self.df) - window_size


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# Prepare Data set
val_size = int(len(data_frame) * VALIDATION_SIZE)
train_size = len(data_frame) - val_size

_dataset = XYDataSet(data_frame.iloc[:train_size], window_size, train=True)
train_data_loader = DataLoader(_dataset, batch_size=BATCH_SIZE,
                               shuffle=True)

_dataset_val = XYDataSet(data_frame.iloc[train_size:], window_size, train=True)
val_data_loader = DataLoader(_dataset_val, batch_size=BATCH_SIZE)

# Create Model
model = StrategyModel(BATCH_SIZE, N_STEPS, 1, 3, 2, N_NEURONS, N_OUTPUTS, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.01)

if LOG_X:
    def create_summary_writer(model, data_loader, log_dir):
        import os
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        data_loader_iter = iter(data_loader)
        x, y = next(data_loader_iter)
        x = x.float().to(device)
        y = y.float().to(device)

        try:
            writer.add_graph(model, x)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
        return writer


def train_and_store_loss(engine, batch):
    model.train()
    inputs, targets = batch
    inputs = inputs.float().to(device)
    targets = targets.float().to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(engine, batch):
    model.eval()
    with torch.no_grad():
        inputs, y = batch
        inputs = inputs.float().to(device)
        y = y.float().to(device)
        y_pred = model(inputs)
        return y_pred, y


# def output_transform(output):
#     # `output` variable is returned by above `process_function`
#     y_pred = output['y_pred']
#     y = output['y_true']
#     return y_pred, y  # output format is according to `Accuracy` docs

trainer = Engine(
    train_and_store_loss)  # create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=device)
evaluator = Engine(evaluate)  # create_supervised_evaluator(model, metrics={'accuracy': Accuracy()})

metric_val = MeanSquaredError()
metric_val.attach(evaluator, 'mse')

loss_metric = Loss(loss_fn=criterion)
loss_metric.attach(evaluator, "nll")

max_epochs = N_EPHOCS
validate_every = 1
checkpoint_every = 10

if LOG_X:
    writer = create_summary_writer(model, train_data_loader, f"{settings.LOG_FOLDER}/tensorx")


@trainer.on(Events.ITERATION_COMPLETED)
def validate(trainer):
    if trainer.state.iteration % validate_every == 0:
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        train_metrics = trainer.state.metrics
        print("After {} iterations, val loss = {:.10f}".format(
            trainer.state.iteration, metrics['mse']
        ))
        if LOG_X:
            writer.add_scalar("training/loss", trainer.state.output, trainer.state.iteration)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_data_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['mse']
    avg_nll = metrics['nll']
    print("Training Results - Epoch: {}  Avg accuracy: {:.10f} Avg loss: {:.10f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))
    if LOG_X:
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_data_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))
    if LOG_X:
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)


check_pointer = ModelCheckpoint(f"{settings.MODEL_FOLDER}/meht_one_algo_one/", "model_1",
                                save_interval=checkpoint_every,
                                create_dir=True, require_empty=False
                                )
trainer.add_event_handler(Events.ITERATION_COMPLETED, check_pointer, {"mymodel": model})
trainer.run(train_data_loader, max_epochs=max_epochs)

if LOG_X:
    writer.close()
