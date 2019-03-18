import matplotlib.pyplot as plt
import pandas as pd
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import MeanSquaredError, Loss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import settings
from app.meth_one import settings as meth_settings
from app.meth_one.data_model import XYDataSet
from app.meth_one.data_processor import prepare_data_set
from app.meth_one.model import build_model

names = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']
df = pd.read_csv(f"{settings.DATA_FOLDER}/EURUSD/HISTDATA_COM_MT_EURUSD_M1201902/DAT_MT_EURUSD_M1_201902.csv",
                 names=names, parse_dates=[['date', 'time']])

df = df.set_index('date_time')
df = df.head(meth_settings.DATA_LENGTH)


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


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

data_frame = prepare_data_set(_data_frame=df, window_size=meth_settings.window_size,
                              consider_value=meth_settings.consider_value,
                              prediction_step=meth_settings.prediction_step,
                              algo_max_window_size=meth_settings.algo_max_window_size)
# Prepare Data set
val_size = int(len(data_frame) * meth_settings.VALIDATION_SIZE)
train_size = len(data_frame) - val_size

_dataset = XYDataSet(data_frame.iloc[:train_size], meth_settings.window_size, train=True)
train_data_loader = DataLoader(_dataset, batch_size=meth_settings.BATCH_SIZE,
                               shuffle=True)

_dataset_val = XYDataSet(data_frame.iloc[train_size:], meth_settings.window_size, train=True)
val_data_loader = DataLoader(_dataset_val, batch_size=meth_settings.BATCH_SIZE)

# Create Model
train_model = build_model(device,meth_settings)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(train_model.parameters(), lr=0.01, momentum=0.01)

if meth_settings.LOG_X:
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
    train_model.train()
    inputs, targets = batch
    inputs = inputs.float().to(device)
    targets = targets.float().to(device)

    optimizer.zero_grad()
    outputs = train_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(engine, batch):
    train_model.eval()
    with torch.no_grad():
        inputs, y = batch
        inputs = inputs.float().to(device)
        y = y.float().to(device)
        y_pred = train_model(inputs)
        return y_pred, y


trainer = Engine(
    train_and_store_loss)  # create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=device)
evaluator = Engine(evaluate)  # create_supervised_evaluator(model, metrics={'accuracy': Accuracy()})

metric_val = MeanSquaredError()
metric_val.attach(evaluator, 'mse')

loss_metric = Loss(loss_fn=criterion)
loss_metric.attach(evaluator, "nll")

max_epochs = meth_settings.N_EPHOCS
validate_every = 1
checkpoint_every = 10

if meth_settings.LOG_X:
    writer = create_summary_writer(train_model, train_data_loader, f"{settings.LOG_FOLDER}/tensorx")


@trainer.on(Events.ITERATION_COMPLETED)
def validate(trainer):
    if trainer.state.iteration % validate_every == 0:
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        train_metrics = trainer.state.metrics
        print("After {} iterations, val loss = {:.10f}".format(
            trainer.state.iteration, metrics['mse']
        ))
        if meth_settings.LOG_X:
            writer.add_scalar("training/loss", trainer.state.output, trainer.state.iteration)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_data_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['mse']
    avg_nll = metrics['nll']
    print("Training Results - Epoch: {}  Avg accuracy: {:.10f} Avg loss: {:.10f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))
    if meth_settings.LOG_X:
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
    if meth_settings.LOG_X:
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)


check_pointer = ModelCheckpoint(f"{settings.MODEL_FOLDER}/meht_one_algo_one/", "model_1",
                                save_interval=checkpoint_every,
                                create_dir=True, require_empty=False
                                )

if __name__ == '__main__':
    trainer.add_event_handler(Events.ITERATION_COMPLETED, check_pointer, {"mymodel": train_model})
    trainer.run(train_data_loader, max_epochs=max_epochs)

    if meth_settings.LOG_X:
        writer.close()
