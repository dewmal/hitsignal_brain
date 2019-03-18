import pandas as pd
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.simple_moving_average import simple_moving_average as sma

from app.meth_one import settings
from app.meth_one.model import build_model


def prepare_data_set(_data_frame, algo_max_window_size=None, window_size=None, consider_value=None,
                     prediction_step=None, train=True):
    # Predict
    if len(_data_frame) < algo_max_window_size + window_size:
        print(f"Predict min size {algo_max_window_size + window_size}")

    # Train
    if len(_data_frame) < algo_max_window_size + window_size + prediction_step:
        print(f"Train min size {algo_max_window_size + window_size + prediction_step}")

    data_frame = pd.DataFrame()
    data_frame[consider_value] = _data_frame[consider_value]

    data_frame = create_panda_data_frame_2min_ready(df=data_frame, value_name=consider_value, sma_1=6, sma_2=14,
                                                    sma_3=26,
                                                    macd_1=12,
                                                    macd_2=26, macd_3=9,
                                                    )
    if train:
        data_frame['prediction'] = data_frame[consider_value].shift(-prediction_step)
        # data_frame['price_change'] = data_frame['prediction'] - data_frame[consider_value]
        data_frame['price_change_pip'] = (data_frame['prediction'] - data_frame[
            consider_value]) / 0.0001  # 1pip equal = 0.0001 point change

    data_frame.dropna(inplace=True)

    print(data_frame.tail())
    return data_frame


def create_panda_data_frame_2min_ready(df, value_name, sma_1, sma_2, sma_3, macd_1, macd_2, macd_3,
                                       new_data_frame=False, drop_na=True):
    if new_data_frame:
        data_frame = pd.DataFrame()
    else:
        data_frame = df

    data_frame[f'sma_{value_name}_{sma_1}'] = sma(data_frame[value_name].values, sma_1)  # 6
    data_frame[f'sma_{value_name}_{sma_2}'] = sma(data_frame[value_name].values, sma_2)  # 14
    data_frame[f'sma_{value_name}_{sma_3}'] = sma(data_frame[value_name].values, sma_3)  # 26
    # 12,26,9
    data_frame[f'mcad_{macd_1}_{macd_2}'] = macd(data_frame.close.values, macd_1, macd_2)
    data_frame[f'mcad_{macd_1}_{macd_2}_signal'] = data_frame[f'mcad_{macd_1}_{macd_2}'].values / ema(
        data_frame.close.values,
        macd_3)  # MACD Signal Line
    if drop_na:
        data_frame.dropna(inplace=True)
    return data_frame


def predict(df, window_size=None, consider_value=None, prediction_step=None, algo_max_window_size=None):
    import torch
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_frame = prepare_data_set(_data_frame=df, window_size=window_size, consider_value=consider_value,
                                  prediction_step=prediction_step,
                                  train=False,
                                  algo_max_window_size=algo_max_window_size)

    df = data_frame.iloc[len(data_frame) - window_size:, :6]
    predict_model = build_model(device, settings=settings)
    predict_model.eval()
    values = df.values
    values = np.reshape(values, (1, values.shape[0], values.shape[1]))
    pred_values = predict_model(torch.from_numpy(values).float().to(device))
    return pred_values.cpu().detach().numpy(), df.index.values[-1] + prediction_step * 60
