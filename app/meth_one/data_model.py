from torch.utils.data import Dataset


class XYDataSet(Dataset):
    def __init__(self, df, window_size, train=True):
        self.train = train
        self.window_size = window_size
        self.df = df

    def __getitem__(self, idx):
        data_window = self.df.iloc[idx:idx + self.window_size, :6]  # At idx data window

        if self.train:
            result_data_window = self.df.iloc[idx + self.window_size, 6:]  # Ai idx prediction must be
            return data_window.values, result_data_window.values
        else:
            return data_window.values

    def __len__(self):
        return len(self.df) - self.window_size
