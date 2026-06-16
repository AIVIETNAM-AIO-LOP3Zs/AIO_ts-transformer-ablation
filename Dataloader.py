import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETTDataset(Dataset):
    """Load ETT dataset"""
    
    def __init__(self, csv_path, seq_len=96, label_len=48, pred_len=24,
                 split='train', scale=True, features='M', target='OT'):
        # Load CSV
        df_raw = pd.read_csv(csv_path)

        # Build temporal marks from the date column before dropping it.
        # Columns/order match TemporalEmbedding: [hour (0-23), day-of-month (1-31), weekday (0-6)]
        dates = pd.to_datetime(df_raw['date'])
        df_stamp = pd.DataFrame({
            'hour': dates.dt.hour,
            'day': dates.dt.day,
            'weekday': dates.dt.weekday,
        })
        df = df_raw.drop(columns=['date'])  # Remove date column

        # Train/Val/Test split (60/20/20) — apply the SAME slice to data and marks
        n = len(df)
        train_idx = int(n * 0.6)
        val_idx = int(n * 0.8)

        if split == 'train':
            sl = slice(0, train_idx)
        elif split == 'val':
            sl = slice(train_idx, val_idx)
        else:
            sl = slice(val_idx, n)
        df = df.iloc[sl]
        df_stamp = df_stamp.iloc[sl]

        # Normalize features only (never the calendar marks)
        if scale:
            self.scaler = StandardScaler()
            df = pd.DataFrame(
                self.scaler.fit_transform(df),
                columns=df.columns
            )

        self.data = df.values
        self.data_stamp = df_stamp.values.astype('float32')
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.columns = df.columns.tolist()
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Encoder input: [t, t+seq_len)
        x_enc = self.data[idx:idx+self.seq_len].copy()
        
        # Decoder input: [t+seq_len-label_len, t+seq_len+pred_len)
        x_dec_start = idx + self.seq_len - self.label_len
        x_dec_end = x_dec_start + self.label_len + self.pred_len
        x_dec = self.data[x_dec_start:x_dec_end].copy()
        x_dec[self.label_len:] = 0  # Zero padding for prediction part

        # Temporal marks aligned with the encoder / decoder windows.
        # Decoder marks keep the real calendar stamps for the prediction horizon
        # (the future timestamps are known even though the values are masked).
        x_enc_mark = self.data_stamp[idx:idx+self.seq_len].copy()
        x_dec_mark = self.data_stamp[x_dec_start:x_dec_end].copy()

        # Target: [t+seq_len, t+seq_len+pred_len)
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len].copy()
        
        # Handle feature modes
        if self.features == 'S':  # Univariate
            target_idx = self.columns.index(self.target)
            x_enc = x_enc[:, [target_idx]]
            x_dec = x_dec[:, [target_idx]]
            y = y[:, [target_idx]]
        elif self.features == 'MS':  # Multi-to-uni
            target_idx = self.columns.index(self.target)
            y = y[:, [target_idx]]
        
        return {
            'x_enc': torch.FloatTensor(x_enc),
            'x_dec': torch.FloatTensor(x_dec),
            'x_enc_mark': torch.FloatTensor(x_enc_mark),
            'x_dec_mark': torch.FloatTensor(x_dec_mark),
            'y': torch.FloatTensor(y),
        }

# Usage
if __name__ == "__main__":
    dataset = ETTDataset('Data/ETTm1.csv', seq_len=96, label_len=48, pred_len=24)
    loader = DataLoader(dataset, batch_size=32)

    for batch in loader:
        print(f"Encoder input:  {batch['x_enc'].shape}")       # (32, 96, 7)
        print(f"Decoder input:  {batch['x_dec'].shape}")       # (32, 72, 7)
        print(f"Encoder marks:  {batch['x_enc_mark'].shape}")  # (32, 96, 3)
        print(f"Decoder marks:  {batch['x_dec_mark'].shape}")  # (32, 72, 3)
        print(f"Target:         {batch['y'].shape}")           # (32, 24, 7)
        break