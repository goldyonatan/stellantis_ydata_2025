import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
try:
    # Add safe_mean and sort_df_by_trip_time
    from HelperFuncs import load_file
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")

WINDOWS_DF_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_30s.pickle"
OUTPUT_MODEL_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_30s.pickle" 
OUTPUT_DRIVERS_EMBEDDINGS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_30s.pickle" 

# AUX_SOC_LOSS = True
WIN_LEN_S = 30         # seconds
MIN_ABS_WIN_DIFF = 5
DROP_MAP = {           # channel-wise “ignore” length
    0: 1,              # accel   ⇒ skip t0
    1: 2,              # jerk    ⇒ skip t0,t1
    2: 1               # slope   ⇒ skip t0
}
SAME_DIFF_RATIO = 1/5
RANDOM_STATE = 42
RNG = np.random.default_rng(seed=RANDOM_STATE)

# --- Standard Column Names (Must match output of preprocessing script) ---
TRIP_ID_COL = 'trip_id'
SEG_ID_COL = 'seg_id'
WIN_ID_COL = 'win_id'

class WinDataset(Dataset):
    def __init__(self, windows_df):       
        self.df = windows_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _row_to_tensor(self, row):
        # stack the three 1-D arrays → shape (L, 3)
        x = np.stack([row.accel, row.jerk, row.slope_rate], axis=-1).astype(np.float32)

        # Boolean mask: True = valid timestep, False = ignore in the loss
        mask = np.ones(x.shape[0], dtype=bool)
        for ch, n_drop in DROP_MAP.items():        # e.g. {0:1, 1:2, 2:1}
            mask[:n_drop] &= False                 # first 1-2 steps marked False

        # Replace NaNs *in place* with 0.0 so the tensor contains only finite numbers
        np.nan_to_num(x, copy=False)

        # Return PyTorch tensors
        return torch.from_numpy(x), torch.from_numpy(mask)

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]

        get_same = RNG.choice([0, 1], p=[1 - SAME_DIFF_RATIO, SAME_DIFF_RATIO])

        if get_same:
            if anchor_row['trip_max_seg'] > 1:
                pool = self.df[
                    (self.df[TRIP_ID_COL] == anchor_row[TRIP_ID_COL]) & 
                    (self.df[SEG_ID_COL] != anchor_row[SEG_ID_COL])
                    ]
            else:
                pool = self.df[
                    (self.df[TRIP_ID_COL] == anchor_row[TRIP_ID_COL]) & 
                    (abs(self.df[WIN_ID_COL] - anchor_row[WIN_ID_COL]) >= MIN_ABS_WIN_DIFF)
                    ]
        else:
            pool = self.df[self.df[TRIP_ID_COL] != anchor_row[TRIP_ID_COL]]

        other_row = pool.sample(n=1, random_state=RANDOM_STATE).iloc[0]
        x_a, m_a = self._row_to_tensor(anchor_row)
        x_b, m_b = self._row_to_tensor(other_row)
        label    = torch.tensor(1 if get_same else 0, dtype=torch.float32)

        return (x_a, m_a, x_b, m_b, label)
    
def main():
    """Loads windows df"""
    print("--- Pipeline for driver embedding ---")

    print(f"\nInput windows df file: {WINDOWS_DF_PATH}")
    print(f"Output model file: {OUTPUT_MODEL_PATH}")
    print(f"Output drivers embeddings file: {OUTPUT_DRIVERS_EMBEDDINGS_PATH}")

    windows_df = load_file(WINDOWS_DF_PATH)
    windows_df_copy = windows_df.copy()
    windows_df_copy['trip_max_seg'] = windows_df_copy.groupby('trip_id')['seg_id'].transform('max')

    dataset = WinDataset(windows_df_copy)

if __name__ == '__main__':
    main()