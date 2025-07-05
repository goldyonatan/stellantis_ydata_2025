import os
import pandas as pd
import numpy as np
import warnings
import traceback

# --- Import necessary functions from HelperFuncs ---
try:
    # Add safe_mean and sort_df_by_trip_time
    from HelperFuncs import (
        load_file, save_file, sort_df_by_trip_time, drop_trips_with_less_than_x_segs, find_seg_start_idx
        )
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")

# --- Configuration Block ---
CLEANED_DATA_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\clean_df\clean_df.parquet'
SEGMENTATION_DICT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict\trips_seg_dict_v7.pickle"

OUTPUT_FEATURE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_30s.pickle" 
OUTPUT_FORMAT = 'pickle'

MIN_SEGMENTS_PER_TRIP = 1
WIN_LEN_S = 30
WIN_HOP_S = 15
FULL_STOP = 3

ADD_REL_SPEED = True
ADD_CURVATURE = True

# AUX_SOC_LOSS = True
#PROCESS_SUBSET = None # To process only x samples

# --- Standard Column Names (Must match output of preprocessing script) ---
TRIP_ID_COL = 'trip_id'
TIME_COL = 'timestamp'
SPEED_ARRAY_COL = 'speed_array'
ALT_ARRAY_COL = 'altitude_array'
# -------------------------------------------------

# --- Feature Extraction Functions ---
def get_windows_list(df: pd.DataFrame, trips_seg_dict: dict, full_stop:int=3, add_rel_speed: bool=True,
                     add_curvature: bool=True):
    """
    edit
    """

    if add_rel_speed:
        print("Relative speed is not implemented yet. Skipping for now.")
    
    if add_curvature:
        print("Curvature is not implemented yet. Skipping for now.")

    windows = []

    for trip_id, trip_labels in trips_seg_dict.items():
        trip_df = df[df[TRIP_ID_COL] == trip_id].copy()
        if len(trip_df) != len(trip_labels):
            warnings.warn(f"Length mismatch for trip_id {trip_id}. Skipping trip.")
            continue

        unique_seg_ids = np.unique(trip_labels[trip_labels > 0]).astype(int)
        if len(unique_seg_ids) == 0: 
            print(f"No segments for trip: {trip_id}")
            continue

        for seg_id in unique_seg_ids:
            labeled_indices = np.where(trip_labels == seg_id)[0]
            if len(labeled_indices) == 0: 
                print(f"No samples for trip_id {trip_id} seg {seg_id}!")
                continue
            
            first_labeled_idx = labeled_indices[0]
            last_labeled_idx = labeled_indices[-1]

            segment_start_idx = find_seg_start_idx(trip_labels, first_labeled_idx, full_stop=full_stop)
            segment_span_indices = np.arange(segment_start_idx, last_labeled_idx + 1)

            df_seg = trip_df.iloc[segment_span_indices,:]

            all_seg_speeds = np.concatenate(df_seg[SPEED_ARRAY_COL].values)

            all_seg_accel = np.empty_like(all_seg_speeds, dtype=float)
            all_seg_accel[0]      = np.nan
            all_seg_accel[1:]     = np.diff(all_seg_speeds)

            all_seg_jerk  = np.empty_like(all_seg_speeds, dtype=float)
            all_seg_jerk[:2]      = np.nan
            all_seg_jerk[2:]      = np.diff(all_seg_accel[1:])

            all_seg_alts = np.concatenate(df_seg[ALT_ARRAY_COL].values)

            orig_len = len(all_seg_alts)
            new_len  = len(all_seg_speeds)

            # original sample indices
            x = np.arange(orig_len)

            # target sample indices at the higher rate
            x_new  = np.linspace(0, orig_len - 1, new_len)

            # interpolate
            upsampled_all_seg_alts = np.interp(x_new, x, all_seg_alts)

            all_seg_slope_rate = np.empty_like(upsampled_all_seg_alts, dtype=float)
            all_seg_slope_rate[0]      = np.nan
            all_seg_slope_rate[1:] = np.diff(upsampled_all_seg_alts) 
            
            win_id = 0
            for i in range(0, new_len - WIN_LEN_S + 1, WIN_HOP_S):
                w = slice(i, i + WIN_LEN_S) 
                windows.append({
                    "trip_id": trip_id,
                    "seg_id" : seg_id,
                    "win_id" : win_id,
                    "start_sample": i,
                    "accel": all_seg_accel[w],
                    "jerk": all_seg_jerk[w],
                    "slope_rate": all_seg_slope_rate[w]
                })
                win_id += 1

    df_windows = pd.DataFrame.from_records(windows)
    return df_windows
                
def main():
    """Loads data, extracts segment features, saves results."""
    print("--- Feature Extraction Pipeline for driver embedding ---")

    print(f"\nInput cleaned data file: {CLEANED_DATA_PATH}")
    print(f"Input segmentation dict: {SEGMENTATION_DICT_PATH}")
    print(f"Output features file: {OUTPUT_FEATURE_PATH}")

    try:
        # 1. Load Cleaned Data
        print("\n--- Loading Cleaned Data ---")
        df_clean = load_file(CLEANED_DATA_PATH)
        if not isinstance(df_clean, pd.DataFrame): raise TypeError("Cleaned data is not a DataFrame.")
        print(f"Loaded cleaned data with shape: {df_clean.shape}")

        # 2. Ensure Sorting using the Helper Function
        print(f"\n--- Sorting data by {TRIP_ID_COL}, {TIME_COL} ---")
        df_clean = sort_df_by_trip_time(df_clean, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)
        print("Data sorting complete.")

        # 3. Load Segmentation Dictionary
        print(f"\n--- Loading Segmentation Dictionary ---")
        trips_seg_dict = load_file(SEGMENTATION_DICT_PATH)
        if not isinstance(trips_seg_dict, dict): raise TypeError("Segmentation data is not a dictionary.")
        print(f"Loaded segmentation dict with {len(trips_seg_dict)} trips.")

        # 4. Filter Segmentation Dict for Min Segments
        df_clean_filtered, filtered_trips_seg_dict = drop_trips_with_less_than_x_segs(
            df_clean, 
            trips_seg_dict, 
            trip_id_col=TRIP_ID_COL, 
            min_segments=MIN_SEGMENTS_PER_TRIP
        )

        # 6. Extract windows list
        windows_df = get_windows_list(
            df_clean_filtered,
            filtered_trips_seg_dict,
            full_stop=FULL_STOP
        )

        # 7. Save Results
        if not windows_df.empty:
            print(f"\n--- Saving windows df ---")
            output_dir = os.path.dirname(OUTPUT_FEATURE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FEATURE_PATH))[0]
            save_file(data=windows_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT)
        else:
            print("\nWarning: Resulting windows df is empty. Nothing to save.")

        print("\n--- Feature Extraction Pipeline Finished ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Feature Extraction Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()