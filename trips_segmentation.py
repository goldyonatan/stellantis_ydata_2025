import numpy as np
import pandas as pd
import os
import warnings

# import contants definitions from cons.py:

# configuration:
from cons import CLEANED_DATA_PATH, SEGMENTATION_DICT_DIR, SEGMENTATION_DICT_FILENAME
from cons import OUTPUT_FORMAT_FOR_SEGMENTATION as OUTPUT_FORMAT

# Parameters from stop-finding and segmentation:
from cons import STOP_SPEED_KPH, STOP_DIST_KM, MIN_SEGMENT_POINTS, CONSECUTIVE_STOPS_FOR_BREAK

# standard column names:
from cons import TRIP_ID_COL, TIME_COL, ODO_COL, SPEED_ARRAY_COL




# --- Import necessary functions from HelperFuncs ---
try:
    # Import the renamed/modified sort function
    from HelperFuncs import load_file, save_file, random_partition, safe_mean, sort_df_by_trip_time
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    # Define placeholders
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")
    def random_partition(total, num_parts): raise NotImplementedError("HelperFuncs not found") # Assuming this exists
    def safe_mean(val): raise NotImplementedError("HelperFuncs not found")
    def sort_df_by_trip_time(df, trip_id_col='trip_id', time_col='timestamp'): raise NotImplementedError("HelperFuncs not found")

# --- Segmentation Functions (Updated for Standard Column Names) ---

def find_natural_stops(df, trip_id_col='trip_id', time_col='timestamp',
                       speed_array_col='speed_array', odo_col='current_odo',
                       thr_speed_kph=2.0, thr_dist_km=0.05, stop_indicator=-1):
    """
    Identifies periods of stopping based on low speed and minimal distance traveled.
    Uses standardized column names.

    Args:
        df (pd.DataFrame): Input DataFrame (should be sorted by trip_id, timestamp).
        trip_id_col (str): Name of the trip identifier column.
        time_col (str): Name of the timestamp column.
        speed_array_col (str): Name of the column containing speed arrays (e.g., 1Hz).
        odo_col (str): Name of the current odometer column.
        thr_speed_kph (float): Speed threshold (kph) below which a stop might occur.
        thr_dist_km (float): Distance threshold (km). If odo change is less than this,
                             it's considered minimal movement.
        stop_indicator (int): Value to assign to stop rows in the output array.

    Returns:
        dict: Dictionary mapping trip_id to a NumPy array indicating stops.
    """
    print("\n--- Finding Natural Stops ---")
    df_copy = df.copy()
    required_cols = [trip_id_col, time_col, speed_array_col, odo_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns for find_natural_stops: {missing}")

    cycles_natural_stops_dict = {}
    # Use groupby for potentially better performance than iterating unique cycles
    for trip_id, trip_df in df_copy.groupby(trip_id_col):
        if len(trip_df) < 2: # Need at least 2 points to calculate diffs
            cycles_natural_stops_dict[trip_id] = np.zeros(len(trip_df)) # Mark all as non-stop
            continue

        # Compute mean speed for each row using the safe_mean helper
        try:
            mean_speeds = trip_df[speed_array_col].apply(safe_mean).values
        except Exception as e:
             print(f"Warning: Error calculating mean speed for trip {trip_id}. Skipping stop detection. Error: {e}")
             cycles_natural_stops_dict[trip_id] = np.zeros(len(trip_df))
             continue

        # Compute odometer changes between consecutive rows
        # Ensure odo is numeric first
        trip_df[odo_col] = pd.to_numeric(trip_df[odo_col], errors='coerce')
        # Use diff() which handles first NaN correctly
        odo_changes = trip_df[odo_col].diff().values # odo_changes[0] will be NaN

        # Initialize flags array with zeros
        flags = np.zeros(len(trip_df))

        # Iterate through the rows to find stop sequences
        i = 0
        while i < len(trip_df):
            # Check if speed is below threshold (handle potential NaN mean speed)
            if pd.isna(mean_speeds[i]) or mean_speeds[i] >= thr_speed_kph:
                i += 1
            else:
                # Start a potential stop sequence
                j = i + 1
                # Extend the sequence
                # Check j bounds, speed condition, and odo change condition
                # odo_changes[j] corresponds to the change *between* row j-1 and row j
                while (j < len(trip_df) and
                       (pd.isna(mean_speeds[j]) or mean_speeds[j] < thr_speed_kph) and
                       (pd.isna(odo_changes[j]) or abs(odo_changes[j]) < thr_dist_km)): # Check abs() change < threshold
                    j += 1

                # Flag all rows in the sequence [i, j)
                flags[i:j] = stop_indicator
                # Move to the next potential sequence start
                i = j

        cycles_natural_stops_dict[trip_id] = flags

    print(f" - Stop detection complete for {len(cycles_natural_stops_dict)} trips.")
    return cycles_natural_stops_dict

def count_cycles_with_x_consecutive_stops(cycles_dict, x, stop_indicator = -1):
    count = 0
    for cycle_id, arr in cycles_dict.items():
        # Iterate through all possible starting indices where a block of x elements can fit.
        found = False
        for i in range(len(arr) - x + 1):
            if np.all(arr[i:i+x] == stop_indicator):
                found = True
                break
        if found:
            count += 1
    return count

# --- (split_rem_non_overlap_walk_forward_seg function remains the same conceptually) ---
def split_rem_non_overlap_walk_forward_seg(cycles_natural_stops_dict, min_win=10, stop_indicator=-1, full_stop=3):
    """Assigns segment IDs based on driving periods between stops."""
    print("\n--- Splitting Trips into Segments ---")
    print(f" - Minimum driving rows per segment (min_win): {min_win}")
    print(f" - Consecutive stops indicating segment end (full_stop): {full_stop}")

    cycles_natural_stops_dict_copy = cycles_natural_stops_dict.copy()
    processed_cycles = 0
    total_segments_created = 0

    for trip_id, stop_flags in cycles_natural_stops_dict_copy.items():
        n_samples_cycle = len(stop_flags)
        segment_labels = stop_flags.copy().astype(float) # Use float to allow NaNs or keep int if preferred
        current_segment_id = 1.0 # Start segment IDs at 1

        i = 0
        while i < n_samples_cycle:
            # Find start of a driving block (skip initial stops)
            if segment_labels[i] == stop_indicator:
                i += 1
                continue
            driving_block_start = i

            # Find end of the driving block
            consecutive_stops = 0
            driving_block_end = driving_block_start # End index (exclusive)
            valid_driving_samples = 0
            j = driving_block_start
            while j < n_samples_cycle:
                if segment_labels[j] == stop_indicator:
                    consecutive_stops += 1
                else:
                    consecutive_stops = 0
                    valid_driving_samples += 1 # Count only non-stop samples

                # End block if enough consecutive stops are found
                if consecutive_stops >= full_stop:
                    driving_block_end = j - consecutive_stops + 1 # End index is start of full stop sequence
                    break
                j += 1
            else: # If loop finishes without breaking (end of trip)
                driving_block_end = n_samples_cycle

            # Process the identified driving block [driving_block_start, driving_block_end)
            num_segments_in_block = valid_driving_samples // min_win

            if num_segments_in_block > 0:
                remainder_samples = valid_driving_samples % min_win
                # Distribute remainder using random_partition helper
                try:
                    partition_additions = random_partition(remainder_samples, num_segments_in_block)
                except Exception as e:
                     print(f"Warning: random_partition failed for trip {trip_id}. Distributing remainder evenly. Error: {e}")
                     # Fallback: distribute as evenly as possible
                     base_add = remainder_samples // num_segments_in_block
                     extra_add = remainder_samples % num_segments_in_block
                     partition_additions = [base_add + 1] * extra_add + [base_add] * (num_segments_in_block - extra_add)


                # Assign segment IDs within the block
                current_pos_in_block = driving_block_start
                for addition in partition_additions:
                    segment_len = min_win + addition
                    labeled_driving_samples = 0
                    k = current_pos_in_block
                    segment_end_pos = k # Track where the segment actually ends
                    while labeled_driving_samples < segment_len and k < driving_block_end:
                        if segment_labels[k] != stop_indicator:
                            segment_labels[k] = current_segment_id
                            labeled_driving_samples += 1
                        segment_end_pos = k + 1 # Move end position
                        k += 1
                    current_pos_in_block = segment_end_pos # Start next segment after this one ends
                    current_segment_id += 1
                    total_segments_created += 1

            # Move main loop index past the processed block (or the full stop sequence)
            i = driving_block_end + (full_stop if consecutive_stops >= full_stop else 0)

        # Update the dictionary with the array containing segment labels
        cycles_natural_stops_dict_copy[trip_id] = segment_labels.astype(int) # Convert back to int if desired
        processed_cycles += 1

    print(f" - Segmentation complete for {processed_cycles} trips.")
    print(f" - Total segments created: {total_segments_created}")
    return cycles_natural_stops_dict_copy

def find_natural_stops(df, trip_id_col='trip_id', time_col='timestamp',
                       speed_array_col='speed_array', odo_col='current_odo',
                       thr_speed_kph=2.0, thr_dist_km=0.05, stop_indicator=-1): # <-- Added parameters here
    """
    Identifies periods of stopping based on low speed and minimal distance traveled.
    Uses standardized column names passed as arguments.

    Args:
        df (pd.DataFrame): Input DataFrame (should be sorted by trip_id, timestamp).
        trip_id_col (str): Name of the trip identifier column.
        time_col (str): Name of the timestamp column.
        speed_array_col (str): Name of the column containing speed arrays (e.g., 1Hz).
        odo_col (str): Name of the current odometer column.
        thr_speed_kph (float): Speed threshold (kph) below which a stop might occur.
        thr_dist_km (float): Distance threshold (km). If odo change is less than this,
                             it's considered minimal movement.
        stop_indicator (int): Value to assign to stop rows in the output array.

    Returns:
        dict: Dictionary mapping trip_id to a NumPy array indicating stops.
    """
    print("\n--- Finding Natural Stops ---")
    required_cols = [trip_id_col, time_col, speed_array_col, odo_col]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns for find_natural_stops: {missing}")

    # Ensure data is sorted (important for diff calculations)
    df_sorted = df.sort_values([trip_id_col, time_col]).copy()
    print(f" - Processing {df_sorted[trip_id_col].nunique()} trips...")

    cycles_natural_stops_dict = {}
    for trip_id, trip_df in df_sorted.groupby(trip_id_col):
        if len(trip_df) < 2:
            cycles_natural_stops_dict[trip_id] = np.zeros(len(trip_df))
            continue

        try:
            mean_speeds = trip_df[speed_array_col].apply(safe_mean).values
        except Exception as e:
             print(f"Warning: Error calculating mean speed for trip {trip_id}. Skipping stop detection. Error: {e}")
             cycles_natural_stops_dict[trip_id] = np.zeros(len(trip_df))
             continue

        trip_df[odo_col] = pd.to_numeric(trip_df[odo_col], errors='coerce')
        odo_changes = trip_df[odo_col].diff().values

        flags = np.zeros(len(trip_df))
        i = 0
        while i < len(trip_df):
            if pd.isna(mean_speeds[i]) or mean_speeds[i] >= thr_speed_kph:
                i += 1
            else:
                j = i + 1
                while (j < len(trip_df) and
                       (pd.isna(mean_speeds[j]) or mean_speeds[j] < thr_speed_kph) and
                       (pd.isna(odo_changes[j]) or abs(odo_changes[j]) < thr_dist_km)):
                    j += 1
                flags[i:j] = stop_indicator
                i = j

        cycles_natural_stops_dict[trip_id] = flags

    print(f" - Stop detection complete for {len(cycles_natural_stops_dict)} trips.")
    return cycles_natural_stops_dict


# --- Main Execution Logic ---
def main():
    """Loads cleaned data, finds stops, segments trips, and saves results."""

    # --- Configuration ---


    print(f"--- Starting Trip Segmentation Pipeline ---")
    print(f"\nInput cleaned data file: {CLEANED_DATA_PATH}")
    print(f"Output directory for segmentation dict: {SEGMENTATION_DICT_DIR}")

    try:
        # 1. Load Cleaned Data
        print("\n--- Loading Cleaned Data ---")
        df_clean = load_file(CLEANED_DATA_PATH)
        if not isinstance(df_clean, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded cleaned data with shape: {df_clean.shape}")

        # 2. Ensure Sorting using the Helper Function
        print(f"\n--- Sorting data by {TRIP_ID_COL}, {TIME_COL} ---")
        # Call the helper function using the standard column names
        df_clean = sort_df_by_trip_time(df_clean, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)
        print("Data sorting complete.")

        # 3. Find Natural Stops
        # Pass standard column names
        cycles_natural_stops_dict = find_natural_stops(
            df_clean, # Pass the sorted DataFrame
            trip_id_col=TRIP_ID_COL, time_col=TIME_COL,
            speed_array_col=SPEED_ARRAY_COL, odo_col=ODO_COL,
            thr_speed_kph=STOP_SPEED_KPH, thr_dist_km=STOP_DIST_KM
        )

        # 4. Segment Trips
        cycles_seg_dict = split_rem_non_overlap_walk_forward_seg(
            cycles_natural_stops_dict,
            min_win=MIN_SEGMENT_POINTS,
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK
        )

        # for cyc in cycles_seg_dict:
        #     print(f"\ncycle: {cyc}")
        #     print(cycles_seg_dict[cyc])

        # 5. Save Segmentation Dictionary
        print(f"\n--- Saving Segmentation Dictionary ---")
        save_file(
            data=cycles_seg_dict, path=SEGMENTATION_DICT_DIR,
            file_name=SEGMENTATION_DICT_FILENAME, format=OUTPUT_FORMAT
        )

        print("\n--- Trip Segmentation Pipeline Complete ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Segmentation Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    import traceback
    main()




# def get_prediction_periods(cycles_natural_stops_dict, 
#                            min_pre_samples=10, 
#                            min_period_samples=10, 
#                            full_stop=3,
#                            stop_indicator = -1):
#     """
#     For each cycle, label consecutive 'prediction periods' 
#     (1, 2, 3, ...) based on:
    
#       1) Only the first period in the cycle has a 'warmup' requirement
#          of min_pre_samples rows.
#       2) We do not start counting that warmup if the row is stop_indicator.
#          Once we see a non-stop_indicator, we begin counting, including
#          any subsequent stop_indicator rows.
#       3) We cannot start a period on stop_indicator.
#       4) We stop the period before any block of 'full_stop'
#          consecutive stop_indicator rows.
#       5) The period must have at least 'min_period_samples' rows.
#     """
#     cycles_natural_stops_dict_copy = cycles_natural_stops_dict.copy()

#     for cyc, cyc_data in cycles_natural_stops_dict_copy.items():

#         n_samples_cycle = len(cyc_data)
        
#         # Tracking for the current cycle
#         curr_period = 1
#         first_period = True  # only the first period in each cycle enforces warmup
        
#         # Warmup logic
#         warmup_started = False
#         counted_samples = 0
        
#         i = 0
#         while i < n_samples_cycle:
            
#             # --------------- FIRST-PERIOD WARMUP LOGIC ---------------
#             if first_period and not warmup_started:
#                 # We have NOT started counting yet
#                 if cyc_data[i] == stop_indicator:
#                     # skip -1 rows until we see a non-stop_indicator to begin counting
#                     i += 1
#                     continue
#                 else:
#                     # Found our first non-stop_indicator row
#                     warmup_started = True
#                     counted_samples = 1
#                     i += 1
#                     continue
            
#             if first_period and warmup_started and counted_samples < min_pre_samples:
#                 # Keep counting rows (including -1) until we reach min_pre_samples
#                 counted_samples += 1
#                 i += 1
#                 continue
            
#             # --------------- READY TO START A PERIOD ---------------
#             # If we are still in the first_period but the warmup is complete,
#             # OR if we're past the first_period entirely, we can look for a start.
            
#             # Skip if the row is stop_indicator. We cannot start a period on stop_indicator.
#             if cyc_data[i] == stop_indicator:
#                 i += 1
#                 continue
            
#             # Now we have a valid starting point for a period
#             period_start = i
            
#             # Find the end by searching for 'full_stop' consecutive stop_indicator
#             consecutive_stops = 0
#             period_end = None
#             j = i
#             while j < n_samples_cycle:
#                 if cyc_data[j] == stop_indicator:
#                     consecutive_stops += 1
#                 else:
#                     consecutive_stops = 0

#                 if consecutive_stops == full_stop:
#                     # Period ends just before this block
#                     period_end = j - full_stop
#                     break
#                 j += 1
            
#             # If we never hit a block of 'full_stop', period ends at the last index
#             if period_end is None:
#                 period_end = n_samples_cycle - 1
            
#             # Check if period has at least min_period_samples rows
#             length = period_end - period_start + 1
#             if length >= min_period_samples:
#                 # Label this period
#                 for k in range(period_start, period_end + 1):
#                     if cyc_data[k] != stop_indicator:
#                         cyc_data[k] = curr_period
#                 curr_period += 1

#             # Advance i to period_end + 1 to look for the next period
#             i = period_end + 1
            
#             # We have handled the first_period, so no more warmup constraints
#             first_period = False
#             warmup_started = False  # not relevant any more, but can reset

#         # Update the cycle data
#         cycles_natural_stops_dict_copy[cyc] = cyc_data

#     return cycles_natural_stops_dict_copy

# def walk_forward_seg(cycles_prediction_periods_dict, min_win=10, max_win=15, stop_indicator=-1):

#     cycles_prediction_periods_dict_copy = cycles_prediction_periods_dict.copy()
    
#     # Process each cycle independently
#     for cyc, cyc_data in cycles_prediction_periods_dict_copy.items():
#         n_samples_cycle = len(cyc_data)
#         curr_per = 0  # Reset per cycle 
#         start_idx = 0  # Start scanning from the beginning of the cycle
        
#         # Continue to segment until we reach the end of the cycle
#         while start_idx < n_samples_cycle:
#             # Find the first valid index that is neither stop_indicator nor curr_per
#             win_start = next((i for i in range(start_idx, n_samples_cycle)
#                               if cyc_data[i] not in [stop_indicator, curr_per]), None)
#             if win_start is None:
#                 break  # No valid start found in the remaining cycle
            
#             # Find an index indicating where the current segment might end
#             per_end = next((i for i in range(win_start, n_samples_cycle)
#                             if cyc_data[i] not in [stop_indicator, curr_per, curr_per + 1]), None)
#             if per_end is None:
#                 per_end = n_samples_cycle - 1
            
#             per_length = per_end - win_start
#             if per_length >= min_win:
#                 # Choose a random end within the allowed window range
#                 win_end = random.randint(win_start + min_win, min(per_end, win_start + max_win))
                
#                 # Label this segment
#                 for k in range(win_start, win_end + 1):
#                     if cyc_data[k] != stop_indicator:
#                         cyc_data[k] = curr_per + 1
                        
#                 # Update current period and advance start_idx to avoid overlapping with already labeled segment
#                 curr_per += 1
#                 start_idx = win_end + 1
#             else:
#                 # If not enough data for a segment, break
#                 break
            
#     return cycles_prediction_periods_dict_copy


# def fixed_non_overlap_walk_forward_seg(cycles_natural_stops_dict, win_size=10, stop_indicator=-1, full_stop=3):
    
#     # Make a shallow copy to avoid mutating original
#     cycles_natural_stops_dict_copy = cycles_natural_stops_dict.copy()
    
#     # Process each cycle independently
#     for cyc, cyc_data in cycles_natural_stops_dict_copy.items():
        
#         n_samples_cycle = len(cyc_data)

#         curr_per = 1

#         valid_prev = False

#         i = 0
#         while i < n_samples_cycle:

#             # Skip stop_indicator rows
#             if cyc_data[i] == stop_indicator:
#                 i += 1
#                 continue
            
#             # Attempt to define one segment of length win_size
#             win_start = i

#             consec_stops = 0
#             j = win_start
#             win_non_stops = 0

#             # Check for a block of full_stop consecutive stops
#             while j < n_samples_cycle:
#                 if cyc_data[j] == stop_indicator:
#                     consec_stops += 1
#                 else:
#                     consec_stops = 0
#                     win_non_stops += 1

#                 if consec_stops == full_stop:
#                     # We found a block of stops => do partial labeling or skip
#                     if valid_prev:
#                         # Label partial
#                         for h in range(win_start, j - full_stop + 1):
#                             if cyc_data[h] != stop_indicator:
#                                 cyc_data[h] = curr_per - 1
                    
#                     i = j + 1
#                     valid_prev = False
#                     break
                
#                 if win_non_stops == win_size:
#                     for k in range(win_start, j + 1):
#                         if cyc_data[k] != stop_indicator:
#                             cyc_data[k] = curr_per

#                     curr_per += 1
#                     i = j + 1
#                     valid_prev = True
#                     break
                
#                 j += 1

#             if win_non_stops <  win_size:
#                 # Label partial remainder if not in first segment
#                 if valid_prev:
#                     for d in range(win_start, n_samples_cycle):
#                         if cyc_data[d] != stop_indicator:
#                             cyc_data[d] = curr_per - 1
#                 break # End of cycle
        
#         cycles_natural_stops_dict_copy[cyc] = cyc_data

#     return cycles_natural_stops_dict_copy


# def split_rem_non_overlap_walk_forward_seg(cycles_natural_stops_dict, min_win=10, stop_indicator=-1, full_stop=3):
    
#     # Make a shallow copy to avoid mutating original
#     cycles_natural_stops_dict_copy = cycles_natural_stops_dict.copy()
    
#     # Process each cycle independently
#     for cyc, cyc_data in cycles_natural_stops_dict_copy.items():
        
#         n_samples_cycle = len(cyc_data)

#         curr_per = 1

#         i = 0
#         while i < n_samples_cycle:

#             # Skip stop_indicator rows
#             if cyc_data[i] == stop_indicator:
#                 i += 1
#                 continue
            
#             win_start = i

#             consec_stops = 0
#             j = win_start
#             valid_samples = 0

#             while j < n_samples_cycle:
#                 if cyc_data[j] == stop_indicator:
#                     consec_stops += 1
#                 else:
#                     consec_stops = 0
#                     valid_samples += 1

#                 if consec_stops == full_stop:

#                     break

#                 j += 1
            
#             n_seg = valid_samples // min_win

#             if n_seg > 0:
#                 to_split = valid_samples % min_win 

#                 # Partition to_split into n_seg parts, ensuring the sum equals to_split.
#                 partition = random_partition(to_split, n_seg)

#                 for addition in partition:
#                     labeled = 0
#                     k = win_start
#                     while labeled < min_win + addition and k < n_samples_cycle:
#                         if cyc_data[k] != stop_indicator:
#                             cyc_data[k] = curr_per
#                             labeled += 1
#                         k += 1
#                     win_start = k
#                     curr_per += 1
                    
#             i = j + 1
                   
#         cycles_natural_stops_dict_copy[cyc] = cyc_data

#     return cycles_natural_stops_dict_copy

#     df = load_file(file_path)
#     print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

#     # sorting the df by cycle and time
#     df = sort_df_by_cycle_time(df)

#     #df = drop_short_long_trips(df, min_dur=0, min_dis=0, max_dur=1_000, max_dis=1_000)

#     cycles_natural_stops_dict = find_natural_stops(df, thr_speed=1.5, thr_dis=0.1)

#     # print(f"num of cycles: {len(cycles_natural_stops_dict)}")
#     # count = sum(1 for arr in cycles_natural_stops_dict.values() if np.any(arr == -1))
#     # print("Number of cycles with at least one stop:", count)
#     # num_cycles = count_cycles_with_x_consecutive_stops(cycles_natural_stops_dict, 3)
#     # print("Number of cycles containing", 3, "consecutive stops:", num_cycles)

#     cycles_seg_dict = split_rem_non_overlap_walk_forward_seg(
#                             cycles_natural_stops_dict, min_win=10, stop_indicator=-1, full_stop=3)
    
#     # for cyc in cycles_seg_dict:
#     #     print(f"\ncycle: {cyc}")
#     #     print(cycles_seg_dict[cyc])

#     # saving segmentation dict as pickle
#     dict_output_directory = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict"
#     dict_filename = 'cycles_seg_dict'
#     save_as_pickle(cycles_seg_dict, dict_output_directory, dict_filename) 


# if __name__ == '__main__':
#     main()
