import os
import pandas as pd
import numpy as np
import joblib
import warnings
import traceback
import pickle
from collections import OrderedDict
import requests # For routing
import time # For routing

# --- Import necessary functions from HelperFuncs ---
try:
    # Add safe_mean and sort_df_by_trip_time
    from HelperFuncs import (
        load_file, save_file, safe_mean, sort_df_by_trip_time, drop_trips_with_less_than_x_segs,
        h3_to_latlon, # Assuming H3 conversion happens in preprocess
        get_osrm_route, get_osrm_match_robust, # <-- Use new robust match helpe
        #get_route_elevation_change # Import placeholder/real implementation
        calculate_kinematics, aggregate_kinematics, calculate_stop_features, # <-- Import NEW helpers
        analyze_speeding
        )
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    # Define placeholders
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")
    def safe_mean(val): raise NotImplementedError("HelperFuncs not found")
    def sort_df_by_trip_time(df, trip_id_col='trip_id', time_col='timestamp'): raise NotImplementedError("HelperFuncs not found")
    def get_route_predictions_osrm(slat,slon,elat,elon): return {'predicted_distance_m': np.nan, 'predicted_duration_s': np.nan}
    def get_route_elevation_change(slat,slon,elat,elon): return {'predicted_elevation_gain': 0.0, 'predicted_elevation_loss': 0.0}

# --- Configuration Block ---
CLEANED_DATA_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\clean_df\clean_df.parquet'
SEGMENTATION_DICT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict\cycles_seg_dict_v2.pickle"
GLOBAL_MODEL_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\models\global_baseline_model.joblib"
OUTPUT_FEATURE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segments_df\segments_features_v4_behavior.parquet" # Updated name
OSRM_URL = "http://router.project-osrm.org" # Public instance - use responsibly! Or set up local instance.

MIN_SEGMENTS_PER_TRIP = 2
OUTPUT_FORMAT = 'parquet'

FETCH_ROUTING_FEATURES = True # For predicted route (start->end)
#FETCH_MAP_MATCHING = True # For matched route comparison
#PROCESS_SUBSET = None # To process only x samples
# *** New Flag: Set to True only AFTER global model is trained and saved ***
CALCULATE_GLOBAL_PREDS = True # Default to False for initial run
# --------------------

# --- Driving Behavior & Matching Thresholds ---
ACCEL_THRESHOLD_MPS2 = 1.5; DECEL_THRESHOLD_MPS2 = -1.5; #JERK_THRESHOLD_MPS3 = 2.0

# Map Matching Config
INITIAL_MATCH_RADIUS = 25; MAX_MATCH_RADIUS = 100; MATCH_RETRY_ATTEMPTS = 3
MATCH_DISTANCE_TOLERANCE_PCT = 10.0; GAPS_THRESHOLD_S = 300
#MAX_POINTS_PER_OSRM_REQUEST = 500 # Limit points sent to OSRM match
#API_DELAY_S = 0.05 # Small delay between OSRM calls to be nice to public server
#MATCH_RADIUS_MULTIPLIER = 1.5
# ---------------------------------------------

# --- Standard Column Names (Must match output of preprocessing script) ---
TRIP_ID_COL = 'trip_id'
TIME_COL = 'timestamp'
ODO_COL = 'current_odo'
SOC_COL = 'current_soc'
SPEED_ARRAY_COL = 'speed_array'
ALT_ARRAY_COL = 'altitude_array'
TEMP_COL = 'outside_temp'
BATT_HEALTH_COL = 'battery_health' # Example static feature
LAT_COL = 'latitude'
LON_COL = 'longitude'

MEAN_SPEED_KPH_COL = 'mean_speed_kph'
ACCEL_MPS2_COL = 'accel_mps2' 
JERK_MPS3_COL = 'jerk_mps3'
TIME_DIFF_S_COL = 'time_diff_s'

# Add ALL other standard column names from clean_df.parquet that are needed
# -------------------------------------------------

# --- Feature Extraction Functions ---

# --- UPDATED get_segments_features (calls robust match) ---
def get_segments_features(df_clean, cycles_seg_dict,
                          trained_global_model, # Can be None
                          global_features, # List of features the global model NEEDS
                          fetch_routing=True, osrm_url=OSRM_URL):
    """
    Extracts segment features:
    1. Start Conditions (_seg_start, _trip)
    2. Actual Outcomes (_seg_end, _gt, _seg_actual)
    3. Predicted Properties (_seg_pred) via routing from start to *estimated* end.
    4. Actual Driving Behavior (_seg_agg, _matched, _deviation, _speeding) via map matching *actual* path.
    5. Global Prediction (_global_pred) based on PREDICTED properties.
    6. Residual Error (_residual_error).

    Args:
        df_clean (pd.DataFrame): Cleaned, row-level DataFrame (sorted).
        cycles_seg_dict (dict): Dictionary mapping trip_id to segment assignments.
        trained_global_model: Trained global baseline model object.
        global_features (list): Feature names expected by the global model
                                (should include predicted route properties).
        fetch_routing (bool): Whether to perform routing queries.
        osrm_url (str): Base URL for the OSRM routing engine API.

    Returns:
        pd.DataFrame: DataFrame with segment features, predictions, and residuals.
    """
    print("\n--- Extracting Segment Features (incl. Robust Map Matching & Behavior) ---")
    if trained_global_model:
        print("   (Global model provided - will calculate predictions and residuals)")
    else:
        print("   (Global model NOT provided - skipping predictions and residuals)")

    segment_features_list = []
    processed_trips = 0

    base_required_cols = [TRIP_ID_COL, TIME_COL, ODO_COL, SOC_COL, LAT_COL, LON_COL] # Need Lat/Lon now
    missing_base = [col for col in base_required_cols if col not in df_clean.columns]
    if missing_base: raise ValueError(f"Input DataFrame df_clean is missing essential columns: {missing_base}")

    for trip_id, seg_assignments in cycles_seg_dict.items():
        cycle_df = df_clean[df_clean[TRIP_ID_COL] == trip_id].copy()
        if len(cycle_df) != len(seg_assignments):
            warnings.warn(f"Length mismatch for trip_id {trip_id}. Skipping trip.")
            continue

        unique_seg_ids = np.unique(seg_assignments[seg_assignments > 0]).astype(int)
        unique_seg_ids = unique_seg_ids[unique_seg_ids > 0]
        if len(unique_seg_ids) == 0: continue

        for seg_id in unique_seg_ids:
            segment_mask = (seg_assignments == seg_id)
            segment_indices = np.where(segment_mask)[0]
            if len(segment_indices) == 0: continue

            first_index_pos = np.min(segment_indices)
            last_index_pos = np.max(segment_indices)
            full_segment_slice = cycle_df.iloc[first_index_pos : last_index_pos + 1]
            driving_only_segment_df = cycle_df[segment_mask]
            # Get the segmentation flags for the full slice duration
            segment_flags_slice = seg_assignments[first_index_pos : last_index_pos + 1]

            if full_segment_slice.empty or driving_only_segment_df.empty: continue

            segment_features = OrderedDict()
            segment_features['trip_id'] = trip_id # Keep simple ID names
            segment_features['segment_id'] = seg_id

            try:
                # === 1. Start-of-Segment k Features === (Use _seg_start suffix)
                segment_features['soc_seg_start'] = full_segment_slice[SOC_COL].iloc[0]
                segment_features['odo_seg_start'] = full_segment_slice[ODO_COL].iloc[0]
                segment_features['time_seg_start'] = full_segment_slice[TIME_COL].iloc[0]
                segment_features['battery_health_seg_start'] = full_segment_slice[BATT_HEALTH_COL].iloc[0]
                segment_features['hour_seg_start'] = segment_features['time_seg_start'].hour
                segment_features['dayofweek_seg_start'] = segment_features['time_seg_start'].dayofweek
                segment_features['month_seg_start'] = segment_features['time_seg_start'].month
                if LAT_COL in full_segment_slice.columns: segment_features['lat_seg_start'] = full_segment_slice[LAT_COL].iloc[0]
                if LON_COL in full_segment_slice.columns: segment_features['lon_seg_start'] = full_segment_slice[LON_COL].iloc[0]
                if TEMP_COL in full_segment_slice.columns: segment_features['temp_seg_start'] = full_segment_slice[TEMP_COL].iloc[0]
                if 'dem_altitude' in full_segment_slice.columns: segment_features['dem_altitude_seg_start'] = full_segment_slice['dem_altitude'].iloc[0]
                if 'dem_slope_deg' in full_segment_slice.columns: segment_features['dem_slope_seg_start'] = full_segment_slice['dem_slope_deg'].iloc[0]
                if 'dem_aspect_deg' in full_segment_slice.columns: segment_features['dem_aspect_seg_start'] = full_segment_slice['dem_aspect_deg'].iloc[0]

                # === 2. Static Trip/Vehicle Features ===
                static_cols_map = { # Original standard name -> New name with _trip
                    'car_model': 'car_model_trip',
                    'manufacturer': 'manufacturer_trip',
                    'empty_weight_kg': 'weight_trip_kg',
                    'battery_type': 'battery_type_trip',
                    'cycle_soc_start': 'soc_trip_start',
                    'cycle_odo_start': 'odo_trip_start'
                    # add others if available
                }
                for static_col, new_name in static_cols_map.items():
                    if static_col in full_segment_slice.columns:
                         segment_features[new_name] = full_segment_slice[static_col].iloc[0]

                # === 3. Actual Segment k Outcome & Aggregate Features ===
                # Use _seg_end, _seg_actual, _seg_agg, gt_ suffixes
                # These describe what *happened* in segment k and will be used for history in the next script
                segment_features['soc_seg_end'] = full_segment_slice[SOC_COL].iloc[-1]
                segment_features['gt_soc_seg_delta'] = segment_features['soc_seg_start'] - segment_features['soc_seg_end'] # Ground Truth

                segment_features['odo_seg_end'] = full_segment_slice[ODO_COL].iloc[-1]
                segment_features['distance_seg_actual_km'] = segment_features['odo_seg_end'] - segment_features['odo_seg_start'] # Actual distance
                segment_features['distance_seg_actual_km'] = max(0, segment_features['distance_seg_actual_km']) # Ensure non-negative

                segment_features['time_seg_end'] = full_segment_slice[TIME_COL].iloc[-1]
                duration = (segment_features['time_seg_end'] - segment_features['time_seg_start']).total_seconds()
                segment_features['duration_seg_actual_s'] = max(0, duration) # Actual duration

                # # In-segment aggregates (use driving_only_segment_df)
                # if SPEED_ARRAY_COL in driving_only_segment_df.columns:
                #     mean_speeds_in_segment = driving_only_segment_df[SPEED_ARRAY_COL].apply(safe_mean)
                #     segment_features['speed_seg_agg_mean'] = mean_speeds_in_segment.mean()
                #     segment_features['speed_seg_agg_std'] = mean_speeds_in_segment.std()
                #     segment_features['speed_seg_agg_max'] = mean_speeds_in_segment.max()
                #     #segment_features['speed_seg_agg_min'] = mean_speeds_in_segment.min() # Added min
                # if ALT_ARRAY_COL in driving_only_segment_df.columns:
                #     mean_alts_in_segment = driving_only_segment_df[ALT_ARRAY_COL].apply(safe_mean)
                #     segment_features['altitude_seg_agg_mean'] = mean_alts_in_segment.mean()
                #     segment_features['altitude_seg_agg_std'] = mean_alts_in_segment.std()
                #     if len(mean_alts_in_segment) > 0:
                #          segment_features['altitude_seg_agg_delta'] = mean_alts_in_segment.iloc[-1] - mean_alts_in_segment.iloc[0]
                #     else: segment_features['altitude_seg_agg_delta'] = 0.0
                # if TEMP_COL in driving_only_segment_df.columns:
                #      segment_features['temp_seg_agg_mean'] = driving_only_segment_df[TEMP_COL].mean()
                #      segment_features['temp_seg_agg_std'] = driving_only_segment_df[TEMP_COL].std()
                #      segment_features['temp_seg_agg_min'] = driving_only_segment_df[TEMP_COL].min() # Added min
                #      segment_features['temp_seg_agg_max'] = driving_only_segment_df[TEMP_COL].max() # Added max
                # # Add other in-segment aggregates as needed

                # === 4. Driving Behavior Aggregates (NEW) ===
                # Calculate kinematics using the full slice (includes stops potentially)
                # Or use driving_only_segment_df if you only want aggregates during driving
                # Let's use full_segment_slice for now to capture stop influence on accel/decel around them
                kinematics_df = pd.DataFrame() # Default empty
                if TIME_COL in full_segment_slice.columns and SPEED_ARRAY_COL in full_segment_slice.columns:
                     kinematics_df = calculate_kinematics(
                         full_segment_slice[TIME_COL],
                         full_segment_slice[SPEED_ARRAY_COL]
                     )

                if not kinematics_df.empty:
                    kinematic_aggs = aggregate_kinematics(
                        kinematics_df,
                        high_accel_thr=ACCEL_THRESHOLD_MPS2,
                        high_decel_thr=DECEL_THRESHOLD_MPS2
                    )
                    # Add these features with _seg_agg suffix
                    segment_features.update(kinematic_aggs) # Adds speed, accel, decel, jerk stats

                # Calculate stop features using the flags for the full slice
                stop_aggs = calculate_stop_features(segment_flags_slice, full_segment_slice[TIME_COL])
                segment_features.update(stop_aggs) # Adds stops_seg_count, stop_duration_seg_agg_s

                # Add other aggregates (Altitude, Temp) - use driving_only for mean during driving? Or full slice?
                # Let's stick to driving_only for these performance means
                if ALT_ARRAY_COL in driving_only_segment_df.columns:
                    mean_alts = driving_only_segment_df[ALT_ARRAY_COL].apply(safe_mean)
                    segment_features['altitude_seg_agg_mean'] = mean_alts.mean()
                    segment_features['altitude_seg_agg_std'] = mean_alts.std()
                    if len(mean_alts) > 0: segment_features['altitude_seg_agg_delta'] = mean_alts.iloc[-1] - mean_alts.iloc[0]
                    else: segment_features['altitude_seg_agg_delta'] = 0.0
                if TEMP_COL in driving_only_segment_df.columns:
                     segment_features['temp_seg_agg_mean'] = driving_only_segment_df[TEMP_COL].mean()
                     segment_features['temp_seg_agg_std'] = driving_only_segment_df[TEMP_COL].std()
                     segment_features['temp_seg_agg_min'] = driving_only_segment_df[TEMP_COL].min()
                     segment_features['temp_seg_agg_max'] = driving_only_segment_df[TEMP_COL].max()

                # === 5. Map Matching & Derived Actual Behavior Features ===
                # Initialize all potential output columns
                segment_features['matched_route_distance_m'] = np.nan; segment_features['matched_route_duration_s'] = np.nan
                segment_features['match_confidence'] = np.nan; segment_features['match_status'] = 'NotAttempted'
                segment_features['match_attempts'] = 0; segment_features['final_match_radius_m'] = np.nan
                segment_features['direct_route_distance_m'] = np.nan; segment_features['route_deviation_pct'] = np.nan
                segment_features['percent_time_over_limit_seg_agg'] = np.nan; segment_features['avg_speed_limit_kph_seg_agg'] = np.nan
                osrm_match_response = None

                if fetch_routing:
                    # Prepare inputs
                    coords = list(zip(full_segment_slice[LAT_COL], full_segment_slice[LON_COL]))
                    timestamps_unix = [int(ts.timestamp()) for ts in pd.to_datetime(full_segment_slice[TIME_COL], errors='coerce')]
                    odo_dist_km_actual = segment_features.get('distance_seg_actual_km', -1.0)

                    # Check validity before calling robust match
                    if len(coords) >= 2 and len(coords) == len(timestamps_unix) and \
                       not any(pd.isna(c) or not np.isfinite(c) for lat, lon in coords for c in [lat, lon]) and \
                       not any(pd.isna(ts) or not np.isfinite(ts) for ts in timestamps_unix) and \
                       np.all(np.diff(timestamps_unix) > 0): # Check monotonicity here too

                        # *** CALL ROBUST MATCH HELPER ***
                        robust_match_result = get_osrm_match_robust(
                            coordinates=coords, timestamps=timestamps_unix,
                            odo_distance_km=odo_dist_km_actual,
                            trip_id_debug=trip_id, segment_id_debug=seg_id, # Pass IDs
                            osrm_base_url=osrm_url,
                            initial_radius_m=INITIAL_MATCH_RADIUS, max_radius_m=MAX_MATCH_RADIUS,
                            max_attempts=MATCH_RETRY_ATTEMPTS + 1, tolerance_pct=MATCH_DISTANCE_TOLERANCE_PCT,
                            gaps_threshold_s=GAPS_THRESHOLD_S, request_timeout=45 # Use increased timeout
                        )
                        # Store results
                        segment_features['matched_route_distance_m'] = robust_match_result['matched_distance_m']
                        segment_features['matched_route_duration_s'] = robust_match_result['matched_duration_s']
                        segment_features['match_confidence'] = robust_match_result['match_confidence']
                        segment_features['match_status'] = robust_match_result['match_status']
                        segment_features['match_attempts'] = robust_match_result['match_attempts']
                        segment_features['final_match_radius_m'] = robust_match_result['final_radius_m']
                        osrm_match_response = robust_match_result.get('osrm_match_result')

                        # Calculate derived features if match was successful (even if tolerance failed)
                        if osrm_match_response:
                            start_lat = segment_features.get('lat_seg_start', np.nan)
                            start_lon = segment_features.get('lon_seg_start', np.nan)
                            end_lat = full_segment_slice[LAT_COL].iloc[-1] if LAT_COL in full_segment_slice.columns else np.nan
                            end_lon = full_segment_slice[LON_COL].iloc[-1] if LON_COL in full_segment_slice.columns else np.nan

                            # Direct Route Comparison
                            if pd.notna(start_lat) and pd.notna(start_lon) and pd.notna(end_lat) and pd.notna(end_lon):
                                direct_route = get_osrm_route([(start_lat, start_lon), (end_lat, end_lon)])
                                if direct_route:
                                    direct_dist_m = direct_route.get('distance', 0)
                                    segment_features['direct_route_distance_m'] = direct_dist_m
                                    if pd.notna(direct_dist_m) and direct_dist_m > 1e-3 and pd.notna(segment_features['matched_route_distance_m']):
                                        segment_features['route_deviation_pct'] = ((segment_features['matched_route_distance_m'] - direct_dist_m) / direct_dist_m) * 100
                                    else: segment_features['route_deviation_pct'] = 0.0

                            # Speed Limit Analysis
                            if not kinematics_df.empty:
                                speeding_results = analyze_speeding(kinematics_df, osrm_match_response)
                                segment_features['percent_time_over_limit_seg_agg'] = speeding_results['percent_time_over_limit_seg_agg']
                                segment_features['avg_speed_limit_kph_seg_agg'] = speeding_results['avg_speed_limit_kph_seg_agg']
                    else:
                        warnings.warn(f"[{trip_id}-{seg_id}] Skipping map matching due to invalid coords/timestamps or insufficient points.")
                        segment_features['match_status'] = 'InputError_PreCheck'


                # === 6. Predicted Segment Features (using Routing) ===
                # *** Use ACTUAL end point as proxy for estimated end point ***
                segment_features['distance_seg_pred_m'] = np.nan
                segment_features['duration_seg_pred_s'] = np.nan
                segment_features['speed_seg_pred_kph'] = np.nan
                start_lat = segment_features.get('lat_seg_start', np.nan)
                start_lon = segment_features.get('lon_seg_start', np.nan)
                end_lat = full_segment_slice[LAT_COL].iloc[-1] if LAT_COL in full_segment_slice.columns else np.nan
                end_lon = full_segment_slice[LON_COL].iloc[-1] if LON_COL in full_segment_slice.columns else np.nan

                if fetch_routing and pd.notna(start_lat) and pd.notna(start_lon) and pd.notna(end_lat) and pd.notna(end_lon):
                    predicted_route = get_osrm_route([(start_lat, start_lon), (end_lat, end_lon)])
                    if predicted_route:
                        segment_features['distance_seg_pred_m'] = predicted_route.get('distance', np.nan)
                        segment_features['duration_seg_pred_s'] = predicted_route.get('duration', np.nan)
                        if pd.notna(segment_features['distance_seg_pred_m']) and pd.notna(segment_features['duration_seg_pred_s']) and segment_features['duration_seg_pred_s'] > 1e-6:
                             pred_dist_km = segment_features['distance_seg_pred_m'] / 1000.0
                             pred_dur_h = segment_features['duration_seg_pred_s'] / 3600.0
                             segment_features['speed_seg_pred_kph'] = pred_dist_km / pred_dur_h
                        else: segment_features['speed_seg_pred_kph'] = np.nan
                            # Add predicted elevation if calculated separately
                            # elev_preds = get_route_elevation_change(start_lat, start_lon, end_lat, end_lon)
                            # segment_features['elevation_gain_seg_pred'] = elev_preds['predicted_elevation_gain']
                            # segment_features['elevation_loss_seg_pred'] = elev_preds['predicted_elevation_loss']
                    # else: warnings.warn(f"Skipping predictive routing for {trip_id}-{seg_id} due to invalid start/end coords.")

                # === 7, 8, 9: Global Prediction & Residual (Conditional) ===
                # === 7. Prepare Input & Get Global Prediction (CONDITIONAL) ===
                # Use _global_pred and _residual_error suffixes
                # ... (Ensure GLOBAL_MODEL_FEATURES list expects 'predicted_distance_m', etc.) ...

                segment_features['global_pred_soc_seg_delta'] = np.nan # Default if model not provided or fails
                segment_features['residual_error_soc_seg_delta'] = np.nan

                if trained_global_model is not None and global_features: # Check if model was loaded
                    global_model_input_dict = {}
                    missing_global_features = []
                    for feature_name in global_features: # global_features should use NEW names now
                        if feature_name in segment_features:
                            global_model_input_dict[feature_name] = segment_features[feature_name]
                        else:
                            warnings.warn(f"Global feature '{feature_name}' not found in calculated segment features for {trip_id}-{seg_id}. Setting to NaN.")
                            global_model_input_dict[feature_name] = np.nan
                            missing_global_features.append(feature_name)

                    X_global_pred_input = pd.DataFrame([global_model_input_dict], columns=global_features)


                # === 6. Get Global Prediction ===
                    if not X_global_pred_input.isnull().values.any():
                        try:
                            global_pred = trained_global_model.predict(X_global_pred_input)
                            segment_features['global_pred_soc_seg_delta'] = global_pred[0]
                        except Exception as pred_err:
                            warnings.warn(f"Global model prediction failed for {trip_id}-{seg_id}: {pred_err}.")
                    # else: warnings.warn(f"Skipping global prediction for {trip_id}-{seg_id} due to missing features.")

                # === 7. Calculate Residual (Target for Personal Model) ===
                # Ensure gt_soc_seg_delta exists before calculating residual
                    if pd.notna(segment_features['gt_soc_seg_delta']) and pd.notna(segment_features['global_pred_soc_seg_delta']):
                        segment_features['residual_error_soc_seg_delta'] = segment_features['gt_soc_seg_delta'] - segment_features['global_pred_soc_seg_delta']

                segment_features_list.append(dict(segment_features))
                
            except KeyError as e: warnings.warn(f"KeyError calculating features for {trip_id}-{seg_id}: Missing column {e}. Skipping.")
            except Exception as e: warnings.warn(f"Unexpected error for {trip_id}-{seg_id}: {e}. Skipping."); traceback.print_exc()

        processed_trips += 1
        if processed_trips % 50 == 0: print(f"   ... processed {processed_trips} trips (routing queries may add delay) ...") # Log more often due to routing

# --- Create Final DataFrame ---
    if not segment_features_list:
        print("Warning: No segment features were successfully extracted.")
        return pd.DataFrame()

    segments_df = pd.DataFrame(segment_features_list)
    print(f"\nFeature extraction complete. Created DataFrame with shape: {segments_df.shape}")

    if not segments_df.empty:
        # --- Attempt to Reorder Columns ---
        try:
            id_cols = ['trip_id', 'segment_id']
            existing_cols = segments_df.columns.tolist()

            start_cols = sorted([col for col in existing_cols if col.endswith('_seg_start') or col.startswith('start_')]) # Catch both old/new if mixed
            outcome_cols = sorted([col for col in existing_cols if col.endswith('_seg_end') or col.startswith('gt_') or col.endswith('_seg_actual_km') or col.endswith('_seg_actual_s')])
            # Updated seg_agg_cols to include new kinematics/stops
            seg_agg_cols = sorted([col for col in existing_cols if col.startswith(('speed_seg_agg', 'altitude_seg_agg', 'temp_seg_agg', 
                                                                                   'accel_seg_agg', 'decel_seg_agg', 'jerk_abs_seg_agg', 
                                                                                   'stops_seg_', 'stop_duration_seg_agg'))])
            static_cols = sorted([col for col in existing_cols if col.endswith('_trip') or col.endswith('_trip_kg') or col.endswith('_trip_start')])
            # Updated pred_cols to include matched route features and match status
            pred_cols = sorted([col for col in existing_cols if col.startswith(('global_pred', 'residual_error', 'predicted_', 'matched_route_', 'direct_route_', 'route_deviation', 'percent_time_over_limit', 'avg_speed_limit', 'match_', 'soc_delta_global_pred', 'soc_delta_residual_error'))]) # Added new pred/error names
            known_cols = set(id_cols + start_cols + outcome_cols + seg_agg_cols + static_cols + pred_cols)
            other_cols = sorted([col for col in existing_cols if col not in known_cols])
            final_col_order = id_cols + start_cols + static_cols + seg_agg_cols + outcome_cols + pred_cols + other_cols

            # Validation
            final_col_order_set = set(final_col_order); original_cols_set = set(existing_cols)
            if len(final_col_order) != len(existing_cols) or final_col_order_set != original_cols_set:
                 missing = original_cols_set - final_col_order_set; extra = final_col_order_set - original_cols_set
                 duplicates = pd.Series(final_col_order)[pd.Series(final_col_order).duplicated()].unique()
                 warnings.warn(f"Column reordering mismatch! Missing: {missing}, Extra: {extra}, Duplicates: {duplicates}. Keeping original order.")
                 final_col_order = existing_cols
            elif pd.Series(final_col_order).duplicated().any():
                 duplicates = pd.Series(final_col_order)[pd.Series(final_col_order).duplicated()].unique()
                 warnings.warn(f"Duplicate columns found in final_col_order construction: {duplicates}. Keeping original order.")
                 final_col_order = existing_cols

            segments_df = segments_df[final_col_order]
            print(f"Columns reordered for readability.")

        except Exception as e_reorder:
            warnings.warn(f"Could not reorder columns due to error: {e_reorder}. Keeping original order.")
            final_col_order = segments_df.columns.tolist()

        print(f"Final columns: {final_col_order}")
    return segments_df


def main():
    """Loads data, extracts segment features including driving behavior, saves results."""
    print("--- Feature Extraction Pipeline (Strategy 1 w/ Routing & Driving Behavior) ---")

    # Define the exact feature names the global model expects.
    # These MUST match columns calculated and stored in segment_features dictionary
    # by get_segments_features. They describe segment k itself or its start conditions.
    # *** This list MUST align with the features used when training the global model ***
    # *** WARNING: Using actual segment outcomes like 'distance_driven', 'duration_s', 'seg_speed_mean' ***
    # *** as features for the global model introduces leakage if the goal is true prediction ***
    # *** based only on start conditions. If the global model was trained this way, proceed, ***
    # *** but consider retraining the global model using only start-of-segment features ***
    # *** or features derived from segment k-1 as proxies for segment k properties. ***
    GLOBAL_MODEL_FEATURES = [
        # Start Conditions for Segment k
        'hour_seg_start',
        'dayofweek_seg_start',
        'month_seg_start',
        'lat_seg_start', # Use with caution or encode/bin
        'lon_seg_start',# Use with caution or encode/bin
        'temp_seg_start', # If available
        'battery_health_seg_start',
        'odo_seg_start',
        'soc_seg_start',
        #'start_dem_altitude', # If available
        #'start_dem_slope',    # If available
        #'start_dem_aspect',   # If available

        # Static Trip/Vehicle Features
        'car_model_trip',            # Needs encoding if not done before
        'manufacturer_trip',         # Needs encoding
        'weight_trip_kg',
        'battery_type_trip',         # Needs encoding

        # *** Predicted Segment k Properties (from Routing/DEM) ***
        'distance_seg_pred_m',
        'duration_seg_pred_s',
        'speed_seg_pred_kph',
        #'predicted_elevation_gain',  # If calculated
        #'predicted_elevation_loss',  # If calculated

        # DO NOT INCLUDE:
        # 'gt_soc_delta', 'distance_driven', 'duration_s', 'seg_speed_mean', etc. (Actual outcomes)
        # 'global_pred_soc_delta', 'global_prediction_error' (Outputs of this model/process)
        # Any 'k-1' or 'hist_' features (Belong to personal model)
    ] if CALCULATE_GLOBAL_PREDS else [] # Empty list if not calculating preds
    # -----------------------------------------------------------------

    print(f"\nInput cleaned data file: {CLEANED_DATA_PATH}")
    print(f"Input segmentation dict: {SEGMENTATION_DICT_PATH}")
    print(f"Input global model: {GLOBAL_MODEL_PATH}")
    print(f"Output features file: {OUTPUT_FEATURE_PATH}")

    # --- Load Global Model (Conditional) ---
    trained_global_model = None
    # --- Load Global Model (Conditional) ---
    trained_global_model = None
    if CALCULATE_GLOBAL_PREDS:
        print("\n--- Loading Trained Global Model ---")
        if not os.path.exists(GLOBAL_MODEL_PATH):
            warnings.warn(f"Model file not found at {GLOBAL_MODEL_PATH}. Global predictions will be NaN.")
            # No need to reset GLOBAL_MODEL_FEATURES here, it's already [] if needed
        else:
            try:
                trained_global_model = joblib.load(GLOBAL_MODEL_PATH)
                print(f"Loaded global model/pipeline from: {GLOBAL_MODEL_PATH}")
                if not hasattr(trained_global_model, 'predict'):
                     warnings.warn("Loaded object lacks .predict() method. Predictions will be NaN.")
                     trained_global_model = None # Invalidate model object
                     # Keep GLOBAL_MODEL_FEATURES as defined above, get_segments_features will handle None model
                elif not GLOBAL_MODEL_FEATURES: # Check if list is empty even if model loaded
                     warnings.warn("GLOBAL_MODEL_FEATURES list is empty, cannot generate predictions.")
                     trained_global_model = None # Invalidate model object
            except Exception as e:
                warnings.warn(f"Error loading global model: {e}. Global predictions will be NaN.")
                trained_global_model = None # Invalidate model object

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
        cycles_seg_dict = load_file(SEGMENTATION_DICT_PATH)
        if not isinstance(cycles_seg_dict, dict): raise TypeError("Segmentation data is not a dictionary.")
        print(f"Loaded segmentation dict with {len(cycles_seg_dict)} trips.")

        # 4. Filter Segmentation Dict for Min Segments
        df_clean_filtered, filtered_cycles_seg_dict = drop_trips_with_less_than_x_segs(
            df_clean, 
            cycles_seg_dict, 
            trip_id_col=TRIP_ID_COL, 
            min_segments=MIN_SEGMENTS_PER_TRIP
        )

        # 6. Extract Segment Features (Now includes map matching & behavior)
        segments_df = get_segments_features(
            df_clean_filtered,
            filtered_cycles_seg_dict,
            trained_global_model,
            GLOBAL_MODEL_FEATURES,
            fetch_routing=FETCH_ROUTING_FEATURES,
            osrm_url=OSRM_URL
        )

        # 7. Save Results
        if not segments_df.empty:
            print(f"\n--- Saving Segment Features ---")
            output_dir = os.path.dirname(OUTPUT_FEATURE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FEATURE_PATH))[0]
            save_file(data=segments_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT, index=False)
        else:
            print("\nWarning: Resulting segment features DataFrame is empty. Nothing to save.")

        print("\n--- Feature Extraction Pipeline Finished ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Feature Extraction Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()