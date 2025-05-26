import pandas as pd
import numpy as np
import os
import math
import warnings # To show warnings for skipped steps
import pickle # Needed by load_file for pickle format
import traceback # For detailed error printing
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
import h3
import traceback 
from datetime import datetime, timedelta # For time handling # For meteostat time range
# --- Import Meteostat ---
try:
    from meteostat import Stations, Point, Hourly
    # Suppress FutureWarning messages about the deprecated 'H' usage in meteostat/pandas
    warnings.filterwarnings("ignore", message="'H' is deprecated", category=FutureWarning)
    METEOSTAT_AVAILABLE = True
except ImportError:
    print("Warning: meteostat library not found. Weather comparison step will be skipped.")
    METEOSTAT_AVAILABLE = False

# Suppress specific FutureWarning messages from pandas/meteostat if needed
warnings.filterwarnings("ignore", message="'H' is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The behavior of DatetimeProperties.to_pydatetime is deprecated", category=FutureWarning)

# --- Import necessary functions from HelperFuncs ---
try:
    # Add safe_mean to the import list
    from HelperFuncs import load_file, haversine_distance, safe_mean,h3_to_latlon, save_file
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    # Define placeholders if needed for testing, but ideally fix the import
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")
    def haversine_distance(lat1, lon1, lat2, lon2): raise NotImplementedError("HelperFuncs not found")
    def safe_mean(val): raise NotImplementedError("HelperFuncs not found")
    def h3_to_latlon(h3_index): raise NotImplementedError("HelperFuncs not found")

# --- Configuration Block (Moved outside main for better visibility) ---
from cons import INPUT_FILE_PATH
from cons import CLEANED_DATA_PATH as OUTPUT_FILE_PATH
# from cons import LAND_SHAPEFILE_PATH

# --- Preprocessing Thresholds ---
from cons import MIN_TRIP_KM, MAX_TRIP_KM, MAX_GPS_JUMP_KM, MAX_CONSECUTIVE_SPEED_DIFF, MAX_CONSECUTIVE_ALT_DIFF
from cons import MAX_CONSECUTIVE_SOC_DIFF, MAX_CONSECUTIVE_TEMP_DIFF, MAX_CONSECUTIVE_TIME_GAP_SECONDS
from cons import MIN_VALID_SOC, MAX_VALID_SOC, MIN_VALID_TEMP, MAX_VALID_TEMP
from cons import MIN_VALID_ALTITUDE, MAX_VALID_ALTITUDE, MIN_VALID_BATT_HEALTH, MAX_VALID_BATT_HEALTH
from cons import MAX_VALID_ODO, MAX_VALID_WEIGHT, MIN_VALID_WEIGHT
from cons import SPEED_DIST_TOLERANCE_FACTOR, SOC_INCREASE_THRESHOLD


# Prprocessing parameters
from cons import FILTER_ZERO_COORDINATES, FILTER_MOSTLY_ZERO_TRIPS, ZERO_TRIP_THRESHOLD
#from cons import MAX_WEATHER_STATION_DIST_KM
from cons import MAX_TEMP_DIFFERENCE_C, FETCH_WEATHER_DATA, PROCESS_SUBSET

# --- Column Rename Mapping ---
from cons import COLUMN_RENAME_MAP

# --- Standard Column Names (used internally after renaming) ---
from cons import TRIP_ID_COL, TIME_COL, ODO_COL, SOC_COL, H3_COL
from cons import LAT_COL, LON_COL, SPEED_ARRAY_COL, ALT_ARRAY_COL, VEHICLE_TEMP_COL
from cons import BATT_HEALTH_COL, WEIGHT_COL, MSG_SESSION_ID_COL

# Columns expected constant per trip
from cons import STATIC_TRIP_COLS, DUPLICATE_CHECK_COLS_STD



# --- Individual Preprocessing Functions (Updated) ---

def filter_inconsistent_trip_data(df, trip_id_col, cols_to_check):
    """Filters out trips where static columns have multiple unique values."""
    print(f"\n--- Checking Trip Data Consistency (Static Cols) ---")
    if trip_id_col not in df.columns:
        print(f"Warning: Trip ID column '{trip_id_col}' not found. Skipping check.")
        return df
    cols_present = [col for col in cols_to_check if col in df.columns]
    if not cols_present:
        print("Warning: None of the specified static columns exist. Skipping check.")
        return df
    print(f" - Checking consistency for: {cols_present}")
    try:
        nunique_counts = df.groupby(trip_id_col)[cols_present].nunique(dropna=True)
    except Exception as e:
         print(f"Warning: Error during groupby/nunique check. Skipping. Error: {e}")
         return df
    inconsistent_trip_mask = (nunique_counts > 1).any(axis=1)
    trip_ids_to_drop = nunique_counts.index[inconsistent_trip_mask]
    if not trip_ids_to_drop.empty:
        num_to_drop = len(trip_ids_to_drop)
        print(f"Found {num_to_drop} trips with inconsistent static values.")
        max_details_to_print = 10
        inconsistent_details = nunique_counts[inconsistent_trip_mask]
        for i, trip_id in enumerate(trip_ids_to_drop):
            if i >= max_details_to_print: print(f"   ... (details omitted for remaining trips)"); break
            problem_cols = inconsistent_details.loc[trip_id][inconsistent_details.loc[trip_id] > 1].index.tolist()
            print(f" - Trip ID: {trip_id}, Inconsistent Columns: {problem_cols}")
        df_filtered = df[~df[trip_id_col].isin(trip_ids_to_drop)].copy()
        print(f"Result shape after removing inconsistent trips: {df_filtered.shape}")
        return df_filtered
    else:
        print("No inconsistent trips found based on static columns.")
        print(f"Result shape: {df.shape}")
        return df

def remove_duplicates(df, subset_cols):
    """Remove duplicate rows based on a specified subset of standard columns."""
    print("\n--- Removing Duplicate Rows ---")
    subset_cols_present = [col for col in subset_cols if col in df.columns]
    if not subset_cols_present:
        print("Warning: No standard columns found for duplicate check. Skipping.")
        return df
    initial_rows = len(df)
    try:
        df = df.drop_duplicates(subset=subset_cols_present, keep='first')
        print(f"Result shape: {df.shape} (Removed {initial_rows - len(df)} rows based on {subset_cols_present})")
    except TypeError as e:
        print(f"Warning: TypeError during drop_duplicates. Skipping. Error: {e}")
    return df

def check_and_drop_correlated_id(df, primary_id_col, secondary_id_col):
    """Checks if secondary ID is redundant w.r.t primary ID and drops if true."""
    print(f"\n--- Checking Correlation: '{primary_id_col}' vs '{secondary_id_col}' ---")
    if not all(col in df.columns for col in [primary_id_col, secondary_id_col]):
        print(f"Warning: Missing one or both ID columns. Skipping check.")
        return df
    try:
        nunique_per_primary = df.groupby(primary_id_col)[secondary_id_col].nunique(dropna=True)
        is_correlated = (nunique_per_primary <= 1).all()
        if is_correlated:
            print(f"'{secondary_id_col}' is redundant. Dropping.")
            df_filtered = df.drop(columns=[secondary_id_col])
            print(f"Result shape: {df_filtered.shape}")
            return df_filtered
        else:
            inconsistent_groups = nunique_per_primary[nunique_per_primary > 1]
            print(f"'{secondary_id_col}' NOT redundant ({len(inconsistent_groups)} groups have >1 value). Keeping.")
            print(f"Result shape: {df.shape}")
            return df
    except Exception as e:
        print(f"Warning: Error during correlation check. Skipping drop. Error: {e}")
        return df

def remove_constant_cols(df):
    """Remove constant or all-NaN columns."""
    const_cols_to_drop = []
    print("\n--- Checking for Constant/All-NaN Columns ---")
    for col in df.columns:
        # Skip array/list columns
        if df[col].iloc[0:min(100, len(df))].apply(lambda x: isinstance(x, (list, np.ndarray))).any(): # Check sample
            # print(f" - Skipping constant check for array/list column '{col}'.")
            continue
        try:
            num_unique = df[col].nunique(dropna=False)
            if num_unique == 1:
                first_valid_index = df[col].first_valid_index()
                if first_valid_index is not None: const_value = df[col].loc[first_valid_index]
                elif df[col].isnull().all(): const_value = "All NaN"
                else: const_value = "Unknown Constant"
                const_value_str = str(const_value);
                if len(const_value_str) > 70: const_value_str = const_value_str[:67] + '...'
                print(f" - Found constant '{col}': {const_value_str}")
                const_cols_to_drop.append(col)
        except TypeError: pass # print(f" - Skipping constant check for column '{col}' due to TypeError.")
        except Exception as e: print(f" - Error checking uniqueness for column '{col}': {e}. Skipping.")
    if const_cols_to_drop:
        print(f"Removing constant/all-NaN columns: {const_cols_to_drop}")
        df = df.drop(columns=const_cols_to_drop)
    else: print("No constant value or all-NaN columns found/checked to remove.")
    print(f"Result shape: {df.shape}")
    return df

def validate_and_filter_latlon(df, lat_col='latitude', lon_col='longitude', trip_id_col='trip_id',
                               filter_zeros=True, zero_tolerance=1e-3,
                               filter_mostly_zero_trips=True, zero_trip_threshold=0.95,
                               max_trips_to_print=10):
    """
    Performs sanity checks and filtering on latitude and longitude columns:
    1. Checks for NaN values.
    2. Filters rows with values outside valid ranges [-90, 90] for lat, [-180, 180] for lon.
    3. Optionally analyzes and filters rows where both lat/lon are close to zero.
    4. Optionally filters *entire trips* where the proportion of near-(0,0) points
       exceeds a threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        trip_id_col (str): Name of the trip identifier column.
        filter_zeros (bool): If True, filter out individual rows where both lat/lon are near zero.
        zero_tolerance (float): Tolerance for checking if coordinates are close to zero.
        filter_mostly_zero_trips (bool): If True, filter entire trips consisting mostly of zero points.
        zero_trip_threshold (float): Proportion (0 to 1) of zero points required to filter a trip.
        max_trips_to_print (int): Max number of trip IDs with issues to print.

    Returns:
        pd.DataFrame: DataFrame with invalid rows and/or trips potentially removed.
    """
    print(f"\n--- Validating and Filtering Lat/Lon ('{lat_col}', '{lon_col}') ---")
    required_cols = [lat_col, lon_col, trip_id_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}). Skipping validation.")
        return df

    initial_rows = len(df)
    df_filtered = df.copy() # Work on a copy

    # --- 1. Check for NaNs ---
    nan_lat_count = df_filtered[lat_col].isna().sum()
    nan_lon_count = df_filtered[lon_col].isna().sum()
    if nan_lat_count > 0 or nan_lon_count > 0:
        print(f" - Found {nan_lat_count} NaN values in '{lat_col}'.")
        print(f" - Found {nan_lon_count} NaN values in '{lon_col}'.")
        # Drop NaNs before range/zero checks as they interfere
        df_filtered = df_filtered.dropna(subset=[lat_col, lon_col])
        print(f"   Dropped {initial_rows - len(df_filtered)} rows with NaN lat/lon.")
        if df_filtered.empty:
             print("DataFrame empty after dropping NaN lat/lon.")
             return df_filtered

    # Ensure columns are numeric after dropping NaNs
    df_filtered[lat_col] = pd.to_numeric(df_filtered[lat_col], errors='coerce')
    df_filtered[lon_col] = pd.to_numeric(df_filtered[lon_col], errors='coerce')
    # Check again if coercion created NaNs (shouldn't if previous dropna worked)
    df_filtered = df_filtered.dropna(subset=[lat_col, lon_col])
    if df_filtered.empty:
        print("DataFrame empty after numeric conversion/dropna.")
        return df_filtered

    # --- 2. Check Valid Range ---
    invalid_lat_mask = (df_filtered[lat_col] < -90) | (df_filtered[lat_col] > 90)
    invalid_lon_mask = (df_filtered[lon_col] < -180) | (df_filtered[lon_col] > 180)
    invalid_range_mask = invalid_lat_mask | invalid_lon_mask
    num_invalid_range = invalid_range_mask.sum()
    if num_invalid_range > 0:
        print(f" - Found {num_invalid_range} rows with lat/lon outside valid ranges.")
        df_filtered = df_filtered[~invalid_range_mask] # Filter out invalid ranges
        print(f"   Dropped {num_invalid_range} rows due to invalid ranges.")
    else:
        print(" - All lat/lon values are within valid ranges.")

    if df_filtered.empty:
        print("DataFrame empty after range filtering.")
        return df_filtered

    # --- 3. Analyze and Optionally Filter Near-(0,0) Points/Trips ---
    if filter_zeros or filter_mostly_zero_trips:
        zero_lat_mask = np.isclose(df_filtered[lat_col], 0.0, atol=zero_tolerance)
        zero_lon_mask = np.isclose(df_filtered[lon_col], 0.0, atol=zero_tolerance)
        zero_zero_mask = zero_lat_mask & zero_lon_mask
        num_zero_zero = zero_zero_mask.sum()

        if num_zero_zero > 0:
            print(f"\n - Found {num_zero_zero} rows with lat/lon close to (0, 0) (tolerance={zero_tolerance}). Analyzing distribution...")
            zero_points_df = df_filtered.loc[zero_zero_mask, [trip_id_col]]
            unique_trips_with_zeros = zero_points_df[trip_id_col].unique()
            num_unique_trips_with_zeros = len(unique_trips_with_zeros)
            print(f"   These points belong to {num_unique_trips_with_zeros} unique trip(s).")

            # --- Check for trips to filter entirely ---
            trips_to_drop_entirely = set()
            if filter_mostly_zero_trips:
                total_points_per_trip = df_filtered[trip_id_col].value_counts()
                zero_points_per_trip = zero_points_df[trip_id_col].value_counts()
                trip_counts = pd.merge(
                    total_points_per_trip.rename('total_count'),
                    zero_points_per_trip.rename('zero_count'),
                    left_index=True, right_index=True, how='left'
                ).fillna({'zero_count': 0})
                trip_counts['zero_ratio'] = trip_counts['zero_count'] / trip_counts['total_count']

                mostly_zero_mask = trip_counts['zero_ratio'] > zero_trip_threshold
                trips_to_drop_entirely = set(trip_counts.index[mostly_zero_mask])
                num_trips_mostly_zeros = len(trips_to_drop_entirely)

                if num_trips_mostly_zeros > 0:
                    print(f"   Identified {num_trips_mostly_zeros} trip(s) where >{zero_trip_threshold*100:.0f}% of points are near (0,0). These trips will be removed entirely.")
                    #print(f"     Example Trip IDs (up to {max_trips_to_print}): {list(trips_to_drop_entirely)[:max_trips_to_print]}" + ("..." if num_trips_mostly_zeros > max_trips_to_print else ""))
                    # Filter out these trips
                    df_filtered = df_filtered[~df_filtered[trip_id_col].isin(trips_to_drop_entirely)]
                    print(f"   Shape after removing mostly-zero trips: {df_filtered.shape}")
                else:
                    print(f"   No trips found exceeding the {zero_trip_threshold*100:.0f}% threshold for near-(0,0) points.")

            # --- Filter individual zero points (if enabled and not already dropped via trip filter) ---
            if filter_zeros:
                # Identify zero points *remaining* after potential trip filtering
                zero_lat_mask_remaining = np.isclose(df_filtered[lat_col], 0.0, atol=zero_tolerance)
                zero_lon_mask_remaining = np.isclose(df_filtered[lon_col], 0.0, atol=zero_tolerance)
                zero_zero_mask_remaining = zero_lat_mask_remaining & zero_lon_mask_remaining
                num_zero_zero_remaining = zero_zero_mask_remaining.sum()

                if num_zero_zero_remaining > 0:
                    # Only filter if not already handled by trip removal
                    if not filter_mostly_zero_trips or num_trips_mostly_zeros == 0:
                         print(f"   Filtering {num_zero_zero_remaining} individual rows near (0, 0).")
                         df_filtered = df_filtered[~zero_zero_mask_remaining]
                    else:
                         # Check if any zero points remain *after* trip filtering
                         # This can happen if filter_mostly_zero_trips is True but some trips had < threshold % zeros
                         zero_points_in_kept_trips = df_filtered.loc[zero_zero_mask_remaining, trip_id_col].nunique()
                         if zero_points_in_kept_trips > 0:
                             print(f"   Filtering {num_zero_zero_remaining} individual near-(0,0) rows from trips below the {zero_trip_threshold*100:.0f}% threshold.")
                             df_filtered = df_filtered[~zero_zero_mask_remaining]
                         else:
                              print("   All near-(0,0) points were part of the removed trips.")

                else:
                    print("   No individual near-(0,0) rows remaining to filter.")
            else:
                 print("   Skipping filtering of individual near-(0,0) rows.")

        else: # num_zero_zero == 0 initially
            print(f" - No rows found with lat/lon close to (0, 0) (tolerance={zero_tolerance}).")
    else:
         print(" - Skipping checks related to (0, 0) coordinates.")


    rows_dropped_total = initial_rows - len(df_filtered)
    print(f"\nResult shape after Lat/Lon validation: {df_filtered.shape} (Removed {rows_dropped_total} rows total)")
    return df_filtered

# --- Weather Fetching Function ---
def get_weather_for_timestamp(row, lat_col='latitude', lon_col='longitude', time_col='timestamp', h3_col='current_location', max_station_dist_km=50.0):
    """
    Fetches hourly weather data for a specific row's timestamp and location,
    adds prefixes to weather columns, and includes nearest station info.

    Returns a dictionary suitable for use with apply(..., result_type='expand').
    Keys are prefixed with 'weather_' or 'station_'.
    """
    # Initialize result with NaNs for all potential weather/station columns
    result = {
        'weather_temp': np.nan, 'weather_dwpt': np.nan, 'weather_rhum': np.nan,
        'weather_prcp': np.nan, 'weather_snow': np.nan, 'weather_wdir': np.nan,
        'weather_wspd': np.nan, 'weather_wpgt': np.nan, 'weather_pres': np.nan,
        'weather_tsun': np.nan, 'weather_coco': np.nan,
        'station_id': None, 'station_dist_km': np.nan
    }

    # --- Get Lat/Lon ---
    lat, lon = np.nan, np.nan
    # Prioritize existing lat/lon columns if they exist
    if lat_col in row.index and lon_col in row.index and not pd.isna(row[lat_col]) and not pd.isna(row[lon_col]):
         lat, lon = row[lat_col], row[lon_col]
    # Fallback to H3 conversion if lat/lon are missing/NaN but H3 exists
    elif h3_col in row.index and not pd.isna(row[h3_col]):
        try:
            lat, lon = h3_to_latlon(row[h3_col]) # Use the H3 conversion helper
        except Exception as e:
            # warnings.warn(f"H3 conversion failed for index {row.name}: {e}")
            pass # Keep lat/lon as NaN

    # --- Get Timestamp ---
    timestamp = pd.NaT
    if time_col in row.index and not pd.isna(row[time_col]):
        timestamp = pd.to_datetime(row[time_col], errors='coerce')

    if pd.isna(lat) or pd.isna(lon) or pd.isna(timestamp):
        return pd.Series(result) # Return Series of NaNs if essential info missing

    # --- Get Nearest Station Info (Optional but recommended) ---
    try:
        # Ensure lat/lon are float for Meteostat
        lat, lon = float(lat), float(lon)
        pt = Point(lat, lon)
        # Find nearby stations (increase count slightly in case closest has no data)
        stations = Stations().nearby(lat, lon)
        station_df = stations.fetch(1) # Fetch the closest one

        if not station_df.empty:
            station_info = station_df.iloc[0]
            station_id = station_info.name
            station_lat = station_info['latitude']
            station_lon = station_info['longitude']
            # Use try-except for haversine, just in case
            try:
                distance = haversine_distance(lat, lon, station_lat, station_lon)
                result['station_id'] = station_id
                result['station_dist_km'] = distance
                # Optional check: Skip if station too far?
                # if distance > max_station_dist_km: return pd.Series(result)
            except Exception as e_dist:
                 print(f"Warning: Error calculating distance for index {row.name}: {e_dist}")

        # else: print(f"Warning: No nearby station found for index {row.name}")

    except Exception as e_stat:
        print(f"Warning: Error finding nearest station for index {row.name}: {e_stat}")

    # --- Fetch Hourly Weather Data ---
    try:
        start_hour = timestamp.floor('H')
        end_hour = start_hour # Fetch data for just that hour

        # Fetch data using the Point defined earlier
        weather_df = Hourly(pt, start_hour, end_hour).fetch()

        if not weather_df.empty:
            weather_data = weather_df.iloc[0] # Get the data for the hour
            # Populate result dict, adding prefix
            for col in result.keys():
                 meteo_col = col.replace('weather_', '')
                 if meteo_col in weather_data and pd.notna(weather_data[meteo_col]):
                      result[col] = weather_data[meteo_col]
        # else: print(f"Warning: No hourly weather data found for index {row.name} at {start_hour}")

    except Exception as e_fetch:
        print(f"Warning: Error fetching hourly weather for index {row.name} at {start_hour}: {e_fetch}")

    return pd.Series(result) # Return as a Series for apply


# --- Temperature Comparison Function ---
def flag_suspicious_temperatures(df, vehicle_temp_col='outside_temp', weather_temp_col='weather_temp',
                                 station_dist_col='station_dist_km', # New argument
                                 max_difference_c=15.0, flag_col_name='temp_suspicious_flag'):
    """
    Compares vehicle temperature to external weather temperature and flags large discrepancies.
    Includes station distance in the output for flagged rows.

    Args:
        df (pd.DataFrame): Input DataFrame (must contain temperature and distance columns).
        vehicle_temp_col (str): Name of the vehicle's temperature column.
        weather_temp_col (str): Name of the fetched weather temperature column.
        station_dist_col (str): Name of the column containing distance to the nearest station (km).
        max_difference_c (float): Maximum allowable absolute difference in Celsius.
        flag_col_name (str): Name for the new boolean flag column.

    Returns:
        pd.DataFrame: DataFrame with the added flag column.
    """
    print(f"\n--- Flagging Suspicious Temperatures (|Diff| > {max_difference_c}°C) ---")
    # Check required temperature columns
    required_temp_cols = [vehicle_temp_col, weather_temp_col]
    if not all(col in df.columns for col in required_temp_cols):
        print(f"Warning: Missing required temperature columns ({required_temp_cols}). Skipping check.")
        df[flag_col_name] = False
        return df

    # Check if station distance column exists for reporting (optional)
    has_station_dist = station_dist_col in df.columns
    if not has_station_dist:
        print(f"Warning: Station distance column '{station_dist_col}' not found. Discrepancy examples will not include distance.")

    # Ensure temps are numeric
    df[vehicle_temp_col] = pd.to_numeric(df[vehicle_temp_col], errors='coerce')
    df[weather_temp_col] = pd.to_numeric(df[weather_temp_col], errors='coerce')
    if has_station_dist:
        df[station_dist_col] = pd.to_numeric(df[station_dist_col], errors='coerce')


    # Calculate difference where both are available
    df['temp_difference'] = df[vehicle_temp_col] - df[weather_temp_col]

    # Create flag: True if difference is valid and exceeds threshold
    df[flag_col_name] = (df['temp_difference'].abs() > max_difference_c) & df['temp_difference'].notna()

    num_flagged = df[flag_col_name].sum()
    num_compared = df['temp_difference'].notna().sum()

    if num_compared > 0:
        print(f" - Compared {num_compared} rows with external temperature data.")
        print(f" - Flagged {num_flagged} rows as suspicious (difference > {max_difference_c}°C).")
        if num_flagged > 0:
            # Prepare columns to display in examples
            example_cols = [vehicle_temp_col, weather_temp_col, 'temp_difference']
            if has_station_dist:
                example_cols.append(station_dist_col) # Add distance if available

            print(f"   Example discrepancies ({', '.join(example_cols)}):")
            # Show rows where the flag is True, selecting the example columns
            print(df.loc[df[flag_col_name], example_cols].head(20))
    else:
        print(" - No external temperature data available for comparison.")

    # Optionally drop the temporary difference column
    # df = df.drop(columns=['temp_difference'])

    print(f"Result shape (with flag added): {df.shape}")
    return df

def filter_gps_at_sea(df, land_geom, lat_col='latitude', lon_col='longitude', max_sea_locations_to_print=20):
    """
    Filters out rows where GPS coordinates fall outside the provided land geometry.
    Reports counts of unique locations identified as being at sea.

    Args:
        df (pd.DataFrame): Input DataFrame with latitude and longitude columns.
        land_geom (shapely.geometry.base.BaseGeometry): Unified Shapely geometry for land.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        max_sea_locations_to_print (int): Max number of unique sea locations to print details for.


    Returns:
        pd.DataFrame: DataFrame with rows outside the land geometry removed.
    """
    print("\n--- Filtering GPS Points At Sea ---")
    required_cols = [lat_col, lon_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}). Skipping filter.")
        return df
    if land_geom is None:
        print("Warning: Land geometry not provided. Skipping filter.")
        return df

    initial_rows = len(df)
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df_to_check = df.dropna(subset=[lat_col, lon_col]).copy()
    rows_dropped_nan = initial_rows - len(df_to_check)
    if rows_dropped_nan > 0:
        print(f" - Temporarily ignoring {rows_dropped_nan} rows with invalid/missing GPS coordinates for this check.")

    if df_to_check.empty:
        print(" - No valid GPS coordinates remaining to check.")
        print(f"Result shape: {df.shape}")
        return df

    try:
        geometry = gpd.points_from_xy(df_to_check[lon_col], df_to_check[lat_col])
        gdf = gpd.GeoDataFrame(df_to_check, geometry=geometry, crs="EPSG:4326")

        print(f" - Performing spatial check against land geometry...")
        is_on_land_mask = gdf.within(land_geom)
        print(f" - Spatial check complete.")

        # --- Identify points AT SEA ---
        at_sea_mask = ~is_on_land_mask
        at_sea_indices = gdf.index[at_sea_mask]
        num_at_sea = len(at_sea_indices)

        if num_at_sea > 0:
            print(f" - Found {num_at_sea} points determined to be at sea.")

            # --- Count unique locations at sea ---
            at_sea_coords = df_to_check.loc[at_sea_indices, [lat_col, lon_col]]
            # Round coordinates slightly to group nearby error points if desired
            # e.g., at_sea_coords = at_sea_coords.round(4)
            unique_sea_locations = at_sea_coords.value_counts().reset_index(name='count')
            num_unique_sea_locations = len(unique_sea_locations)
            print(f" - These points correspond to {num_unique_sea_locations} unique lat/lon locations.")

            # Print details of the most common sea locations
            print(f"   Most common locations identified as 'at sea' (up to {max_sea_locations_to_print}):")
            # Sort by count descending
            unique_sea_locations = unique_sea_locations.sort_values('count', ascending=False)
            for i, row in enumerate(unique_sea_locations.head(max_sea_locations_to_print).itertuples()):
                 print(f"     Lat: {row.latitude:<12.6f} Lon: {row.longitude:<12.6f} Count: {row.count}")
            if num_unique_sea_locations > max_sea_locations_to_print:
                 print(f"     ... ({num_unique_sea_locations - max_sea_locations_to_print} more unique locations omitted)")
            # ------------------------------------

        else:
             print(" - No points determined to be at sea.")


        # --- Filter the DataFrame ---
        # Keep rows that are ON LAND or had initial NaN coordinates
        land_indices = gdf.index[is_on_land_mask]
        original_indices_to_keep = df.index[df[lat_col].isna() | df[lon_col].isna()]
        final_indices_to_keep = land_indices.union(original_indices_to_keep)
        df_filtered = df.loc[final_indices_to_keep].copy()
        rows_dropped_total = initial_rows - len(df_filtered)

        print(f"\nResult shape after filtering points at sea: {df_filtered.shape} (Removed {rows_dropped_total} rows total)")
        return df_filtered

    except ImportError:
        print("Warning: GeoPandas or Shapely not installed. Skipping GPS land check.")
        return df
    except Exception as e:
        print(f"Warning: Error during GPS land check: {e}. Skipping filter.")
        traceback.print_exc()
        return df

def filter_unrealistic_ranges(df, range_checks, max_values_to_print=10):
    """
    Filters rows where values fall outside specified min/max ranges.
    Prints details about the out-of-range values found.

    Args:
        df (pd.DataFrame): The input DataFrame.
        range_checks (dict): Dictionary mapping column names to (min_val, max_val) tuples.
        max_values_to_print (int): Max number of out-of-range values to print per column.

    Returns:
        pd.DataFrame: DataFrame with rows containing out-of-range values removed.
    """
    print("\n--- Filtering Unrealistic Sensor Ranges ---")
    df_filtered = df.copy() # Work on a copy to avoid modifying original during checks
    initial_rows = len(df_filtered)
    total_rows_dropped = 0
    temp_cols_created = [] # Keep track of temporary columns

    for col, (min_val, max_val) in range_checks.items():
        if col not in df_filtered.columns:
            print(f"Warning: Column '{col}' not found for range check. Skipping.")
            continue

        print(f"\n - Checking range for '{col}' ({min_val} to {max_val}).")
        col_to_check = col # Default to original column
        is_array_like = df_filtered[col].iloc[0:min(100, len(df_filtered))].apply(lambda x: isinstance(x, (list, np.ndarray))).any()

        # --- Prepare column for checking (handle arrays, ensure numeric) ---
        if is_array_like:
            print(f"   (Applying range check to mean of array column '{col}')")
            temp_mean_col = f"{col}_mean_temp_range_check"
            temp_cols_created.append(temp_mean_col)
            df_filtered[temp_mean_col] = df_filtered[col].apply(safe_mean)
            col_to_check = temp_mean_col
        else:
            # Ensure scalar column is numeric, store original before coercion for potential printing
            original_values = df_filtered[col].copy()
            df_filtered[col_to_check] = pd.to_numeric(df_filtered[col], errors='coerce')

        if col_to_check not in df_filtered.columns:
             print(f"   Warning: Could not perform range check for '{col}' (column or mean calculation failed).")
             continue

        # --- Identify and Print Out-of-Range Values ---
        # Condition: Value is NOT NaN AND (Value < min_val OR Value > max_val)
        out_of_range_mask = (
            df_filtered[col_to_check].notna() &
            ((df_filtered[col_to_check] < min_val) | (df_filtered[col_to_check] > max_val))
        )
        num_out_of_range = out_of_range_mask.sum()

        if num_out_of_range > 0:
            print(f"   Found {num_out_of_range} out-of-range values for '{col}'.")
            # Get the actual out-of-range values (from the column used for checking)
            out_of_range_values = df_filtered.loc[out_of_range_mask, col_to_check]

            # If checking mean of array, maybe show original array too? More complex.
            # For now, just show the value that failed the check (mean or scalar)
            values_to_print = out_of_range_values.unique() # Show unique bad values found
            print(f"     Unique out-of-range values found (up to {max_values_to_print}): ", end="")
            print(f"{values_to_print[:max_values_to_print].tolist()}" + ("..." if len(values_to_print) > max_values_to_print else ""))

            # --- Filter the DataFrame ---
            # Keep rows where value is within range OR is NaN
            df_filtered = df_filtered[~out_of_range_mask]
            print(f"   Dropped {num_out_of_range} rows.")
            total_rows_dropped += num_out_of_range
        else:
            print(f"   No out-of-range values found for '{col}'.")

    # --- Clean up temporary columns ---
    df_filtered = df_filtered.drop(columns=temp_cols_created, errors='ignore')

    print(f"\nResult shape after range checks: {df_filtered.shape} (Removed {total_rows_dropped} rows total)")
    return df_filtered

# --- NEW: Negative Value Filtering Function ---
def filter_negative_values_extended(df, cols_to_check):
    """
    Filters rows where specified columns have negative values.
    Provides more detailed output on which columns caused rows to be dropped.
    """
    print("\n--- Filtering Negative Values ---")
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Input DataFrame is empty or invalid. Skipping.")
        return df

    initial_rows = len(df)
    # Keep track of rows to drop based on checks for each column
    rows_to_drop_indices = set()
    # Keep track of how many rows are dropped due to each column check
    drop_counts_per_col = {col: 0 for col in cols_to_check if col in df.columns}

    for col in cols_to_check:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found for negative value check. Skipping.")
            continue

        print(f"\n - Checking non-negativity for '{col}':")
        is_array_like = df[col].iloc[0:min(100, len(df))].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
        col_to_evaluate = None # Series holding the values to check (original or mean)

        try:
            if is_array_like:
                print(f"   (Applying check to mean of array column '{col}')")
                col_to_evaluate = df[col].apply(safe_mean)
            else:
                # Ensure scalar column is numeric
                col_to_evaluate = pd.to_numeric(df[col], errors='coerce')

            if col_to_evaluate is None: # Should not happen unless apply fails silently
                 print(f"   Warning: Could not evaluate values for '{col}'. Skipping column.")
                 continue

            # Identify rows where the evaluated value is negative (and not NaN)
            negative_mask = (col_to_evaluate < 0) & col_to_evaluate.notna()
            num_negative = negative_mask.sum()

            if num_negative > 0:
                print(f"   Found {num_negative} negative value(s) in '{col}'.")
                # Add the indices of these rows to the set of rows to be dropped
                rows_to_drop_indices.update(df.index[negative_mask])
                drop_counts_per_col[col] = num_negative # Store count for this column
            else:
                 print(f"   No negative values found in '{col}'.")

        except Exception as e:
            print(f"Warning: Error during negativity check for '{col}'. Skipping column. Error: {e}")

    # Filter the DataFrame by removing all identified rows at once
    if rows_to_drop_indices:
        num_unique_rows_to_drop = len(rows_to_drop_indices)
        print(f"\nSummary of Negative Value Filtering:")
        for col, count in drop_counts_per_col.items():
            if count > 0:
                print(f" - Column '{col}': {count} negative value(s) identified.")
        print(f"Total unique rows to be removed due to negative values: {num_unique_rows_to_drop}")
        df_filtered = df.drop(index=list(rows_to_drop_indices))
    else:
        print("\nNo rows needed removal based on negative value checks.")
        df_filtered = df # No changes needed

    print(f"\nResult shape after negative value checks: {df_filtered.shape} (Removed {initial_rows - len(df_filtered)} rows total)")
    return df_filtered

# --- (Keep and potentially enhance existing direction/difference filters) ---
# Example Enhancements:
# - filter_non_increasing_timestamps could also check for MAX_CONSECUTIVE_TIME_GAP_SECONDS
# - filter_big_differences could be enhanced to calculate odo_diff and gps_dist if needed

def filter_non_increasing_timestamps(df, trip_id_col, time_col, max_gap_seconds=None):
    """Filters non-increasing timestamps and optionally large time gaps."""
    print(f"\n--- Filtering Timestamp Sequence (Max Gap={max_gap_seconds}s) ---")
    if not all(col in df.columns for col in [trip_id_col, time_col]):
        print(f"Warning: Missing required columns. Skipping.")
        return df
    df = df.sort_values([trip_id_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=[time_col], inplace=True)
    if len(df) < initial_rows: print(f" - Dropped {initial_rows - len(df)} rows with invalid timestamps.")
    if df.empty: return df

    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
         df['time_diff_sec'] = df.groupby(trip_id_col)[time_col].diff().dt.total_seconds()
    else:
         print(f"Warning: Column '{time_col}' not datetime type.")
         df['time_diff_sec'] = df.groupby(trip_id_col)[time_col].diff()

    # Filter 1: Non-increasing timestamps
    valid_mask = (df['time_diff_sec'] > 0) | df['time_diff_sec'].isna()
    rows_dropped_noninc = (~valid_mask).sum()
    if rows_dropped_noninc > 0: print(f" - Dropped {rows_dropped_noninc} rows due to non-increasing timestamps.")

    # Filter 2: Large time gaps (optional)
    rows_dropped_gap = 0
    if max_gap_seconds is not None:
        gap_mask = (df['time_diff_sec'] <= max_gap_seconds) | df['time_diff_sec'].isna()
        rows_dropped_gap = (~gap_mask & valid_mask).sum() # Count only those not already dropped
        if rows_dropped_gap > 0: print(f" - Dropped {rows_dropped_gap} rows due to time gap > {max_gap_seconds}s.")
        valid_mask &= gap_mask

    df_filtered = df[valid_mask].drop(columns=['time_diff_sec'])
    print(f"Result shape: {df_filtered.shape}")
    return df_filtered

# --- NEW: Speed-Distance Consistency Check ---
def filter_speed_distance_inconsistency(df, trip_id_col, time_col, odo_col, lat_col, lon_col, speed_col, tolerance_factor=1.5):
    """Filters rows where distance covered seems inconsistent with speed and time."""
    print(f"\n--- Filtering Speed/Distance Inconsistency (Tolerance={tolerance_factor}) ---")
    required_cols = [trip_id_col, time_col, odo_col, lat_col, lon_col, speed_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}). Skipping check.")
        return df

    df = df.sort_values([trip_id_col, time_col])

    # Calculate time diff (ensure timestamp is datetime)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['time_diff_sec_consist'] = df.groupby(trip_id_col)[time_col].diff().dt.total_seconds()

    # Calculate odo diff
    df[odo_col] = pd.to_numeric(df[odo_col], errors='coerce')
    df['odo_diff_consist'] = df.groupby(trip_id_col)[odo_col].diff()

    # Calculate GPS dist
    df['prev_lat_consist'] = df.groupby(trip_id_col)[lat_col].shift(1)
    df['prev_lon_consist'] = df.groupby(trip_id_col)[lon_col].shift(1)
    gps_distances = []
    for _, row in df.iterrows():
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]) or pd.isna(row['prev_lat_consist']) or pd.isna(row['prev_lon_consist']):
            gps_distances.append(np.nan) # Can't calculate distance if coords missing
        else:
            try: gps_distances.append(haversine_distance(row[lat_col], row[lon_col], row['prev_lat_consist'], row['prev_lon_consist']))
            except: gps_distances.append(np.nan)
    df['gps_dist_consist'] = gps_distances
    df['gps_dist_consist'] = df['gps_dist_consist'].fillna(0) # Fill NaN distances with 0 for comparison

    # Calculate mean speed in m/s
    df['mean_speed_mps_consist'] = df[speed_col].apply(safe_mean) * (1000/3600) # KPH to m/s

    # Calculate max plausible distance based on speed and time
    # Use average speed between current and previous point for better estimate? Simpler: use current mean speed.
    df['max_plausible_dist_m'] = df['mean_speed_mps_consist'] * df['time_diff_sec_consist'] * tolerance_factor * 1000 # Convert to km

    # Define inconsistency mask
    # Inconsistent if odo diff > max plausible OR gps dist > max plausible
    # Only apply check where time_diff > 0 and speed is not NaN
    inconsistent_mask = (
        (df['time_diff_sec_consist'] > 0) & df['mean_speed_mps_consist'].notna() &
        (
            (df['odo_diff_consist'].notna() & (df['odo_diff_consist'] > df['max_plausible_dist_m'])) |
            (df['gps_dist_consist'].notna() & (df['gps_dist_consist'] > df['max_plausible_dist_m']))
        )
    )

    rows_dropped = inconsistent_mask.sum()
    if rows_dropped > 0:
        print(f" - Dropped {rows_dropped} rows due to speed/distance inconsistency.")
        # print("Examples of inconsistent rows:") # Optional debugging
        # print(df[inconsistent_mask][['time_diff_sec_consist', 'odo_diff_consist', 'gps_dist_consist', 'mean_speed_mps_consist', 'max_plausible_dist_m']].head())

    df_filtered = df[~inconsistent_mask].drop(columns=[
        'time_diff_sec_consist', 'odo_diff_consist', 'prev_lat_consist',
        'prev_lon_consist', 'gps_dist_consist', 'mean_speed_mps_consist',
        'max_plausible_dist_m'
    ])
    print(f"Result shape: {df_filtered.shape}")
    return df_filtered

# --- NEW: Trip Start/End Consistency Check ---
def filter_trip_start_end_inconsistency(df, trip_id_col, check_pairs):
    """Filters entire trips where start/end values violate expected logic."""
    print("\n--- Checking Trip Start/End Consistency ---")
    if trip_id_col not in df.columns:
        print(f"Warning: Trip ID column '{trip_id_col}' not found. Skipping check.")
        return df

    trips_to_drop = set()
    required_cols = set([trip_id_col])
    for start_col, end_col, _ in check_pairs:
        required_cols.add(start_col)
        required_cols.add(end_col)

    if not required_cols.issubset(df.columns):
         missing = required_cols - set(df.columns)
         print(f"Warning: Missing columns required for start/end check: {missing}. Skipping.")
         return df

    # Get first/last non-NaN values per trip for the check columns
    try:
        grouped = df.groupby(trip_id_col)
        first_vals = grouped.first() # Gets first non-NaN for all columns
        last_vals = grouped.last()  # Gets last non-NaN for all columns
    except Exception as e:
        print(f"Warning: Error during groupby first/last aggregation. Skipping check. Error: {e}")
        return df


    for start_col, end_col, relationship in check_pairs:
        print(f" - Checking: {end_col} {relationship.replace('_',' ')} {start_col}")
        # Ensure columns are numeric for comparison
        try:
            start_series = pd.to_numeric(first_vals[start_col], errors='coerce')
            end_series = pd.to_numeric(last_vals[end_col], errors='coerce')
        except KeyError:
             print(f"   Warning: Columns '{start_col}' or '{end_col}' missing after group aggregation. Skipping this pair.")
             continue


        if relationship == 'increase':
            # End must be strictly greater than Start
            inconsistent = start_series >= end_series
        elif relationship == 'increase_allow_equal':
            # End must be greater than or equal to Start
            inconsistent = start_series > end_series
        elif relationship == 'decrease':
            # End must be strictly less than Start
            inconsistent = start_series <= end_series
        elif relationship == 'decrease_allow_equal':
            # End must be less than or equal to Start
            inconsistent = start_series < end_series
        else:
            print(f"Warning: Unknown relationship '{relationship}' specified. Skipping this pair.")
            continue

        # Add trip IDs that fail the check (ignoring trips where start or end was NaN)
        failed_trips = inconsistent.index[inconsistent & start_series.notna() & end_series.notna()]
        if not failed_trips.empty:
             print(f"   Found {len(failed_trips)} trips failing {start_col}/{end_col} check.")
             trips_to_drop.update(failed_trips.tolist())

    if trips_to_drop:
        num_to_drop = len(trips_to_drop)
        print(f"Found {num_to_drop} trips failing start/end consistency checks.")
        df_filtered = df[~df[trip_id_col].isin(trips_to_drop)].copy()
        print(f"Result shape after removing inconsistent start/end trips: {df_filtered.shape}")
        return df_filtered
    else:
        print("No trips failed start/end consistency checks.")
        print(f"Result shape: {df.shape}")
        return df

# --- (Keep filter_distance_from_route placeholder) ---
def filter_distance_from_route(df):
    """Filter based on distance from matched route (placeholder)."""
    print("\n--- Filtering Distance From Route (Placeholder) ---")
    print("Warning: filter_distance_from_route is a placeholder and not performing checks.")
    print(f"Result shape: {df.shape}")
    return df

# --- Main Execution Logic ---
def main():
    """Runs the preprocessing pipeline using parameters defined in the configuration block."""
    print(f"--- Starting Preprocessing Pipeline ---")
    print(f"\nInput file: {INPUT_FILE_PATH}")
    print(f"Output file: {OUTPUT_FILE_PATH}")

    # # --- Load Land Geometry (Do this ONCE) ---
    # land_geom = None
    # try:
    #     print(f"\n--- Loading Land Shapefile ---")
    #     print(f"Path: {LAND_SHAPEFILE_PATH}")
    #     # Check if file exists before trying to load
    #     if not os.path.exists(LAND_SHAPEFILE_PATH):
    #          raise FileNotFoundError(f"Shapefile not found at: {LAND_SHAPEFILE_PATH}")
    #     world = gpd.read_file(LAND_SHAPEFILE_PATH)
    #     # Ensure it's in WGS84 (EPSG:4326) to match lat/lon
    #     if world.crs != "EPSG:4326":
    #          print(f"Converting shapefile CRS from {world.crs} to EPSG:4326...")
    #          world = world.to_crs("EPSG:4326")
    #     print("Unifying land geometry...")
    #     land_geom = world.geometry.union_all()
    #     print("Land geometry loaded and unified.")
    # except ImportError:
    #     print("Warning: GeoPandas not found. Cannot perform GPS land check.")
    # except FileNotFoundError as e:
    #      print(f"Warning: {e}. Cannot perform GPS land check.")
    # except Exception as e:
    #     print(f"Warning: Error loading or processing land shapefile: {e}. Cannot perform GPS land check.")
    # # ------------------------------------------

    try:
        # 1. Load Data
        print("\n--- Loading Data ---")
        df = load_file(INPUT_FILE_PATH)
        if not isinstance(df, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Initial shape: {df.shape}")

        # --- Subset for Testing (Optional) ---
        if PROCESS_SUBSET is not None and PROCESS_SUBSET < len(df):
             print(f"\n--- Processing Subset: First {PROCESS_SUBSET} rows ---")
             df = df.head(PROCESS_SUBSET).copy()
        # ------------------------------------

        # 2. Rename Columns
        print("\n--- Renaming Columns ---")
        rename_map_existing = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map_existing)
        cols_to_keep = list(rename_map_existing.values())
        if LAT_COL not in cols_to_keep: cols_to_keep.append(LAT_COL)
        if LON_COL not in cols_to_keep: cols_to_keep.append(LON_COL)
        #df = df[cols_to_keep] # Keep only mapped columns
        df = df[[col for col in cols_to_keep if col in df.columns]]
        print(f"Columns after renaming: {df.columns.tolist()}")
        # --- Apply Preprocessing Steps Sequentially ---

        # 3. Convert H3 Index to Lat/Lon (Add this step early)
        print(f"\n--- Converting H3 Index ('{H3_COL}') to Lat/Lon ---")
        if H3_COL in df.columns:
            # Apply the conversion function; it returns tuples (lat, lon)
            lat_lon_tuples = df[H3_COL].apply(h3_to_latlon)

            # Create the new columns from the tuples
            df[LAT_COL] = lat_lon_tuples.apply(lambda x: x[0])
            df[LON_COL] = lat_lon_tuples.apply(lambda x: x[1])

            # Report potential failures
            failed_conversions = df[LAT_COL].isna().sum()
            if failed_conversions > 0:
                print(f"Warning: {failed_conversions} rows failed H3 to Lat/Lon conversion (resulted in NaN).")
            print(f"Added '{LAT_COL}' and '{LON_COL}' columns.")
        else:
            print(f"Warning: H3 index column '{H3_COL}' not found. Skipping conversion.")
        print(f"Shape after H3 conversion: {df.shape}")
        # ----------------------------------------------------

        # *** 8. Validate and Filter Lat/Lon (includes trip filter) *** 
        df = validate_and_filter_latlon(
            df,
            lat_col=LAT_COL,
            lon_col=LON_COL,
            trip_id_col=TRIP_ID_COL,
            filter_zeros=FILTER_ZERO_COORDINATES,
            zero_tolerance=1e-3, # Use appropriate tolerance
            filter_mostly_zero_trips=FILTER_MOSTLY_ZERO_TRIPS,
            zero_trip_threshold=ZERO_TRIP_THRESHOLD
        )
        # **********************************************************

        # 3. Check Trip Static Data Consistency (Trip Level)
        df = filter_inconsistent_trip_data(df, trip_id_col=TRIP_ID_COL, cols_to_check=STATIC_TRIP_COLS)

        # 4. Check/Drop Redundant ID (Trip Level)
        df = check_and_drop_correlated_id(df, primary_id_col=TRIP_ID_COL, secondary_id_col=MSG_SESSION_ID_COL)

        # 5. Basic Duplicates (Row Level)
        df = remove_duplicates(df, subset_cols=DUPLICATE_CHECK_COLS_STD)

        # 6. Remove Constant/All-NaN Columns
        df = remove_constant_cols(df)

        # *** Apply GPS Land Filter ***
        # Pass the standard LAT_COL and LON_COL names
        # df = filter_gps_at_sea(df, land_geom=land_geom, lat_col=LAT_COL, lon_col=LON_COL)

        # --- Fetch Weather Data ---
        if FETCH_WEATHER_DATA:
            print("\n--- Fetching External Weather Data (Row-by-Row) ---")
            print("Applying weather function... This may take a significant amount of time.")
            # Apply the function - result is a DataFrame with new weather columns
            # Make sure standard column names are passed if needed by the function
            weather_cols_df = df.apply(
                get_weather_for_timestamp, # Use the function defined above
                axis=1,
                result_type='expand', # Creates new columns directly
                lat_col=LAT_COL,
                lon_col=LON_COL,
                time_col=TIME_COL,
                h3_col=H3_COL
            )
            # Concatenate the new weather columns to the original DataFrame
            df = pd.concat([df, weather_cols_df], axis=1)
            print(f"Shape after adding weather columns: {df.shape}")
            print(f"Weather columns added: {weather_cols_df.columns.tolist()}")
        else:
            print("\n--- Skipping External Weather Data Fetching ---")

        # --- Flag Suspicious Temperatures ---
        if FETCH_WEATHER_DATA and 'weather_temp' in df.columns:
             df = flag_suspicious_temperatures(
                 df,
                 vehicle_temp_col=VEHICLE_TEMP_COL, # Standard name for vehicle temp
                 weather_temp_col='weather_temp', # Name created by get_weather...
                 station_dist_col='station_dist_km', # Explicitly pass or rely on default
                 max_difference_c=MAX_TEMP_DIFFERENCE_C,
                 flag_col_name='temp_suspicious_flag' # Optional: change flag name if desired
             )
        else:
             print("\n--- Skipping Temperature Comparison (No weather data fetched or temp column missing) ---")

        # 7. Filter Unrealistic Ranges (Row Level)
        range_checks = {
            SOC_COL: (MIN_VALID_SOC, MAX_VALID_SOC),
            VEHICLE_TEMP_COL: (MIN_VALID_TEMP, MAX_VALID_TEMP),
            ALT_ARRAY_COL: (MIN_VALID_ALTITUDE, MAX_VALID_ALTITUDE), # Checks mean
            BATT_HEALTH_COL: (MIN_VALID_BATT_HEALTH, MAX_VALID_BATT_HEALTH),
            ODO_COL: (0, MAX_VALID_ODO), # Odo should start >= 0
            WEIGHT_COL: (MIN_VALID_WEIGHT, MAX_VALID_WEIGHT),
            LAT_COL: (-90, 90), # Add Lat check
            LON_COL: (-180, 180), # Add Lon check
            # Add speed range check if desired (checking mean of speed_array)
            SPEED_ARRAY_COL: (0, 250) # Example: 0 to 250 kph for mean speed
        }
        df = filter_unrealistic_ranges(df, range_checks)

        # 8. Filter Negative Values (Row Level) - Extended
        cols_must_be_non_negative = [
            ODO_COL, BATT_HEALTH_COL, WEIGHT_COL,
            SPEED_ARRAY_COL # Checks mean
        ]
        df = filter_negative_values_extended(df, cols_to_check=cols_must_be_non_negative)

        # 9. Timestamp Checks (Row Level) - Includes Max Gap
        #df = filter_non_increasing_timestamps(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL, max_gap_seconds=MAX_CONSECUTIVE_TIME_GAP_SECONDS)

        # 10. Odometer Direction Check (Row Level)
        #df = filter_decreasing_odometer(df, odo_col=ODO_COL, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)

        # 11. SoC Direction Check (Row Level)
        #df = filter_increasing_soc(df, soc_col=SOC_COL, trip_id_col=TRIP_ID_COL, time_col=TIME_COL, threshold=SOC_INCREASE_THRESHOLD)

        # 12. GPS Checks (Now uses the created LAT_COL, LON_COL)
        #df = filter_gps_over_sea(df, lat_col=LAT_COL, lon_col=LON_COL) 
        #df = filter_big_jumps(df, max_distance_km=MAX_GPS_JUMP_KM, lat_col=LAT_COL, lon_col=LON_COL, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)
        
        # 13. Big Difference Checks (Row Level) - Enhanced to use means for arrays
        #df = filter_big_differences(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL,
                                    # speed_col=SPEED_ARRAY_COL, alt_col=ALT_ARRAY_COL, soc_col=SOC_COL, temp_col=TEMP_COL,
                                    # max_speed_diff=MAX_CONSECUTIVE_SPEED_DIFF, max_alt_diff=MAX_CONSECUTIVE_ALT_DIFF,
                                    # max_soc_diff=MAX_CONSECUTIVE_SOC_DIFF, max_temp_diff=MAX_CONSECUTIVE_TEMP_DIFF)

        # 14. Speed/Distance Consistency Check (Row Level)
        #df = filter_speed_distance_inconsistency(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL,
                                                # odo_col=ODO_COL, lat_col=LAT_COL, lon_col=LON_COL,
                                                # speed_col=SPEED_ARRAY_COL, tolerance_factor=SPEED_DIST_TOLERANCE_FACTOR)

        # 15. Trip Length Checks (Trip Level)
        #df = filter_short_trips(df, min_length_km=MIN_TRIP_KM, odo_col=ODO_COL, trip_id_col=TRIP_ID_COL)
        #df = filter_long_trips(df, max_length_km=MAX_TRIP_KM, odo_col=ODO_COL, trip_id_col=TRIP_ID_COL)

        # 16. Trip Start/End Consistency Check (Trip Level)
        # Use RENAMED cycle start/end columns
        start_end_pairs = [
            ('cycle_datetime_start', 'cycle_datetime_end', 'increase'), # Time must increase
            ('cycle_odo_start', 'cycle_odo_end', 'increase_allow_equal'), # Odo must increase or stay same
            ('cycle_soc_start', 'cycle_soc_end', 'decrease_allow_equal') # SoC should decrease or stay same
        ]
        df = filter_trip_start_end_inconsistency(df, trip_id_col=TRIP_ID_COL, check_pairs=start_end_pairs)

        # 17. Placeholder Filter
        df = filter_distance_from_route(df)

        # --- Final Check & Save ---
        print("\n--- Final Data Check ---")
        if df.empty: print("Warning: DataFrame is empty after preprocessing.")
        else: 
            print(f"Final shape: {df.shape}")
            print(f"Final columns: {df.columns.tolist()}")

        # --- Save the cleaned data using save_file helper ---
        print(f"\n--- Saving Cleaned Data ---")
        try:
            # Extract directory, base filename, and format from the full path
            output_dir = os.path.dirname(OUTPUT_FILE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FILE_PATH))[0]
            output_extension = os.path.splitext(OUTPUT_FILE_PATH)[1].lower().lstrip('.')

            # Determine format for the helper function
            if output_extension in ['parquet', 'pickle']:
                save_format = output_extension
            else:
                # Default or fallback if extension is different (e.g., .csv)
                # save_file currently only supports parquet/pickle, so this might error
                # or you might adjust save_file to handle csv too.
                # For now, let's try passing the extension, save_file will validate.
                warnings.warn(f"Output file extension '.{output_extension}' may not be directly supported by save_file helper (expects 'parquet' or 'pickle'). Attempting anyway.")
                save_format = output_extension # Let save_file handle validation

            # Call the helper function
            save_file(
                data=df,
                path=output_dir,
                file_name=base_file_name,
                format=save_format, # Pass the determined format
                index=False         # Don't save index for parquet
            )
            # Success message is printed inside save_file
            print("Preprocessing complete.")

        except (ValueError, TypeError, ImportError, Exception) as e:
            # Catch errors specifically from save_file or path extraction
            print(f"Failed to save file using helper function: {e}")

    except (FileNotFoundError, TypeError, ValueError, IsADirectoryError, ImportError, KeyError, Exception) as e:
        print(f"\n--- Preprocessing Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()

# make sure that all GPS is in land (i.e., not in the sea)
# make sure that the following cols do not have higher than logical diff between consecutive (within cycle) timestamps: distance (GPS and odometer), duration, temp, alltitude, speed, soc, battery health).
# make sure that the diff is in logical direction fof the relevant cols (e.g. odometer increasing, timestamp increasing, soc decreasing etc.)
# make sure that there are no negative values where there shouldn't be (e.g. battery health, odometer etc.)
# make sure that the odometer change / gps change is reasonble to the speed.
# make sure that the feature are in logical ranges (e.g. for weight, temp, soc, battery health,  alltitude, speed, odometer etc.)
# make sure that all the differences between cycles start to cycle ends are logical (e.g. decrease in soc, increase in time, increase in odometer etc.)

