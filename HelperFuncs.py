import math
import requests
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
import os
import pandas as pd
import numpy as np
import random
import pickle
import warnings
import h3
import time
from polyline import decode
import traceback # For detailed error logging
import matplotlib.pyplot as plt
import seaborn as sns
import fiona
from itertools import chain
from dataclasses import dataclass
from typing import List, Any, Callable, Tuple

def load_file(filepath, **kwargs):
    """
    Loads data from various file types into memory.

    Supports CSV, Parquet, Excel (xls, xlsx), JSON, and Pickle (pkl, pickle) files.

    Args:
        filepath (str): The path to the file to load.
        **kwargs: Additional keyword arguments to pass to the underlying
                  pandas read function (read_csv, read_parquet, read_excel, read_json).
                  These arguments are IGNORED for pickle files.

    Returns:
        pandas.DataFrame: For CSV, Parquet, Excel, JSON files.
        object: The deserialized Python object for Pickle files (could be
                dict, list, DataFrame, etc.).

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the filepath does not exist.
        Exception: Other exceptions specific to the file type or reading process
                   (e.g., pd.errors.ParserError, pickle.UnpicklingError).
    """
    # Check if file exists first for a clearer error
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: '{filepath}'")
    if not os.path.isfile(filepath):
         raise IsADirectoryError(f"Path is not a file: '{filepath}'") # Or similar check
    
    extension = os.path.splitext(filepath)[1].lower()
    if extension == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif extension == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif extension in ('.xls', '.xlsx'):
        return pd.read_excel(filepath, **kwargs)
    elif extension == '.json':
        return pd.read_json(filepath, **kwargs)
    elif extension in ('.pkl', '.pickle'):
        try:
            with open(filepath, 'rb') as f: # Open in binary read mode ('rb')
                # Load the object using pickle.load
                # Note: We generally don't pass pandas **kwargs to pickle.load
                loaded_object = pickle.load(f)
            return loaded_object
        except pickle.UnpicklingError as e:
            # Raise a specific error if unpickling fails
            raise pickle.UnpicklingError(f"Error unpickling file {filepath}: {e}")
        except Exception as e:
            # Catch other potential errors during file open/read for pickle
            raise Exception(f"An error occurred loading pickle file {filepath}: {e}")
    # --- End Pickle Support ---
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    

def save_file(data, path, file_name, format='pickle', index=False):
    """
    Saves the given data object to a file in the specified path.
    Supports 'pickle' and 'parquet' formats.

    For 'pickle', the '.pickle' extension is automatically appended.
    For 'parquet', the '.parquet' extension is automatically appended.
    The directory specified by 'path' will be created if it doesn't exist.

    Args:
        data (any): The Python object to save. Must be a pandas DataFrame
                    if format='parquet'.
        path (str): The directory path where the file should be saved.
        file_name (str): The base name for the file (without extension).
        format (str): The format to save in ('pickle' or 'parquet').
                      Defaults to 'pickle'.
        index (bool): For Parquet format only. Whether to write the DataFrame
                      index as a column. Defaults to False.

    Raises:
        ValueError: If an unsupported format is specified.
        TypeError: If trying to save non-DataFrame data as Parquet.
        OSError: If the directory cannot be created.
        pickle.PicklingError: If the object cannot be pickled (format='pickle').
        Exception: For other file I/O or saving errors (e.g., Parquet engine issues).
    """
    # Validate format
    supported_formats = ['pickle', 'parquet']
    if format.lower() not in supported_formats:
        raise ValueError(f"Unsupported format '{format}'. Supported formats are: {supported_formats}")

    format = format.lower() # Ensure lowercase for comparison

    # Ensure the target directory exists
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        raise

    # Determine file extension and construct full path
    extension = '.pickle' if format == 'pickle' else '.parquet'
    full_file_path = os.path.join(path, file_name + extension)

    print(f"Attempting to save data to: {full_file_path} (Format: {format})")

    try:
        if format == 'pickle':
            # Save as Pickle
            with open(full_file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == 'parquet':
            # Save as Parquet
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Data must be a pandas DataFrame to save as Parquet. Got type: {type(data)}")
            # Use pyarrow engine by default, requires 'pyarrow' package
            # Can add 'engine' as another parameter if needed ('fastparquet')
            try:
                data.to_parquet(full_file_path, index=index, engine='pyarrow')
            except ImportError:
                warnings.warn("Attempting Parquet save with 'fastparquet' engine as 'pyarrow' was not found. Install 'pyarrow' for better compatibility.")
                try:
                     data.to_parquet(full_file_path, index=index, engine='fastparquet')
                except ImportError:
                     raise ImportError("Could not save as Parquet. Please install 'pyarrow' or 'fastparquet'. (e.g., pip install pyarrow)")


        print(f"Data successfully saved.")

    except pickle.PicklingError as e:
        print(f"Error pickling data: {e}")
        raise
    except TypeError as e: # Catch the TypeError we raise for non-DataFrames in Parquet
         print(f"Type error saving file: {e}")
         raise
    except ImportError as e: # Catch engine import errors
         print(f"Import error saving file: {e}")
         raise
    except Exception as e:
        print(f"An error occurred while saving file: {e}")
        raise

def sort_df_by_trip_time(df: pd.DataFrame, trip_id_col='trip_id', time_col='timestamp') -> pd.DataFrame:
    """
    Sorts the input DataFrame by trip ID and timestamp columns
    in ascending order and returns the sorted DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to sort.
        trip_id_col (str): Name of the trip identifier column.
        time_col (str): Name of the timestamp column.

    Returns:
        pd.DataFrame: A new DataFrame sorted as specified.

    Raises:
        KeyError: If specified columns are not found in the DataFrame.
    """
    required_columns = [trip_id_col, time_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        # Use f-string for cleaner error message
        raise KeyError(f"DataFrame is missing required sorting columns: {missing_columns}")

    # Use inplace=False (default) and return the result
    # Sort using the provided column names
    sorted_df = df.sort_values(
        by=required_columns,
        ascending=[True, True],
        inplace=False
    )
    return sorted_df

def h3_to_latlon(h3_index):
    """
    Converts a single H3 index (numeric type or string representation of a
    decimal integer) to a (latitude, longitude) tuple. Aligns with the
    logic: loc = int(val) if is_string else val; hex_val = format(loc, 'x').

    Args:
        h3_index: The H3 index (numeric or string decimal integer).

    Returns:
        tuple: A tuple containing (latitude, longitude) in degrees,
               or (np.nan, np.nan) if the input is invalid or conversion fails.
    """
    if h3_index is None or pd.isna(h3_index):
        return (np.nan, np.nan)

    original_input = h3_index
    h3_hex_str = None

    try:
        # --- Conversion Logic (Mirroring user snippet) ---
        # 1. Ensure we have an integer type. This handles:
        #    - Actual integers/numpy integers
        #    - Floats that represent integers (e.g., 123.0)
        #    - String representations of DECIMAL integers (e.g., "123")
        try:
            loc_int = int(h3_index)
        except (ValueError, TypeError) as e:
            # This happens if h3_index is a non-integer string (like hex, or text)
            # or another non-convertible type.
            warnings.warn(f"Could not convert H3 input '{original_input}' to integer: {e}. Assuming invalid.")
            return (np.nan, np.nan)

        # 2. Convert the integer to its hex string representation
        h3_hex_str = format(loc_int, 'x')
        # -------------------------------------------------

        if not h3_hex_str: # Should not happen if int conversion succeeded
             warnings.warn(f"Failed to create hex string from integer {loc_int} (original: '{original_input}').")
             return (np.nan, np.nan)

        # 3. Use cell_to_latlng with the generated hex string
        lat, lon = h3.cell_to_latlng(h3_hex_str)
        return (lat, lon)

    except h3.H3ValueError as e: # Catch specific H3 validation errors
        # This means the hex string (derived from the int) wasn't valid H3
        # warnings.warn(f"Invalid H3 index '{h3_hex_str}' (from original '{original_input}'): {e}")
        return (np.nan, np.nan)
    except Exception as e:
        # Catch any other unexpected errors (less likely now)
        warnings.warn(f"Unexpected error converting H3 index '{h3_hex_str}' (from original '{original_input}'): {e}")
        return (np.nan, np.nan)

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers. For miles, use 3956.
    R = 6371.0  
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    
    a = math.sin(d_lat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # distance in kilometers
    return distance

world = gpd.read_file(r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\110m_cultural\ne_110m_admin_0_countries.shp")
# Instead of using 'unary_union', use 'union_all()' on the GeoSeries of geometries:
land = world.geometry.union_all()

def is_on_land(lat, lon):
    # Create a Point (note: order is longitude, latitude)
    point = ShapelyPoint(lon, lat)
    return land.contains(point)

# --- Main Function to get the geometry of the target countries ---
# --- NEW GADM-FOCUSED Function to get the geometry ---
def get_gadm_countries_geometry(gadm_directory, country_iso_codes, buffer_meters=1000):
    """
    Loads high-resolution GADM data for specified countries, automatically finds
    the correct layer name, combines them, and applies a buffer.
    """
    print(f"Loading GADM data from: {gadm_directory}")
    country_geometries = []

    for code in country_iso_codes:
        file_path = os.path.join(gadm_directory, f"gadm41_{code}.gpkg")
        if not os.path.exists(file_path):
            print(f"Warning: GADM file not found for {code} at {file_path}. Skipping.")
            continue
        
        try:
            # --- Automatic Layer Detection ---
            # 1. List all layers in the GeoPackage
            available_layers = fiona.listlayers(file_path)
            
            # 2. Find the layer that ends with '_0' (this is the country boundary)
            target_layer = next((layer for layer in available_layers if layer.endswith('_0')), None)
            
            if not target_layer:
                print(f"Error: Could not find a Level 0 boundary layer in {file_path}. Available: {available_layers}")
                continue
            # --- End of Detection ---

            print(f"Found target layer '{target_layer}' for {code}.")
            
            # 3. Read the data using the discovered layer name
            country_gdf = gpd.read_file(file_path, layer=target_layer)
            country_geometries.append(country_gdf)
            print(f"Successfully loaded GADM data for {code}.")

        except Exception as e:
            print(f"An error occurred while processing the file for {code}: {e}")
            continue

    if not country_geometries:
        print("Error: No GADM data could be loaded. Please check paths and codes.")
        return None

    # Combine all loaded countries into a single GeoDataFrame
    all_countries_gdf = pd.concat(country_geometries, ignore_index=True)
    all_countries_gdf = all_countries_gdf.to_crs("EPSG:4326")

    print(f"Applying a {buffer_meters}-meter buffer to the high-resolution geometry...")
    metric_gdf = all_countries_gdf.to_crs("EPSG:3035")
    metric_gdf['geometry'] = metric_gdf.geometry.buffer(buffer_meters)
    buffered_gdf = metric_gdf.to_crs("EPSG:4326")
    
    return buffered_gdf.geometry.unary_union

# --- MODIFIED function to find and return points outside the area ---
def find_points_outside_target_area(df, target_geometry, trip_id_col='trip_id', lat_col='latitude', lon_col='longitude'):
    """
    Identifies points in a DataFrame that are outside a given target geometry
    and returns their trip ID and coordinates.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        target_geometry (shapely.geometry): The unified geometry of the target area.
        trip_id_col (str): The name of the trip ID column.
        lat_col (str): The name of the latitude column.
        lon_col (str): The name of the longitude column.
        h3_col (str, optional): The name of the H3 index column to use for coordinates.

    Returns:
        pd.DataFrame: A DataFrame with trip_id, latitude, and longitude of points
                      that are OUTSIDE the target area.
    """
    if target_geometry is None:
        print("Error: Target geometry is invalid. Cannot perform check.")
        return pd.DataFrame(columns=[trip_id_col, lat_col, lon_col])

    df_copy = df.copy()

    # Ensure required columns exist before proceeding
    required_cols = [trip_id_col, lat_col, lon_col]
    if not all(col in df_copy.columns for col in required_cols):
        print(f"Error: Missing one of the required columns: {required_cols}")
        return pd.DataFrame(columns=required_cols)

    # Create a GeoDataFrame from the points
    points_gdf = gpd.GeoDataFrame(
        df_copy,
        geometry=gpd.points_from_xy(df_copy[lon_col], df_copy[lat_col]),
        crs="EPSG:4326"
    )

    # Perform the spatial check to find points inside the target area
    is_inside = points_gdf.within(target_geometry)

    # Filter to get only the points that are OUTSIDE
    outside_points_gdf = points_gdf[~is_inside]

    # Return only the requested columns for the outside points
    return outside_points_gdf[[trip_id_col, lat_col, lon_col]].reset_index(drop=True)

# --- Placeholder for DEM-based Elevation Change along Route ---
# This would require querying DEM multiple times along the route geometry from OSRM/GraphHopper
# Or using GraphHopper/Valhalla if they provide elevation profile directly.
def get_route_elevation_change(start_lat, start_lon, end_lat, end_lon):
     """Placeholder: Calculates elevation change along a route."""
     warnings.warn("get_route_elevation_change is a placeholder and returns 0.")
     return {'predicted_elevation_gain': 0.0, 'predicted_elevation_loss': 0.0}

def get_trip_level_df(df):
    """
    Aggregate dataframe to trip level by 'trip_id'.

    Parameters:
    - df (pd.DataFrame): Input dataframe with potentially multiple rows per trip.

    Returns:
    - trip_level_df (pd.DataFrame): Dataframe with one row per 'trip_id'.
    """
    df_copy = df.copy()
    # Define trip-level columns
    trip_level_cols = [
        'FAMILY_LABEL', 'COMMERCIAL_BRAND_COMPONENT_TYP', 'TYPE_OF_TRACTION_CHAIN',
        'EMPTY_WEIGHT_KG', 'DESTINATION_COUNTRY_CODE', 'BATTERY_TYPE',
        'MESS_ID_START', 'DATETIME_START', 'DATETIME_END', 'SOC_START', 'SOC_END',
        'ODO_START', 'ODO_END', 'dt', 'geoindex_10_start', 'geoindex_10_end'
    ]

    # Check which columns are actually present in the dataframe
    available_cols = [col for col in trip_level_cols if col in df_copy.columns]
    agg_dict = {col: 'first' for col in available_cols}
    trip_level_df = df_copy.groupby('trip_id').agg(agg_dict).reset_index()

    return trip_level_df

def random_partition(total, parts, random_state=42):
    """Randomly partition an integer 'total' into 'parts' nonnegative integers that sum to 'total'."""
    # If parts is less than or equal to 1, no partitioning is needed.
    if parts <= 1:
        return [total]
    
    # Initialize a local RNG for reproducibility
    if isinstance(random_state, random.Random):
        rng = random_state
    else:
        rng = random.Random(random_state)

    # Pick 'parts-1' divider positions uniformly from 1 to total+parts-1 (inclusive).
    dividers = sorted(rng.sample(range(1, total + parts), parts - 1))
    
    # The first partition: stars before the first divider.
    partition = [dividers[0] - 1]
    # Middle partitions: gaps between dividers (subtracting 1 for each divider).
    partition += [dividers[i] - dividers[i - 1] - 1 for i in range(1, len(dividers))]
    # Last partition: stars after the last divider.
    partition.append(total + parts - 1 - dividers[-1])
    
    return partition
   
# --- Safely calculate mean of potential array/scalar ---
def safe_mean(val):
    """
    Safely calculates the mean of a value which might be a list, array, or scalar.
    Handles non-numeric types and NaNs within arrays.

    Args:
        val: The value or iterable to calculate the mean from.

    Returns:
        float: The mean of the numeric values, or np.nan if calculation fails
               or no numeric values are found.
    """
    try:
        if isinstance(val, (list, np.ndarray)):
            # Filter only numeric types and exclude NaNs before calculating mean
            numeric_vals = [v for v in val if isinstance(v, (int, float, np.number)) and not np.isnan(v)]
            return np.mean(numeric_vals) if numeric_vals else np.nan
        elif isinstance(val, (int, float, np.number)) and not np.isnan(val):
            # Handle scalar numeric types
            return float(val)
        else:
            # Handle None, strings, other non-numeric types
            return np.nan
    except Exception:
        # Catch any unexpected errors during processing
        return np.nan
    
def drop_short_long_trips(df, min_dur=0, min_dis=0, max_dur=1_000, max_dis=1_000):

    df_copy = df.copy()

    df_filtered = df_copy[(df_copy['trip_duration'] > min_dur) & (df_copy['trip_duration'] < max_dur)] 

    df_filtered = df_filtered[(df_filtered['trip_distance'] > min_dis) & (df_filtered['trip_distance'] < max_dis)] 

    return df_filtered

# --- NEW Kinematics Calculation Helper ---
def calculate_kinematics(timestamps, speed_kph_arrays):
    """
    Calculates time difference, speed (m/s), acceleration (m/s^2), and jerk (m/s^3)
    from timestamp and speed array data within a segment slice.

    Args:
        timestamps (pd.Series): Series of timestamps for the segment slice.
        speed_kph_arrays (pd.Series): Series of speed arrays (kph) for the segment slice.

    Returns:
        pd.DataFrame: DataFrame with calculated kinematics (dt_s, speed_mps, accel_mps2, jerk_mps3),
                      indexed like the input Series. Returns empty DataFrame on error.
    """
    if len(timestamps) != len(speed_kph_arrays) or len(timestamps) < 2:
        return pd.DataFrame(columns=['dt_s', 'speed_mps', 'accel_mps2', 'jerk_mps3'])

    try:
        # Ensure timestamps are datetime objects
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        if timestamps.isnull().any():
            warnings.warn("NaNs found in timestamps during kinematics calculation.")
            # Optionally handle or return empty
            # return pd.DataFrame(columns=['dt_s', 'speed_mps', 'accel_mps2', 'jerk_mps3'])


        # Calculate time differences in seconds for each row relative to the previous
        # Use total_seconds() for accurate float representation
        dt_s = timestamps.diff().dt.total_seconds()
        # dt_s.iloc[0] will be NaN, handle later or fill with a sensible value (e.g., median diff)
        median_dt = dt_s.median()
        if pd.isna(median_dt) or median_dt <= 0: median_dt = 1.0 # Default to 1s if median is invalid
        dt_s = dt_s.fillna(median_dt) # Fill first NaN

        # Calculate mean speed in m/s for each row
        speed_mps = speed_kph_arrays.apply(safe_mean) * (1000 / 3600) # KPH to m/s

        # Calculate acceleration (change in speed / change in time)
        # Shift speed to get previous speed, then calculate diff
        speed_diff_mps = speed_mps.diff()
        # Avoid division by zero or very small dt; use forward fill for accel where dt is bad
        accel_mps2 = (speed_diff_mps / dt_s).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # accel_mps2.iloc[0] will be NaN/0 based on fillna

        # Calculate jerk (change in acceleration / change in time)
        accel_diff_mps2 = accel_mps2.diff()
        jerk_mps3 = (accel_diff_mps2 / dt_s).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # jerk_mps3.iloc[0] will be NaN/0

        kinematics_df = pd.DataFrame({
            'dt_s': dt_s,
            'speed_mps': speed_mps,
            'accel_mps2': accel_mps2,
            'jerk_mps3': jerk_mps3
        }, index=timestamps.index) # Ensure index alignment

        return kinematics_df

    except Exception as e:
        warnings.warn(f"Error during kinematics calculation: {e}")
        return pd.DataFrame(columns=['dt_s', 'speed_mps', 'accel_mps2', 'jerk_mps3'])

# --- NEW Kinematics Aggregation Helper ---
def aggregate_kinematics(kinematics_df, high_accel_thr=2.0, high_decel_thr=-2.0):
    """
    Calculates aggregate statistics from the kinematics DataFrame.

    Args:
        kinematics_df (pd.DataFrame): Output from calculate_kinematics.
        high_accel_thr (float): Threshold for high acceleration events (m/s^2).
        high_decel_thr (float): Threshold for high deceleration events (m/s^2).

    Returns:
        dict: Dictionary of aggregated kinematic features.
    """
    aggs = {}
    if kinematics_df.empty:
        return aggs # Return empty dict if no data

    # Basic speed stats (from m/s)
    aggs['speed_seg_agg_mean_mps'] = kinematics_df['speed_mps'].mean()
    aggs['speed_seg_agg_std_mps'] = kinematics_df['speed_mps'].std()
    aggs['speed_seg_agg_max_mps'] = kinematics_df['speed_mps'].max()
    aggs['speed_seg_agg_min_mps'] = kinematics_df['speed_mps'].min()

    # Acceleration stats (positive values)
    accel_positive = kinematics_df['accel_mps2'][kinematics_df['accel_mps2'] > 1e-3] # Exclude near-zero
    aggs['accel_seg_agg_mean'] = accel_positive.mean()
    aggs['accel_seg_agg_std'] = accel_positive.std()
    aggs['accel_seg_agg_max'] = accel_positive.max()
    aggs['accel_high_event_count'] = (accel_positive >= high_accel_thr).sum()

    # Deceleration stats (negative values)
    decel = kinematics_df['accel_mps2'][kinematics_df['accel_mps2'] < -1e-3] # Exclude near-zero
    aggs['decel_seg_agg_mean'] = decel.mean() # Will be negative
    aggs['decel_seg_agg_std'] = decel.std()
    aggs['decel_seg_agg_max'] = decel.min() # Max deceleration is the most negative value
    aggs['decel_high_event_count'] = (decel <= high_decel_thr).sum()

    # Jerk stats (absolute value often more informative)
    aggs['jerk_abs_seg_agg_mean'] = kinematics_df['jerk_mps3'].abs().mean()
    aggs['jerk_abs_seg_agg_std'] = kinematics_df['jerk_mps3'].abs().std()
    aggs['jerk_abs_seg_agg_max'] = kinematics_df['jerk_mps3'].abs().max()

    # Fill NaNs that might result from empty series (e.g., no positive accel)
    for key, value in aggs.items():
        if pd.isna(value):
            aggs[key] = 0.0 # Replace NaN aggregates with 0

    return aggs

# --- NEW Stop Feature Calculation Helper ---
def calculate_stop_features(segment_flags, timestamps):
    """
    Calculates number and total duration of stops within a segment span.

    Args:
        segment_flags (np.ndarray): Array of segmentation flags (-1 for stop) for the segment span.
        timestamps (pd.Series): Corresponding timestamps for the segment span.

    Returns:
        dict: Dictionary containing 'stops_seg_count' and 'stop_duration_seg_agg_s'.
    """
    stop_features = {'stops_seg_count': 0, 'stop_duration_seg_agg_s': 0.0}
    if len(segment_flags) != len(timestamps) or len(segment_flags) < 1:
        return stop_features

    try:
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        stop_indices = np.where(segment_flags == -1)[0]

        if len(stop_indices) == 0:
            return stop_features # No stops

        # Calculate duration for each stop point (time since previous point)
        time_diffs = timestamps.diff().dt.total_seconds()
        # Use median diff for the very first point if it's a stop
        median_dt = time_diffs.median()
        if pd.isna(median_dt) or median_dt <= 0: median_dt = 1.0
        time_diffs = time_diffs.fillna(median_dt)

        # Sum durations where flag is -1
        stop_features['stop_duration_seg_agg_s'] = time_diffs.iloc[stop_indices].sum()

        # Count contiguous blocks of stops as one stop 'event'
        if len(stop_indices) > 0:
            # Find where the difference between consecutive stop indices is > 1
            stop_blocks = np.split(stop_indices, np.where(np.diff(stop_indices) > 1)[0] + 1)
            stop_features['stops_seg_count'] = len(stop_blocks)

        return stop_features

    except Exception as e:
        warnings.warn(f"Error calculating stop features: {e}")
        return {'stops_seg_count': 0, 'stop_duration_seg_agg_s': 0.0}
    
# --- OSRM Interaction Helpers ---
def get_osrm_route(coordinates, profile='driving', base_url="http://router.project-osrm.org"):
    """Gets the direct route between the first and last coordinate using OSRM."""
    # (Keep implementation from previous step - gets direct route distance)
    if not coordinates or len(coordinates) < 2: return None
    start_lat, start_lon = coordinates[0]; end_lat, end_lon = coordinates[-1]
    if not all(isinstance(c, (int, float)) and not np.isnan(c) for c in [start_lat, start_lon, end_lat, end_lon]):
        warnings.warn("Invalid coordinate types/NaNs for direct route query.")
        return None
    coord_str = f"{start_lon:.6f},{start_lat:.6f};{end_lon:.6f},{end_lat:.6f}"
    url = f"{base_url}/route/v1/{profile}/{coord_str}?overview=full&geometries=geojson&annotations=false" # No annotations needed
    max_retries=2; retry_delay=0.5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get('code') == 'Ok' and data.get('routes'): return data['routes'][0]
            else: return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1: time.sleep(retry_delay * (attempt + 1))
            else: warnings.warn(f"OSRM route failed after retries: {e}"); return None
        except Exception as e: warnings.warn(f"Unexpected error OSRM route: {e}"); return None
    return None

# --- UPDATED Robust OSRM Match Function ---
def get_osrm_match_robust(
    coordinates,
    timestamps,
    odo_distance_km, # Actual odometer distance for this trace
    trip_id_debug="N/A", # Add IDs for better logging
    segment_id_debug="N/A", # Add IDs for better logging
    osrm_base_url="http://router.project-osrm.org",
    service="/match/v1/driving/",
    initial_radius_m=30.0,
    max_radius_m=150.0,
    radius_multiplier=1.5,
    max_attempts=4, # Initial attempt + 3 retries
    tolerance_pct=10.0,
    gaps_threshold_s=300,
    max_retries_api=2,
    retry_delay_api=0.5,
    tidy=True,
    request_timeout=45 # Increased default timeout
):
    """
    Gets map-matched route, iteratively increasing search radius if the matched
    distance significantly differs from the odometer distance. Fetches annotations.
    Includes enhanced validation and error handling.

    Args:
        # ... (previous args) ...
        trip_id_debug (any): Identifier for logging.
        segment_id_debug (any): Identifier for logging.
        request_timeout (int): Timeout for API requests in seconds.

    Returns:
        dict: {
            'matched_distance_m': float, 'matched_duration_s': float,
            'match_confidence': float, 'match_status': str,
            'match_attempts': int, 'final_radius_m': float,
            'osrm_match_result': dict or None
        }
    """
    result = {
        'matched_distance_m': np.nan, 'matched_duration_s': np.nan,
        'match_confidence': np.nan, 'match_status': 'InputError',
        'match_attempts': 0, 'final_radius_m': initial_radius_m,
        'osrm_match_result': None
    }
    n_points = len(coordinates)

    # --- Input Validation ---
    if not coordinates or not timestamps or len(coordinates) != len(timestamps) or n_points < 2:
        warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] InputError: Insufficient or mismatched coordinates/timestamps.")
        return result
    # Check for NaN/inf in coordinates
    if any(np.isnan(c) or not np.isfinite(c) for lat, lon in coordinates for c in [lat, lon]):
         warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] InputError: NaN or infinite values found in coordinates.")
         return result
    # Check timestamps are valid numbers
    if any(np.isnan(ts) or not np.isfinite(ts) for ts in timestamps):
         warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] InputError: NaN or infinite values found in timestamps.")
         return result
    # Check timestamp monotonicity
    time_diffs = np.diff(timestamps)
    if not np.all(time_diffs > 0):
        warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] InputError: Timestamps are not strictly monotonically increasing.")
        # Find first non-positive diff index for debugging
        non_mono_idx = np.where(time_diffs <= 0)[0]
        if len(non_mono_idx) > 0:
             idx = non_mono_idx[0]
             # print(f"  DEBUG: Non-monotonic step at index {idx}: ts[{idx}]={timestamps[idx]}, ts[{idx+1}]={timestamps[idx+1]}")
        return result # OSRM will likely reject this

    # Handle odo_distance_km
    check_tolerance = True
    if np.isnan(odo_distance_km) or not np.isfinite(odo_distance_km) or odo_distance_km < 0:
         check_tolerance = False; odo_distance_km = -1

    # Prepare OSRM Query Strings
    try:
        coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coordinates])
        ts_str = ";".join([str(int(ts)) for ts in timestamps])
    except Exception as e:
        warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] InputError: Error formatting coords/timestamps: {e}")
        return result

    tidy_str = "&tidy=true" if tidy else ""
    use_gaps_split = np.any(time_diffs > gaps_threshold_s)
    gaps_str = "&gaps=split" if use_gaps_split else "&gaps=ignore"
    annotations_str = "&annotations=duration,distance,speed,nodes"
    overview_str = "&overview=full"
    geometry_str = "&geometries=geojson"

    current_radiuses = [initial_radius_m] * n_points
    result['final_radius_m'] = initial_radius_m

    # Iterative Matching Loop
    for attempt in range(max_attempts):
        result['match_attempts'] = attempt + 1
        radiuses_str = "&radiuses=" + ";".join(str(int(r)) for r in current_radiuses)

        url = (
            f"{osrm_base_url.rstrip('/')}{service.rstrip('/')}/{coord_str}"
            f"?timestamps={ts_str}"
            f"{radiuses_str}{gaps_str}{tidy_str}{annotations_str}{overview_str}{geometry_str}"
        )

        # Inner loop for API retries
        response_data = None
        for api_retry in range(max_retries_api):
            try:
                # print(f"    DEBUG: Attempt {attempt+1}, API Retry {api_retry+1}, URL: {url[:150]}...") # Debug URL
                response = requests.get(url, timeout=request_timeout)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response_data = response.json()
                break # Success
            except requests.exceptions.Timeout:
                warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] OSRM Match timed out (API retry {api_retry + 1}/{max_retries_api}). URL: {url[:150]}...")
                if api_retry == max_retries_api - 1: result['match_status'] = 'Error_Timeout'; return result
            except requests.exceptions.HTTPError as http_err:
                 # Log specific HTTP errors (like 400 Bad Request)
                 warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] OSRM Match HTTP Error (API retry {api_retry + 1}/{max_retries_api}): {http_err}. URL: {url[:150]}...")
                 # Log request details for 400 errors
                 if response.status_code == 400:
                      print(f"    DEBUG 400 Input: Coords={len(coordinates)}, Timestamps={len(timestamps)}, Radii={len(current_radiuses)}")
                      # print(f"    DEBUG 400 URL: {url}") # Can be very long
                 if api_retry == max_retries_api - 1: result['match_status'] = f'Error_HTTP_{response.status_code}'; return result
            except requests.exceptions.RequestException as e:
                warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] OSRM Match API request failed (API retry {api_retry + 1}/{max_retries_api}): {e}")
                if api_retry == max_retries_api - 1: result['match_status'] = 'Error_Request'; return result
            except Exception as e:
                warnings.warn(f"[{trip_id_debug}-{segment_id_debug}] Unexpected error during OSRM Match query: {e}")
                traceback.print_exc() # Print full traceback for unexpected errors
                result['match_status'] = 'Error_Unexpected'; return result
            # Wait before API retry
            if api_retry < max_retries_api - 1: time.sleep(retry_delay_api * (api_retry + 1))

        if response_data is None: continue # API failed after retries, proceed to next match attempt if any

        # Process OSRM response
        if response_data.get('code') == 'Ok' and response_data.get('matchings'):
            match_summary = response_data['matchings'][0]
            matched_dist_m = match_summary.get('distance')
            matched_dur_s = match_summary.get('duration')
            confidence = match_summary.get('confidence')

            result['matched_distance_m'] = float(matched_dist_m) if matched_dist_m is not None else np.nan
            result['matched_duration_s'] = float(matched_dur_s) if matched_dur_s is not None else np.nan
            result['match_confidence'] = float(confidence) if confidence is not None else np.nan
            result['final_radius_m'] = current_radiuses[0]
            result['osrm_match_result'] = match_summary

            if check_tolerance and pd.notna(result['matched_distance_m']):
                odo_distance_m = odo_distance_km * 1000.0
                if odo_distance_m > 10:
                    tolerance_m = (tolerance_pct / 100.0) * odo_distance_m
                    distance_diff = abs(result['matched_distance_m'] - odo_distance_m)
                    if distance_diff <= tolerance_m:
                        result['match_status'] = 'OK_InTolerance'
                        return result
                    else: # Failed tolerance check
                        if attempt == max_attempts - 1: result['match_status'] = 'OK_ToleranceFail'; return result
                        else: current_radiuses = [min(r * radius_multiplier, max_radius_m) for r in current_radiuses]; continue
                else: result['match_status'] = 'OK_OdoTooSmall'; return result
            else: # Odo invalid or match distance NaN, accept first valid match
                result['match_status'] = 'OK_NoToleranceCheck' if pd.notna(result['matched_distance_m']) else 'Error_MatchDistanceNaN'
                return result

        elif response_data.get('code') == 'NoMatch':
            if attempt == max_attempts - 1: result['match_status'] = 'NoMatch'; return result
            else: current_radiuses = [min(r * radius_multiplier, max_radius_m) for r in current_radiuses]; continue
        else: # Other API error code
            if attempt == max_attempts - 1: result['match_status'] = f"Error_{response_data.get('code', 'Unknown')}"; return result
            else: current_radiuses = [min(r * radius_multiplier, max_radius_m) for r in current_radiuses]; continue

    result['match_status'] = 'Error_MaxAttemptsReached'
    return result

# --- Speed Limit Analysis Helper ---
def analyze_speeding(segment_kinematics, osrm_match_result):
    """Estimates time spent speeding based on OSRM matched route speed limits."""
    # (Keep implementation from previous step - it uses the 'osrm_match_result' dict)
    # ... [Copy implementation from previous step] ...
    results = {'percent_time_over_limit_seg_agg': np.nan, 'avg_speed_limit_kph_seg_agg': np.nan}
    if osrm_match_result is None or 'legs' not in osrm_match_result or not isinstance(segment_kinematics, pd.DataFrame) or segment_kinematics.empty:
        return results
    try:
        total_segment_duration = segment_kinematics['dt_s'].sum()
        if total_segment_duration <= 0: return results
        time_over_limit_s = 0.0; weighted_speed_limit_sum = 0.0; total_leg_duration = 0.0
        tracepoint_indices = [wp['waypoint_index'] for wp in osrm_match_result.get('tracepoints', []) if wp is not None]
        if len(tracepoint_indices) != len(segment_kinematics): pass # warnings.warn("Mismatch tracepoints/kinematics rows.")
        current_tracepoint_idx = 0
        for leg in osrm_match_result.get('legs', []):
            annotation = leg.get('annotation', {})
            # OSRM uses 'speed' for speed limits in annotations if available
            maxspeeds = annotation.get('speed', []) # Check 'speed' first
            if not maxspeeds: maxspeeds = annotation.get('maxspeed', []) # Fallback to 'maxspeed' if 'speed' not present
            durations = annotation.get('duration', [])
            if not maxspeeds or not durations or len(maxspeeds) != len(durations): continue
            num_annotation_segments = len(maxspeeds)
            leg_duration_sum = sum(d for d in durations if d is not None)
            total_leg_duration += leg_duration_sum
            for k in range(num_annotation_segments):
                speed_limit_kph = maxspeeds[k] if maxspeeds[k] is not None else np.inf
                duration_s = durations[k] if durations[k] is not None else 0
                if isinstance(speed_limit_kph, str) and speed_limit_kph.lower() == 'none': speed_limit_kph = np.inf
                try: speed_limit_kph = float(speed_limit_kph)
                except: speed_limit_kph = np.inf
                speed_limit_mps = speed_limit_kph * (1000/3600)
                if pd.notna(speed_limit_kph) and speed_limit_kph != np.inf: weighted_speed_limit_sum += speed_limit_kph * duration_s
                start_trace_idx = current_tracepoint_idx + k; end_trace_idx = current_tracepoint_idx + k + 1
                if start_trace_idx < len(tracepoint_indices) and end_trace_idx < len(tracepoint_indices):
                    original_start_row_idx = tracepoint_indices[start_trace_idx]; original_end_row_idx = tracepoint_indices[end_trace_idx]
                    try: relevant_kinematics = segment_kinematics.iloc[original_start_row_idx:original_end_row_idx]
                    except IndexError: continue
                    if not relevant_kinematics.empty:
                        is_speeding = relevant_kinematics['speed_mps'] > speed_limit_mps
                        time_over_limit_s += relevant_kinematics.loc[is_speeding, 'dt_s'].sum()
            current_tracepoint_idx += num_annotation_segments # Approximate advancement
        if total_segment_duration > 0: results['percent_time_over_limit_seg_agg'] = np.clip((time_over_limit_s / total_segment_duration) * 100, 0, 100)
        if total_leg_duration > 0: results['avg_speed_limit_kph_seg_agg'] = weighted_speed_limit_sum / total_leg_duration
    except Exception as e: warnings.warn(f"Error during speed limit analysis: {e}")
    results['percent_time_over_limit_seg_agg'] = results.get('percent_time_over_limit_seg_agg', 0.0)
    if pd.isna(results['percent_time_over_limit_seg_agg']): results['percent_time_over_limit_seg_agg'] = 0.0
    return results

# --- Helper function for trip-level split ---
def split_trips_train_test(segments_df, trip_id_col, test_size_trips=0.2, random_state=None):
    """Splits unique trip IDs into training and testing sets."""
    unique_trips = segments_df[trip_id_col].unique()
    if len(unique_trips) < 2: # Cannot split if only one trip or no trips
        warnings.warn("Not enough unique trips to perform a train/test split. Using all trips for both if possible.")
        return unique_trips, unique_trips # Or handle as an error

    # Ensure test_size_trips results in at least one trip in each set if possible
    if int(len(unique_trips) * test_size_trips) < 1 and len(unique_trips) > 1:
        # Ensure at least one test trip if there are multiple trips
        num_test_trips = 1
    elif int(len(unique_trips) * (1 - test_size_trips)) < 1 and len(unique_trips) > 1:
        # Ensure at least one training trip
        num_test_trips = len(unique_trips) - 1
    else:
        num_test_trips = int(len(unique_trips) * test_size_trips)

    if num_test_trips == 0 and len(unique_trips) > 0: # If only 1 trip, test_size might make it 0
        num_test_trips = 1 if len(unique_trips) > 1 else 0 # Ensure test has 1 if possible, else 0
    
    num_train_trips = len(unique_trips) - num_test_trips
    if num_train_trips < 1 and len(unique_trips) > num_test_trips : # Ensure train has at least 1 if possible
        num_train_trips = 1
        num_test_trips = len(unique_trips) - 1


    if random_state is not None:
        np.random.seed(random_state)
    
    shuffled_trips = np.random.permutation(unique_trips)
    
    # Ensure split logic handles small numbers of trips correctly
    if num_test_trips >= len(shuffled_trips): # If test size is too large, assign all but one to test (if >1 trips)
        test_trip_ids = shuffled_trips
        train_trip_ids = np.array([]) if len(shuffled_trips) <=1 else shuffled_trips[:1] # Keep at least one for train if possible
        if len(shuffled_trips) > 1 and num_test_trips == len(shuffled_trips): # if all are test, move one to train
            train_trip_ids = np.array([shuffled_trips[0]])
            test_trip_ids = shuffled_trips[1:]

    elif num_train_trips >= len(shuffled_trips): # If train size is too large
        train_trip_ids = shuffled_trips
        test_trip_ids = np.array([]) if len(shuffled_trips) <=1 else shuffled_trips[:1]
        if len(shuffled_trips) > 1 and num_train_trips == len(shuffled_trips):
            test_trip_ids = np.array([shuffled_trips[0]])
            train_trip_ids = shuffled_trips[1:]
    else:
        test_trip_ids = shuffled_trips[:num_test_trips]
        train_trip_ids = shuffled_trips[num_test_trips:]

    if len(train_trip_ids) == 0 and len(test_trip_ids) > 0: # Ensure train is not empty if test has trips
        train_trip_ids = np.array([test_trip_ids[0]])
        test_trip_ids = test_trip_ids[1:]
    elif len(test_trip_ids) == 0 and len(train_trip_ids) > 0: # Ensure test is not empty if train has trips
        test_trip_ids = np.array([train_trip_ids[0]])
        train_trip_ids = train_trip_ids[1:]


    print(f"   Trip split: {len(train_trip_ids)} train trip(s), {len(test_trip_ids)} test trip(s).")
    if len(train_trip_ids) == 0 or len(test_trip_ids) == 0 and len(unique_trips) > 1:
        warnings.warn(f"Train/Test split resulted in an empty set (Train: {len(train_trip_ids)}, Test: {len(test_trip_ids)}) for {len(unique_trips)} unique trips. This may cause issues.")

    return train_trip_ids, test_trip_ids

def remove_highly_collinear_numerical_features(df_numerical, threshold=0.90):
    """
    Removes highly collinear numerical features from a DataFrame with verbose output.

    Args:
        df_numerical (pd.DataFrame): DataFrame containing only numerical features.
        threshold (float): Correlation threshold above which features are considered collinear.

    Returns:
        pd.DataFrame: DataFrame with highly collinear features removed.
        list: List of column names that were removed.
    """
    if df_numerical.empty or df_numerical.shape[1] < 2:
        return df_numerical, []

    print(f"   Applying collinearity removal to {df_numerical.shape[1]} numerical features (threshold={threshold})...")
    
    # Work on a copy to avoid modifying the original df passed to this function during iterations
    df_to_filter = df_numerical.copy()
    
    to_drop_overall = set()
    removed_features_overall = []

    iteration = 0
    while True: # Keep iterating until no more features are dropped in an iteration
        iteration += 1
        # print(f"     Collinearity Iteration {iteration}")
        if df_to_filter.shape[1] < 2: break # Not enough columns to compare

        corr_matrix = df_to_filter.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        dropped_in_this_iteration = False
        
        # Iterate through columns to find the first highly correlated pair not yet processed
        # Sorting columns ensures a deterministic order of processing pairs
        sorted_columns = sorted(upper.columns)

        for i in range(len(sorted_columns)):
            col1 = sorted_columns[i]
            if col1 in to_drop_overall: continue

            for j in range(i + 1, len(sorted_columns)):
                col2 = sorted_columns[j]
                if col2 in to_drop_overall: continue

                if upper.loc[col1, col2] > threshold: # Found a highly correlated pair
                    print(f"     - Pair: ('{col1}', '{col2}') correlation = {upper.loc[col1, col2]:.4f} (> {threshold})")
                    
                    # Heuristic: Drop the one with higher average correlation with *other remaining* features
                    # Calculate average correlation for col1 with other features (excluding col2 and already dropped)
                    other_features_for_col1 = list(set(df_to_filter.columns) - {col1, col2})
                    avg_corr_col1 = 0
                    if other_features_for_col1:
                         avg_corr_col1 = df_to_filter[other_features_for_col1].corrwith(df_to_filter[col1]).abs().mean()
                         if pd.isna(avg_corr_col1): avg_corr_col1 = 0 # Handle case with no other features or all NaN corrs
                    
                    # Calculate average correlation for col2 with other features (excluding col1 and already dropped)
                    other_features_for_col2 = list(set(df_to_filter.columns) - {col1, col2})
                    avg_corr_col2 = 0
                    if other_features_for_col2:
                        avg_corr_col2 = df_to_filter[other_features_for_col2].corrwith(df_to_filter[col2]).abs().mean()
                        if pd.isna(avg_corr_col2): avg_corr_col2 = 0

                    print(f"       - Avg Abs Corr with others: '{col1}'={avg_corr_col1:.4f}, '{col2}'={avg_corr_col2:.4f}")

                    if avg_corr_col1 >= avg_corr_col2: # Drop col1 (or if equal, drop the one appearing first in sorted list)
                        feature_to_drop_from_pair = col1
                        feature_to_keep_from_pair = col2
                    else: # Drop col2
                        feature_to_drop_from_pair = col2
                        feature_to_keep_from_pair = col1
                    
                    print(f"       - Dropping '{feature_to_drop_from_pair}', Keeping '{feature_to_keep_from_pair}' from this pair.")
                    to_drop_overall.add(feature_to_drop_from_pair)
                    removed_features_overall.append(feature_to_drop_from_pair)
                    df_to_filter = df_to_filter.drop(columns=[feature_to_drop_from_pair])
                    dropped_in_this_iteration = True
                    break # Restart inner loop (j) because columns changed
            if dropped_in_this_iteration:
                break # Restart outer loop (i) because columns changed

        if not dropped_in_this_iteration:
            break # No more features dropped in a full pass, stable now

    print(f"   Collinearity removal dropped {len(removed_features_overall)} numerical features in total.")
    return df_numerical.drop(columns=list(to_drop_overall), errors='ignore'), removed_features_overall

def display_and_save_feature_importances(model, feature_names, model_name_label, output_dir, top_n=20):
    """
    Calculates, displays, and saves a bar plot of feature importances.

    Args:
        model: The trained model object.
        feature_names (list): List of feature names corresponding to the model's input.
        model_name_label (str): A label for the model (e.g., "RandomForest_LOSO_Global").
        output_dir (str): Directory to save the plot.
        top_n (int): Number of top features to display.
    """
    if hasattr(model, 'feature_importances_'): # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'): # Linear models
        importances = np.abs(model.coef_)
        if importances.ndim > 1 and importances.shape[0] == 1: # For some linear models coef_ is 2D
            importances = importances.flatten()
    else:
        warnings.warn(f"Model {type(model).__name__} does not have 'feature_importances_' or 'coef_'. Skipping importance plot.")
        return

    if len(importances) != len(feature_names):
        warnings.warn(f"Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}). Skipping importance plot.")
        # print(f"DEBUG: Importances: {importances[:5]}")
        # print(f"DEBUG: Feature Names: {feature_names[:5]}")
        return

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n)

    print(f"\n--- Top {top_n} Feature Importances for {model_name_label} ---")
    print(importance_df)

    plt.figure(figsize=(10, max(6, top_n * 0.35))) # Adjust height based on top_n
    sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
    plt.title(f'Top {top_n} Feature Importances: {model_name_label}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    plot_filename = f"feature_importance_{model_name_label.replace(' ', '_').lower()}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path, dpi=150)
        print(f"Saved feature importance plot to: {plot_path}")
    except Exception as e:
        warnings.warn(f"Could not save feature importance plot: {e}")
    # plt.show() # Optionally show plot
    plt.close()

def plot_trip_speed_profiles(df, trip_ids, col, y_max, output_dir, trip_id_col='trip_id', plots_per_figure=8):
    """
    Generates and saves grids of line plots for all specified trip IDs,
    batching them into multiple image files.

    Args:
        df (pd.DataFrame): The DataFrame containing the trip data.
        trip_ids (list or np.ndarray): An array of all trip IDs to plot.
        col (str): The name of the array-like column to plot (e.g., 'speed_array').
        y_max (float): The maximum valid value to draw as a reference line.
        output_dir (str): The directory where the plot images will be saved.
        trip_id_col (str): The name of the trip identifier column.
        plots_per_figure (int): The number of subplots to include in each saved image file.
    """
    if not trip_ids.any():
        return

    # --- Create the output directory if it doesn't exist ---
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"   Warning: Could not create directory {output_dir}. Error: {e}")
        return

    print(f"   Generating speed profile plots for {len(trip_ids)} affected trips...")
    
    # --- Loop through the trip IDs in batches ---
    for i in range(0, len(trip_ids), plots_per_figure):
        chunk_of_trip_ids = trip_ids[i : i + plots_per_figure]
        
        num_plots_in_chunk = len(chunk_of_trip_ids)
        num_cols = 2
        num_rows = (num_plots_in_chunk + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False, tight_layout=True)
        axes = axes.flatten()

        for j, trip_id in enumerate(chunk_of_trip_ids):
            ax = axes[j]
            trip_df = df[df[trip_id_col] == trip_id]
            
            raw_values = list(chain.from_iterable(trip_df[col].dropna()))

            if not raw_values:
                ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            else:
                ax.plot(raw_values, label='Raw Speed', color='dodgerblue', linewidth=1)
                ax.axhline(y=y_max, color='red', linestyle='--', label=f'Max Limit ({y_max})')
                ax.legend()
                # Adjust y-axis to make sure the spike is visible
                plot_upper_bound = max(max(raw_values) + 50, y_max * 1.2)
                ax.set_ylim(-10, plot_upper_bound)

            ax.set_title(f"Trip: {trip_id}", fontsize=10)
            ax.set_xlabel("Sample Index (Time)")
            ax.set_ylabel("Raw Speed (kph)")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Hide any unused subplots in the last figure
        for k in range(num_plots_in_chunk, len(axes)):
            fig.delaxes(axes[k])

        # --- Save the current figure ---
        batch_num = (i // plots_per_figure) + 1
        plot_filename = f"{col}_profiles_batch_{batch_num}.png"
        full_path = os.path.join(output_dir, plot_filename)
        
        try:
            plt.savefig(full_path, dpi=120)
            print(f"   Saved diagnostic plot batch {batch_num} to: {full_path}")
        except Exception as e:
            print(f"   Warning: Could not save diagnostic plot {full_path}. Error: {e}")
        
        plt.close(fig) # Close the figure to free up memory for the next batch

def count_outliers_in_array(arr, min_val, max_val):
    """Counts how many numeric values in an array are outside the given bounds."""
    if not isinstance(arr, (list, np.ndarray)):
        return 0
    
    # Use a generator expression with sum() for a concise and efficient count
    return sum(1 for val in arr if isinstance(val, (int, float)) and not (min_val <= val <= max_val))

# Helper to safely get the length of an array/list, returns -1 for non-array types or NaNs
def get_array_length(arr):
    if isinstance(arr, (list, np.ndarray)):
        return len(arr)
    return -1  # Use -1 to signify 'not an array' or 'is NaN'

########################### Misalignment Report Helpers #########################################

def safe_apply(series: pd.Series, func: Callable) -> pd.Series:
    """Applies a function to each element of a series, handling non-iterable types."""
    return pd.Series([func(arr) if isinstance(arr, (list, np.ndarray)) else np.nan for arr in series], index=series.index)

safe_first = lambda arr: arr[0]  if isinstance(arr, (list, np.ndarray)) and len(arr) > 0 else np.nan
safe_last  = lambda arr: arr[-1] if isinstance(arr, (list, np.ndarray)) and len(arr) > 0 else np.nan

def safe_max_internal_delta(arr: Any) -> float:
    """Calculates the maximum absolute consecutive change within a numeric array."""
    if not isinstance(arr, (list, np.ndarray)) or len(arr) < 2: return np.nan
    numeric_arr = pd.to_numeric(np.asarray(arr), errors='coerce')
    numeric_arr = numeric_arr[~np.isnan(numeric_arr)]
    if len(numeric_arr) < 2: return np.nan
    return np.nanmax(np.abs(np.diff(numeric_arr)))

# --- Configuration and Rule DSL (Domain-Specific Language) ---
@dataclass(frozen=True)
class MisalignmentConfig:
    """A frozen (immutable) dataclass to standardize column names used in the analysis."""
    trip_id: str = "trip_id"
    time: str = "timestamp"
    lat: str = "latitude"
    lon: str = "longitude"
    odo: str = "current_odo"
    speed_array: str = "speed_array"
    soc: str = "current_soc"
    alt_array: str = "altitude_array"
    battery_capacity_kWh_col: str = "battery_capacity_kWh"

@dataclass(frozen=True)
class Rule:
    """Defines a rule with a name, the columns it acts on, and the boolean function to apply."""
    name: str
    lhs: str
    rhs: str
    fn: Callable[[pd.Series, pd.Series], pd.Series]

def _safe_division(a: pd.Series, b: pd.Series, tol: float = 1e-6) -> pd.Series:
    """Safely divides two series, returning np.nan where the denominator is near zero."""
    valid_mask = b.abs() > tol
    out = pd.Series(np.nan, index=a.index)
    out.loc[valid_mask] = a[valid_mask] / b[valid_mask]
    return out

def _haversine_np(lon1: pd.Series, lat1: pd.Series, lon2: pd.Series, lat2: pd.Series) -> np.ndarray:
    """Calculates Haversine distance in a vectorized way."""
    # POINT 1 FIX: Convert to numpy arrays to ensure correct dtype for trigonometric functions
    lon1, lat1, lon2, lat2 = lon1.to_numpy(), lat1.to_numpy(), lon2.to_numpy(), lat2.to_numpy()
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def calculate_all_deltas(df: pd.DataFrame, cfg: MisalignmentConfig) -> pd.DataFrame:
    """Calculates all necessary deltas in a fully vectorized way, respecting trip boundaries."""
    # POINT 5 FIX: Add reset_index to guarantee unique labels and prevent errors
    df_sorted = df.sort_values([cfg.trip_id, cfg.time]).reset_index(drop=True)
    
    grouped = df_sorted.groupby(cfg.trip_id, observed=True)
    is_first_row = grouped.cumcount() == 0

    deltas = pd.DataFrame(index=df_sorted.index)

    # ... (rest of the delta calculations are the same) ...
    deltas['Δ_time_s'] = grouped[cfg.time].diff().dt.total_seconds()
    deltas['Δ_odo_km'] = grouped[cfg.odo].diff()
    deltas['Δ_soc_pct'] = grouped[cfg.soc].diff()
    lon1, lat1 = grouped[cfg.lon].shift(), grouped[cfg.lat].shift()
    deltas['Δ_gps_km'] = _haversine_np(lon1, lat1, df_sorted[cfg.lon], df_sorted[cfg.lat])
    deltas['Δ_dist_from_speed_km'] = safe_apply(df_sorted[cfg.speed_array], lambda arr: np.nansum(arr) / 3600.0)
    deltas['speed_max_internal_delta_kph'] = safe_apply(df_sorted[cfg.speed_array], safe_max_internal_delta)
    deltas['alt_max_internal_delta_m'] = safe_apply(df_sorted[cfg.alt_array], safe_max_internal_delta)
    alt_means = safe_apply(df_sorted[cfg.alt_array], np.nanmean)
    deltas['Δ_alt_m'] = alt_means.groupby(df_sorted[cfg.trip_id]).diff()
    first_speed = safe_apply(df_sorted[cfg.speed_array], safe_first)
    last_speed_shifted = safe_apply(df_sorted[cfg.speed_array], safe_last).groupby(df_sorted[cfg.trip_id]).shift()
    deltas['speed_edge_delta_kph'] = first_speed - last_speed_shifted
    first_alt = safe_apply(df_sorted[cfg.alt_array], safe_first)
    last_alt_shifted = safe_apply(df_sorted[cfg.alt_array], safe_last).groupby(df_sorted[cfg.trip_id]).shift()
    deltas['alt_edge_delta_m'] = first_alt - last_alt_shifted
    if cfg.battery_capacity_kWh_col in df_sorted.columns:
        deltas['Δ_energy_consumed_kWh'] = (deltas['Δ_soc_pct'] / 100.0) * df_sorted[cfg.battery_capacity_kWh_col]

    # Mask out all deltas for the first row of each trip
    deltas.loc[is_first_row, :] = np.nan
    return deltas

def evaluate_rules(deltas_df: pd.DataFrame, rules: List[Rule]) -> pd.DataFrame:
    """Applies a list of rules to the deltas DataFrame, returning a boolean flag matrix."""
    flags_df = pd.DataFrame(index=deltas_df.index, dtype=bool)
    for rule in rules:
        if rule.lhs in deltas_df.columns and rule.rhs in deltas_df.columns:
            lhs, rhs = deltas_df[rule.lhs], deltas_df[rule.rhs]
            valid_mask = lhs.notna() & rhs.notna()
            flags = pd.Series(False, index=deltas_df.index)
            if valid_mask.any():
                flags.loc[valid_mask] = rule.fn(lhs[valid_mask], rhs[valid_mask])
            flags_df[rule.name] = flags.fillna(False)
    return flags_df

# --- Rule Definition Helpers (The DSL) ---
def ratio_rule(lhs: str, rhs: str, max_ratio: float, min_move: float = 0.05) -> Rule:
    """Creates a symmetric ratio rule, but only for values above a minimum movement threshold."""
    def rule_fn(a: pd.Series, b: pd.Series) -> pd.Series:
        movement_mask = (a.abs() > min_move) | (b.abs() > min_move)
        flags = pd.Series(False, index=a.index)
        if movement_mask.any():
            a_m, b_m = a[movement_mask].abs(), b[movement_mask].abs()
            # POINT 2 FIX: Use np.maximum/minimum and reconstruct Series to preserve index
            num = np.maximum(a_m, b_m)
            den = np.minimum(a_m, b_m)
            ratio = pd.Series(_safe_division(num, den), index=a_m.index)
            flags.loc[movement_mask] = (ratio > max_ratio).fillna(False)
        return flags
    return Rule(name=f"Ratio_{lhs}_vs_{rhs}", lhs=lhs, rhs=rhs, fn=rule_fn)

def simple_threshold_rule(name: str, col: str, threshold: float, op: str = 'gt') -> Rule:
    """Creates a rule for a simple comparison (gt, lt, eq, abs_gt)."""
    ops = {'gt': lambda x: x > threshold, 'lt': lambda x: x < threshold, 'eq': lambda x: x == threshold, 'abs_gt': lambda x: x.abs() > threshold}
    return Rule(name=name, lhs=col, rhs=col, fn=lambda a, b: ops[op](a))

def grade_rule(name: str, dist_col: str, max_grade: float = 0.25, min_move_km: float = 0.05) -> Rule:
    """Creates a rule to check for physically impossible road grades."""
    # POINT 3 FIX: Renamed parameters for clarity
    def rule_fn(alt_delta: pd.Series, dist_km: pd.Series) -> pd.Series:
        movement_mask = dist_km.abs() >= min_move_km
        grade = _safe_division(alt_delta, dist_km * 1000)
        return movement_mask & (grade.abs() > max_grade)
    return Rule(name=name, lhs='Δ_alt_m', rhs=dist_col, fn=rule_fn)

def energy_per_100km_rule(name: str, min_kWh_100km: float, max_kWh_100km: float, min_move_km: float = 0.5) -> Rule:
    """Flags energy consumption (kWh/100km) outside a plausible physical range."""
    def rule_fn(energy_kWh: pd.Series, dist_km: pd.Series) -> pd.Series:
        movement_mask = (dist_km.abs() > min_move_km)
        # POINT 4 FIX: Only check for consumption (energy_kWh is negative), not regeneration.
        consumption_mask = energy_kWh < 0
        
        # Note: A negative energy delta means consumption (SoC went down).
        kwh_per_100km = _safe_division(-energy_kWh, dist_km) * 100
        consumption_outlier = (kwh_per_100km < min_kWh_100km) | (kwh_per_100km > max_kWh_100km)
        
        return movement_mask & consumption_mask & consumption_outlier
    return Rule(name=name, lhs='Δ_energy_consumed_kWh', rhs='Δ_gps_km', fn=rule_fn)

# POINT 6: Add helper functions for the new powerful rules
def speed_spike_rule(limit: float = 40.0) -> Rule:
    """Rule for excessive within-array acceleration (e.g., >40 km/h in one second)."""
    return simple_threshold_rule(name="Speed_Spike_Internal", col="speed_max_internal_delta_kph", threshold=limit, op='gt')

def alt_spike_across_rows_rule(limit: float = 25.0) -> Rule:
    """Rule for a large altitude jump between consecutive rows (GPS glitch, tunnel)."""
    return simple_threshold_rule(name="Altitude_Row_Spike", col="Δ_alt_m", threshold=limit, op='abs_gt')

# (Other rule helpers like charging_while_moving_rule, uphill_no_energy_rule remain the same)
def charging_while_moving_rule(min_regen_pct: float = 0.01, min_dist_km: float = 0.01) -> Rule:
    def rule_fn(soc_delta: pd.Series, dist_delta: pd.Series) -> pd.Series:
        is_charging = soc_delta > min_regen_pct
        is_moving = dist_delta > min_dist_km
        return is_charging & is_moving
    return Rule(name="Charging_While_Moving", lhs="Δ_soc_pct", rhs="Δ_gps_km", fn=rule_fn)

def uphill_no_energy_rule(min_alt_gain_m: float = 20.0, max_soc_gain_pct: float = 0.1) -> Rule:
    def rule_fn(alt_delta: pd.Series, soc_delta: pd.Series) -> pd.Series:
        is_uphill = alt_delta > min_alt_gain_m
        not_discharging = soc_delta < max_soc_gain_pct
        return is_uphill & not_discharging
    return Rule(name="Uphill_No_Discharge", lhs="Δ_alt_m", rhs="Δ_soc_pct", fn=rule_fn)

def time_gap_rule(min_s: float, max_s: float) -> Rule:
    """Flags rows where the time delta is outside the normal [min_s, max_s] band."""
    def rule_fn(dt: pd.Series, _unused: pd.Series) -> pd.Series:
        return (dt < min_s) | (dt > max_s)
    return Rule(name="Time_Gap_Anomaly", lhs="Δ_time_s", rhs="Δ_time_s", fn=rule_fn)

def descent_no_speed_change_rule(min_alt_loss_m: float = 50.0, max_speed_delta_kph: float = 5.0) -> Rule:
    """Flags steep descents where the vehicle's speed barely changes."""
    def rule_fn(alt_delta: pd.Series, speed_delta: pd.Series) -> pd.Series:
        is_steep_descent = alt_delta < -min_alt_loss_m
        speed_is_stuck = speed_delta.abs() < max_speed_delta_kph
        return is_steep_descent & speed_is_stuck
    # Note: We use speed_max_internal_delta_kph to check for any speed change within the minute.
    return Rule(name="Descent_No_Speed_Change", lhs="Δ_alt_m", rhs="speed_max_internal_delta_kph", fn=rule_fn)

def speed_spike_while_stationary_rule(speed_spike_thr: float = 30.0, max_dist_km: float = 0.02) -> Rule:
    """Flags a large internal speed spike when the vehicle is effectively stationary."""
    def rule_fn(speed_spike: pd.Series, gps_dist: pd.Series) -> pd.Series:
        return (speed_spike > speed_spike_thr) & (gps_dist.abs() < max_dist_km)
    return Rule(name="Speed_Spike_While_Stationary", lhs="speed_max_internal_delta_kph", rhs="Δ_gps_km", fn=rule_fn)

def odo_jump_short_time_rule(odo_jump_km: float = 1.0, max_time_s: float = 10.0) -> Rule:
    """Flags a large odometer jump occurring in a very short time (implies impossible speed)."""
    def rule_fn(odo_delta: pd.Series, time_delta: pd.Series) -> pd.Series:
        return (odo_delta.abs() > odo_jump_km) & (time_delta < max_time_s)
    return Rule(name="Odo_Jump_Short_Time", lhs="Δ_odo_km", rhs="Δ_time_s", fn=rule_fn)

def teleport_with_no_energy_change_rule(teleport_dist_km: float = 2.0, max_soc_change_pct: float = 0.5) -> Rule:
    """Flags a large GPS jump that has no corresponding energy change (e.g., ferry, transport)."""
    def rule_fn(gps_delta: pd.Series, soc_delta: pd.Series) -> pd.Series:
        return (gps_delta > teleport_dist_km) & (soc_delta.abs() < max_soc_change_pct)
    return Rule(name="Teleport_With_No_Energy_Change", lhs="Δ_gps_km", rhs="Δ_soc_pct", fn=rule_fn)

def unrealistic_regen_on_descent_rule(min_alt_loss_m: float = 100.0, max_regen_pct: float = 2.0) -> Rule:
    """Flags an implausibly high amount of regeneration during a steep descent."""
    def rule_fn(alt_delta: pd.Series, soc_delta: pd.Series) -> pd.Series:
        # A negative soc_delta means regen (gaining charge), so we check if it's "too negative".
        return (alt_delta < -min_alt_loss_m) & (soc_delta < -max_regen_pct)
    return Rule(name="Unrealistic_Regen_On_Descent", lhs="Δ_alt_m", rhs="Δ_soc_pct", fn=rule_fn)

# --- Main Entry Point and Reporting ---
def analyze_feature_misalignment(df: pd.DataFrame, cfg: MisalignmentConfig, rules: List[Rule]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("--- Analyzing Feature Misalignments (V3 - Hardened) ---")
    deltas_df = calculate_all_deltas(df, cfg)

    if not deltas_df.columns.is_unique:
        dupes = deltas_df.columns[deltas_df.columns.duplicated()].unique()
        raise RuntimeError(f"Duplicate delta columns: {dupes}")

    flags_df = evaluate_rules(deltas_df, rules)
    return deltas_df, flags_df

def build_report(df: pd.DataFrame, deltas: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    if not flags.any().any(): return pd.DataFrame()
    flagged_rows = flags.any(axis=1)
    # Ensure columns from all three dataframes are concatenated correctly
    return pd.concat([df.loc[flagged_rows], deltas.loc[flagged_rows], flags.loc[flagged_rows]], axis=1)

def print_misalignment_report(report_df: pd.DataFrame, raw_df: pd.DataFrame, deltas: pd.DataFrame, all_rules: List[Rule], cfg: MisalignmentConfig, context_window: int, max_trips: int):
    if report_df.empty:
        print("No misalignments found."); return
    
    flag_cols = [r.name for r in all_rules if r.name in report_df.columns]
    flagged_trips = report_df[cfg.trip_id].unique()
    
    print(f"\n--- Misalignment Report Summary ---")
    print(f"Trips checked : {raw_df[cfg.trip_id].nunique():>5}")
    print(f"Trips flagged : {len(flagged_trips):>5}")
    print(f"Rows flagged  : {len(report_df):>5}\n")

    for i, trip_id in enumerate(flagged_trips):
        if i >= max_trips: print(f"... (omitting details for {len(flagged_trips) - max_trips} more trips)"); break
        
        print(f"--- Details for Trip ID: {trip_id} ---")
        trip_report_df = report_df[report_df[cfg.trip_id] == trip_id]
        
        # Use the full deltas dataframe for context, not just the flagged rows
        full_trip_context_df = pd.concat([
            raw_df[raw_df[cfg.trip_id] == trip_id],
            deltas[raw_df[cfg.trip_id] == trip_id]
        ], axis=1)

        for idx in trip_report_df.index:
            fired_rules = [rule for rule in flag_cols if trip_report_df.loc[idx, rule]]
            print(f"\n  - Row Index {idx} ({trip_report_df.loc[idx, cfg.time]}): VIOLATIONS -> {fired_rules}")
            
            try:
                row_loc = full_trip_context_df.index.get_loc(idx)
                start_loc = max(0, row_loc - context_window)
                end_loc = min(len(full_trip_context_df), row_loc + 1 + context_window)
                
                context_df = full_trip_context_df.iloc[start_loc:end_loc]
                print("    Context (Raw Data + Deltas):")
                display_cols = [cfg.time, cfg.odo, cfg.soc] + [c for c in context_df.columns if c.startswith('Δ_')]
                # Ensure display_cols exist before trying to print
                display_cols_exist = [c for c in display_cols if c in context_df.columns]
                print(context_df[display_cols_exist].to_string(float_format="%.3f"))
            except KeyError:
                print(f"    Could not find index {idx} in the context DataFrame for trip {trip_id}.")