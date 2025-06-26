import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import h3
import requests
import os
import numpy as np

# --- Define Standard Column Names (CORRECTED FOR RAW DATA) ---
# Using the ORIGINAL column names from df_sample.parquet
TRIP_ID_COL = 'CYCLE_ID'
TIMESTAMP_COL = 'HEAD_COLL_TIMS'
LOCATION_COL = 'geoindex_10'
SPEED_ARRAY_COL = 'PERD_VEHC_LONG_SPD' # This is the original name for the speed array

# --- Data Loading & OSRM Functions ---

def load_data_from_local_file(file_path):
    """Loads a Parquet DataFrame from a local file path."""
    if not os.path.exists(file_path):
        st.error(f"Data file not found at: {file_path}")
        st.info(f"Please update the `LOCAL_DATA_PATH` variable in the script to point to your main Parquet file.")
        return None
    try:
        df = pd.read_parquet(file_path)
        st.success("Local data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Failed to read the Parquet file: {e}")
        return None

def get_osrm_match_route(coordinates, timestamps):
    """Gets the matched route using OSRM match service."""
    coord_str = ";".join([f"{lng},{lat}" for lat, lng in coordinates])
    ts_str = ";".join([str(ts) for ts in timestamps])
    radiuses_str = "&radiuses=" + ";".join(["30"] * len(coordinates))
    url = (
        f"http://router.project-osrm.org/match/v1/driving/{coord_str}"
        f"?timestamps={ts_str}{radiuses_str}&gaps=split&tidy=true"
        "&overview=full&geometries=geojson"
    )
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'matchings' in data and len(data['matchings']) > 0:
                return data['matchings'][0]
        else:
            st.warning(f"OSRM Match Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to OSRM service: {e}")
    return None

# --- Helper for Speed Outlier Check ---
def has_speed_error(row, speed_col, min_val, max_val):
    """Checks if any value in a speed array of a given row is out of range."""
    speed_array = row.get(speed_col)
    if not isinstance(speed_array, (list, np.ndarray)):
        return False
    for val in speed_array:
        if isinstance(val, (int, float)) and not (min_val <= val <= max_val):
            return True
    return False

# --- Main Diagnostic App ---
def main():
    st.title("Trip Data Diagnostic Plotter")

    st.sidebar.header("Select Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose a diagnostic to view:",
        ("At Sea GPS Analysis", "Speed Outlier Analysis")
    )

    at_sea_trip_ids = ['578de678-5ab8-5f66-890f-a7f4bd5445e7', '2b5ec2c6-dcd5-5ffb-b4d3-56ca0369bd66', 'ea86decd-1f16-58a7-b46a-83fb4890794d', '8de496a7-fdb8-56e3-8b6d-630636d1fae8']
    speed_outlier_trip_ids = ['180dbd02-027b-5988-a501-50a74cc32ad7', 'fdeb00c2-fad6-5395-8800-6b642d305b2a', '42a5e014-6443-5d9f-ae66-0375328e282e', '1890065f-86a7-54a1-8f4a-90725f5cabcc', '10fca1f0-2c95-5026-bb27-f7251c59c4e8', '10e9d30a-3465-5506-9d47-faed0caafce4', '68bdd3a2-a92e-587f-9fb4-41f708e23aa8', '5f2e9c5c-b4aa-564d-9d20-2a4d2d20df9a', 'bc81dd74-141a-56b5-883e-043d36f14ef8', '700cf316-1738-5350-816b-94d1bd52c301', 'afb8ef9e-5001-5796-81d7-287624890f02', '18a45b9c-4ca5-550d-a6e6-b59aa4cc149f', 'd2ad2a7f-d743-588d-81d4-e2ab903dc9a5', '11e35c00-e94c-5de8-880f-54c85cef95d5', 'd4c85538-8bd9-5a64-bd9b-4261c3485c94', 'b6651e50-3fc8-5933-b550-8353054e7745', '4fd43cb1-b222-5433-a817-782896ea147c', 'dc66abae-dfca-5900-b454-6f286e6cdf15', 'b6903c5e-cb7c-54cf-b3f6-b18ff38242d4', '063edcf2-a87e-5e98-95be-f4c205b97651', '632fecb6-5068-557b-8f11-df2cbc9eb3d1', 'f9315a56-4aff-57ff-ae5c-b6aba3d9a878', 'f86919fe-6fab-56bb-ab2e-ea22c196b176', 'd345ae81-1e24-5d6c-ab9f-635ae6c8a40a', '6163d0cd-c637-5b79-a40f-7b62f7058ec5', '83f38999-e274-55f2-88a8-33a000bfc4e1', '0fdc461c-2544-5a3b-a0da-4283842a25b0', 'fee22fcf-703c-531f-bee0-c61616fb754c', '9cfb494a-7d8f-5988-85dd-b74bb2d5f1b9', 'e3f42d16-d5df-59e1-a231-b54fc5318bbe', 'c9499a01-8e4f-5fe5-9b42-3a48745c2d70', '5c6d1e35-b6f2-5c19-969b-a69daf813277', '3a5682c0-2bd4-56af-b59a-6d948ff36d1a']

    LOCAL_DATA_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\df_sample\df_sample.parquet"

    if 'main_df' not in st.session_state:
        with st.spinner("Loading main dataset from local file..."):
            st.session_state.main_df = load_data_from_local_file(LOCAL_DATA_PATH)

    df = st.session_state.main_df
    if df is None:
        st.stop()

    if 'trip_index' not in st.session_state:
        st.session_state.trip_index = 0
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = analysis_mode

    if st.session_state.current_mode != analysis_mode:
        st.session_state.current_mode = analysis_mode
        st.session_state.trip_index = 0

    col1, col2 = st.sidebar.columns(2)
    if col1.button("⬅️ Previous Trip"):
        st.session_state.trip_index -= 1
    if col2.button("Next Trip ➡️"):
        st.session_state.trip_index += 1

    if analysis_mode == "At Sea GPS Analysis":
        st.header("Analysis of Trips with 'At Sea' GPS Points")
        st.sidebar.info("Blue dots are the raw GPS trace. Red markers are points flagged as 'outside' the land/bridge area. The green line is the OSRM map-matched route.")
        
        trip_ids = at_sea_trip_ids
        st.session_state.trip_index = st.session_state.trip_index % len(trip_ids)
        selected_trip_id = trip_ids[st.session_state.trip_index]
        
        st.subheader(f"Displaying Trip: `{selected_trip_id}` ({st.session_state.trip_index + 1}/{len(trip_ids)})")
        
        trip_df = df[df[TRIP_ID_COL] == selected_trip_id].copy()
        trip_df.sort_values(TIMESTAMP_COL, inplace=True)
        
        if trip_df.empty:
            st.error("No data for this trip.")
            st.stop()
            
        coords, timestamps = [], []
        for _, row in trip_df.iterrows():
            try:
                lat, lng = h3.cell_to_latlng(format(int(row[LOCATION_COL]), 'x'))
                coords.append((lat, lng))
                timestamps.append(int(pd.Timestamp(row[TIMESTAMP_COL]).timestamp()))
            except (ValueError, TypeError, KeyError):
                continue
        
        if not coords:
            st.error("No valid coordinates for this trip.")
            st.stop()

        m = folium.Map(location=[np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])], zoom_start=13)
        for lat, lng in coords:
            folium.CircleMarker(location=[lat, lng], radius=3, color='dodgerblue', fill=True).add_to(m)
        
        outside_points_data = {'trip_id': at_sea_trip_ids, 'latitude': [47.278671, 47.275275, 45.855204, 43.304867], 'longitude': [-2.165381, -2.163538, -1.183532, 5.337614]}
        flagged_points_df = pd.DataFrame(outside_points_data)
        for _, row in flagged_points_df[flagged_points_df['trip_id'] == selected_trip_id].iterrows():
            folium.Marker(location=[row['latitude'], row['longitude']], icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

        matched_route = get_osrm_match_route(coords, timestamps)
        if matched_route:
            route_coords = [[c[1], c[0]] for c in matched_route.get("geometry", {}).get("coordinates", [])]
            folium.PolyLine(route_coords, color="limegreen", weight=5, opacity=0.8).add_to(m)

        m.fit_bounds([[min(c[0] for c in coords), min(c[1] for c in coords)], [max(c[0] for c in coords), max(c[1] for c in coords)]])
        folium_static(m, width=800, height=600)

    elif analysis_mode == "Speed Outlier Analysis":
        st.header("Analysis of Trips with Speed Outliers")
        st.sidebar.info("Blue segments are normal. Red segments are connected to a row containing a speed error (> 250 kph). Circles mark the GPS points.")

        trip_ids = speed_outlier_trip_ids
        st.session_state.trip_index = st.session_state.trip_index % len(trip_ids)
        selected_trip_id = trip_ids[st.session_state.trip_index]

        st.subheader(f"Displaying Trip: `{selected_trip_id}` ({st.session_state.trip_index + 1}/{len(trip_ids)})")

        trip_df = df[df[TRIP_ID_COL] == selected_trip_id].copy()
        trip_df.sort_values(TIMESTAMP_COL, inplace=True)

        if trip_df.empty:
            st.error("No data found for this trip ID in the main dataset.")
            st.stop()

        points_data = []
        for index, row in trip_df.iterrows():
            try:
                lat, lng = h3.cell_to_latlng(format(int(row[LOCATION_COL]), 'x'))
                error_present = has_speed_error(row, speed_col=SPEED_ARRAY_COL, min_val=0, max_val=250)
                points_data.append({'lat': lat, 'lng': lng, 'has_error': error_present, 'idx': len(points_data)})
            except (ValueError, TypeError, KeyError):
                continue
        
        if not points_data:
            st.error("Could not extract valid coordinates for this trip.")
            st.stop()

        with st.expander("Show Raw Point Data and Error Flags"):
            st.dataframe(pd.DataFrame(points_data))

        avg_lat = np.mean([p['lat'] for p in points_data])
        avg_lng = np.mean([p['lng'] for p in points_data])
        
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13, tiles="OpenStreetMap")

        for i in range(len(points_data) - 1):
            start_point = points_data[i]
            end_point = points_data[i+1]
            
            is_error_edge = start_point['has_error'] or end_point['has_error']
            
            line_color = "crimson" if is_error_edge else "royalblue"
            line_weight = 7 if is_error_edge else 4
            
            folium.PolyLine(
                locations=[(start_point['lat'], start_point['lng']), (end_point['lat'], end_point['lng'])],
                color=line_color,
                weight=line_weight,
                opacity=0.9
            ).add_to(m)

        for p in points_data:
            folium.CircleMarker(
                location=[p['lat'], p['lng']],
                radius=5,
                color="black",
                weight=1,
                fill=True,
                fill_color="green",
                fill_opacity=1.0,
                popup=f"Point Index: {p['idx']}"
            ).add_to(m)

        m.fit_bounds([[min(p['lat'] for p in points_data), min(p['lng'] for p in points_data)],
                      [max(p['lat'] for p in points_data), max(p['lng'] for p in points_data)]])
        
        folium_static(m, width=800, height=600)

if __name__ == "__main__":
    main()