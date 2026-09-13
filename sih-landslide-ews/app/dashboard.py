import streamlit as st
import pandas as pd
import requests
import datetime
import time

st.set_page_config(page_title="RiskRadar", layout="centered")
st.title("🏔️ RiskRadar — NER Landslide Early Warning System")
st.write("Real-time risk prediction using rainfall, terrain, seismic, and vegetation data")

st.divider()
st.header("🔍 Manual Risk Check")

col1, col2 = st.columns(2)
with col1:
    rainfall_mm = st.slider("Daily Rainfall (mm)", 0, 400, 100)
    rainfall_intensity = st.slider("Rainfall Intensity (mm/hr)", 0, 80, 15)
    rainfall_3day = st.slider("3-Day Cumulative Rainfall (mm)", 0, 900, 300)
    rainfall_15day = st.slider("15-Day Cumulative Rainfall (mm)", 0, 600, 200)
    temperature = st.slider("Temperature (°C)", 15, 40, 25)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 80, 15)
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 50)

with col2:
    groundwater = st.slider("Groundwater / Root-zone Moisture (%)", 0, 100, 50)
    slope_angle = st.slider("Slope Angle (degrees)", 0, 60, 25)
    elevation_m = st.slider("Elevation (m)", 50, 1900, 500)
    distance_to_river = st.slider("Distance to River (m)", 0, 5000, 1000)
    seismic_score = st.slider("Seismic Activity Score (0-10)", 0.0, 10.0, 2.0)
    ndvi = st.slider("Vegetation Cover (NDVI, 0=bare, 1=dense)", 0.0, 1.0, 0.5)
    soil_weakness = st.slider("Soil/Rock Weakness (0=strong, 1=very weak)", 0.0, 1.0, 0.5)

if st.button("🔍 Predict Risk", use_container_width=True):
    payload = {
        "rainfall_mm": rainfall_mm,
        "rainfall_intensity_mmhr": rainfall_intensity,
        "rainfall_3day_cumulative": rainfall_3day,
        "rainfall_15day_cumulative": rainfall_15day,
        "temperature_c": temperature,
        "wind_speed_kmh": wind_speed,
        "soil_moisture": soil_moisture,
        "groundwater_level": groundwater,
        "slope_angle": slope_angle,
        "elevation_m": elevation_m,
        "distance_to_river_m": distance_to_river,
        "seismic_activity_score": seismic_score,
        "vegetation_ndvi": ndvi,
        "soil_weakness_score": soil_weakness
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        risk_level = result["risk_level"]
        confidence = result["confidence_percent"]
        if risk_level == "HIGH":
            st.error(f"⚠️ HIGH RISK — Confidence: {confidence}%")
        else:
            st.success(f"✅ LOW RISK — Confidence: {confidence}%")
        st.progress(confidence / 100)
    except requests.exceptions.ConnectionError:
        st.error("Backend not reachable. Make sure FastAPI server is running on port 8000.")

# ---------------------------------------------------------------
st.divider()
st.header("📊 Real Event Backtest — Tupul, Manipur (29-30 June 2022)")
st.write("Real historical rainfall, temperature, wind, groundwater, and seismic data for the Tupul landslide site (61 deaths). Slope/soil/vegetation values are from published landslide studies.")

if st.button("▶️ Run Real Backtest", use_container_width=True):
    site_lat, site_lon = 24.804, 93.674
    event_date = datetime.date(2022, 6, 30)
    lookback_days = 60
    start_date = event_date - datetime.timedelta(days=lookback_days)

    # Documented site characteristics from published landslide studies
    documented_slope = 42.0
    documented_elevation = 1063.0
    documented_river_distance = 80.0
    documented_ndvi = 0.30       # low — site was deforested/cut for construction
    documented_soil_weakness = 0.85  # high — fractured shale/mudstone/sandstone, faulted zone

    with st.spinner("Fetching real historical daily weather (rainfall, temp, wind)..."):
        daily_res = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": site_lat, "longitude": site_lon,
                "start_date": start_date.isoformat(),
                "end_date": event_date.isoformat(),
                "daily": "precipitation_sum,temperature_2m_max,windspeed_10m_max",
                "timezone": "Asia/Kolkata"
            }
        ).json()

    with st.spinner("Fetching hourly rainfall for intensity..."):
        hourly_res = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": site_lat, "longitude": site_lon,
                "start_date": start_date.isoformat(),
                "end_date": event_date.isoformat(),
                "hourly": "precipitation",
                "timezone": "Asia/Kolkata"
            }
        ).json()

    with st.spinner("Fetching real soil moisture & groundwater proxy (NASA POWER)..."):
        soil_res = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params={
                "parameters": "GWETTOP,GWETROOT",
                "community": "AG",
                "longitude": site_lon, "latitude": site_lat,
                "start": start_date.strftime("%Y%m%d"),
                "end": event_date.strftime("%Y%m%d"),
                "format": "JSON"
            }
        ).json()

    with st.spinner("Fetching real seismic activity (USGS)..."):
        usgs_res = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start_date.isoformat(), "endtime": event_date.isoformat(),
                "latitude": site_lat, "longitude": site_lon,
                "maxradiuskm": 200, "minmagnitude": 2.0
            }
        ).json()

    dates = daily_res["daily"]["time"]
    rainfall_series = pd.Series([v if v is not None else 0 for v in daily_res["daily"]["precipitation_sum"]], index=dates)
    temp_series = pd.Series([v if v is not None else 25 for v in daily_res["daily"]["temperature_2m_max"]], index=dates)
    wind_series = pd.Series([v if v is not None else 5 for v in daily_res["daily"]["windspeed_10m_max"]], index=dates)
    cum3_series = rainfall_series.rolling(3, min_periods=1).sum()
    cum15_series = rainfall_series.rolling(15, min_periods=1).sum()

    # Max hourly intensity per day
    hourly_times = hourly_res["hourly"]["time"]
    hourly_precip = [v if v is not None else 0 for v in hourly_res["hourly"]["precipitation"]]
    intensity_by_date = {}
    for t, p in zip(hourly_times, hourly_precip):
        d = t[:10]
        intensity_by_date[d] = max(intensity_by_date.get(d, 0), p)

    gwet_top = soil_res.get("properties", {}).get("parameter", {}).get("GWETTOP", {})
    gwet_root = soil_res.get("properties", {}).get("parameter", {}).get("GWETROOT", {})
    soil_by_date = {f"{k[:4]}-{k[4:6]}-{k[6:8]}": max(0, v) * 100 if v and v > -900 else 40 for k, v in gwet_top.items()}
    ground_by_date = {f"{k[:4]}-{k[4:6]}-{k[6:8]}": max(0, v) * 100 if v and v > -900 else 40 for k, v in gwet_root.items()}

    quake_counts = {}
    for feature in usgs_res.get("features", []):
        eq_date = datetime.datetime.utcfromtimestamp(feature["properties"]["time"] / 1000).date().isoformat()
        quake_counts[eq_date] = quake_counts.get(eq_date, 0) + (feature["properties"]["mag"] or 0)

    results = []
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()

    for i, d in enumerate(dates):
        payload = {
            "rainfall_mm": float(rainfall_series[d]),
            "rainfall_intensity_mmhr": float(intensity_by_date.get(d, 0)),
            "rainfall_3day_cumulative": float(cum3_series[d]),
            "rainfall_15day_cumulative": float(cum15_series[d]),
            "temperature_c": float(temp_series[d]),
            "wind_speed_kmh": float(wind_series[d]),
            "soil_moisture": float(soil_by_date.get(d, 40)),
            "groundwater_level": float(ground_by_date.get(d, 40)),
            "slope_angle": documented_slope,
            "elevation_m": documented_elevation,
            "distance_to_river_m": documented_river_distance,
            "seismic_activity_score": float(min(10, quake_counts.get(d, 0.3))),
            "vegetation_ndvi": documented_ndvi,
            "soil_weakness_score": documented_soil_weakness
        }

        res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()

        results.append({
            "Date": d,
            "Rainfall (mm)": round(payload["rainfall_mm"], 1),
            "Intensity (mm/hr)": round(payload["rainfall_intensity_mmhr"], 1),
            "Soil Moisture (%)": round(payload["soil_moisture"], 1),
            "Groundwater (%)": round(payload["groundwater_level"], 1),
            "Risk Level": res["risk_level"],
            "Confidence (%)": res["confidence_percent"]
        })

        df_results = pd.DataFrame(results)
        chart_placeholder.line_chart(df_results.set_index("Date")[["Rainfall (mm)", "Soil Moisture (%)"]])
        progress_bar.progress((i + 1) / len(dates))

        if res["risk_level"] == "HIGH":
            alert_placeholder.error(f"🚨 {d}: HIGH RISK — Confidence {res['confidence_percent']}%")
        elif res["risk_level"] == "MODERATE":
            alert_placeholder.warning(f"⚠️ {d}: MODERATE RISK — Confidence {res['confidence_percent']}%")
        else:
            alert_placeholder.info(f"{d}: Low risk — Confidence {res['confidence_percent']}%")
        time.sleep(0.08)

    st.divider()
    st.subheader("Full Real-Data Log")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    first_alert = next((r["Date"] for r in results if r["Risk Level"] == "HIGH"), None)
    st.divider()
    if first_alert:
        days_early = (event_date - datetime.date.fromisoformat(first_alert)).days
        st.success(f"✅ System first flagged HIGH RISK on {first_alert} — {days_early} day(s) before the actual landslide on {event_date}.")
    else:
        st.warning("⚠️ Model did not flag HIGH risk before the event — useful finding for further tuning.")

# ---------------------------------------------------------------
st.divider()
st.header("📍 Field Report — Auto Risk Assessment")
st.write("Select your location — the system fetches real weather, groundwater, and seismic data automatically")

name = st.text_input("Name")
contact = st.text_input("Contact Number")

ner_locations = {
    "Assam": ["Kamrup (Guwahati)", "Dima Hasao (Haflong)", "Karbi Anglong", "Nagaon", "Sonitpur", "Jorhat", "Dibrugarh", "Cachar (Silchar)"],
    "Meghalaya": ["East Khasi Hills (Shillong)", "West Khasi Hills", "Ri Bhoi", "Jaintia Hills", "East Garo Hills"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Kolasib", "Serchhip"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Wokha", "Zunheboto"],
    "Manipur": ["Imphal East", "Imphal West", "Churachandpur", "Ukhrul", "Senapati", "Noney"],
    "Tripura": ["West Tripura (Agartala)", "Gomati", "Dhalai", "North Tripura"],
    "Arunachal Pradesh": ["Papum Pare (Itanagar)", "West Kameng", "East Siang", "Lohit"],
    "Sikkim": ["East Sikkim (Gangtok)", "West Sikkim", "North Sikkim", "South Sikkim"]
}

col_a, col_b = st.columns(2)
with col_a:
    state = st.selectbox("State", list(ner_locations.keys()))
with col_b:
    district = st.selectbox("District", ner_locations[state])

village = st.text_input("Village/Locality name (optional, improves accuracy)")

if st.button("🌍 Fetch Data & Predict Risk", use_container_width=True):
    search_query = f"{village}, {district}" if village else district
    try:
        geo_res = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": search_query, "count": 1}).json()
        if "results" not in geo_res:
            geo_res = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": district.split(" (")[0], "count": 1}).json()

        if "results" not in geo_res:
            st.error("Could not locate this place. Try a different district.")
        else:
            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]
            found_name = geo_res["results"][0]["name"]
            st.info(f"📌 Matched: {found_name} ({lat}, {lon})")

            today = datetime.date.today()
            past_start = today - datetime.timedelta(days=15)

            weather_res = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "precipitation_sum,temperature_2m_max,windspeed_10m_max",
                    "hourly": "precipitation",
                    "past_days": 15, "forecast_days": 1, "timezone": "auto"
                }
            ).json()

            daily_rain = [v if v is not None else 0 for v in weather_res["daily"]["precipitation_sum"]]
            today_rainfall = daily_rain[-1]
            cumulative_3day = sum(daily_rain[-4:-1])
            cumulative_15day = sum(daily_rain[:-1])
            today_temp = weather_res["daily"]["temperature_2m_max"][-1] or 25
            today_wind = weather_res["daily"]["windspeed_10m_max"][-1] or 10

            hourly_precip_today = [v if v is not None else 0 for t, v in zip(weather_res["hourly"]["time"], weather_res["hourly"]["precipitation"]) if t[:10] == weather_res["daily"]["time"][-1]]
            intensity_today = max(hourly_precip_today) if hourly_precip_today else 0

            elev_res = requests.get("https://api.open-elevation.com/api/v1/lookup", params={"locations": f"{lat},{lon}"}).json()
            elevation = elev_res["results"][0]["elevation"]

            power_res = requests.get(
                "https://power.larc.nasa.gov/api/temporal/daily/point",
                params={
                    "parameters": "GWETTOP,GWETROOT", "community": "AG",
                    "longitude": lon, "latitude": lat,
                    "start": past_start.strftime("%Y%m%d"), "end": today.strftime("%Y%m%d"),
                    "format": "JSON"
                }
            ).json()
            gwet_top_vals = list(power_res.get("properties", {}).get("parameter", {}).get("GWETTOP", {}).values())
            gwet_root_vals = list(power_res.get("properties", {}).get("parameter", {}).get("GWETROOT", {}).values())
            soil_moisture_auto = max(0, gwet_top_vals[-1]) * 100 if gwet_top_vals else 40
            groundwater_auto = max(0, gwet_root_vals[-1]) * 100 if gwet_root_vals else 40

            usgs_res = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params={
                    "format": "geojson",
                    "starttime": past_start.isoformat(), "endtime": today.isoformat(),
                    "latitude": lat, "longitude": lon, "maxradiuskm": 200, "minmagnitude": 2.0
                }
            ).json()
            quakes = usgs_res.get("features", [])
            seismic_auto = min(10, sum([q["properties"]["mag"] or 0 for q in quakes]) / 2) if quakes else 0.5

            st.write("**Auto-fetched real data:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rainfall Today", f"{today_rainfall} mm")
            c2.metric("15-Day Cumulative", f"{cumulative_15day:.0f} mm")
            c3.metric("Elevation", f"{elevation} m")
            c4.metric("Seismic (30d)", f"{len(quakes)} events")

            st.write("**Estimated (adjust if you know local terrain/land-use):**")
            e1, e2 = st.columns(2)
            with e1:
                slope_est = st.slider("Estimated Slope Angle (°)", 0, 60, 30, key="auto_slope")
                river_dist_est = st.slider("Estimated Distance to River (m)", 0, 5000, 500, key="auto_river")
            with e2:
                ndvi_est = st.slider("Vegetation Cover (NDVI)", 0.0, 1.0, 0.5, key="auto_ndvi")
                weakness_est = st.slider("Soil/Rock Weakness", 0.0, 1.0, 0.5, key="auto_weak")

            payload = {
                "rainfall_mm": float(today_rainfall),
                "rainfall_intensity_mmhr": float(intensity_today),
                "rainfall_3day_cumulative": float(cumulative_3day),
                "rainfall_15day_cumulative": float(cumulative_15day),
                "temperature_c": float(today_temp),
                "wind_speed_kmh": float(today_wind),
                "soil_moisture": float(soil_moisture_auto),
                "groundwater_level": float(groundwater_auto),
                "slope_angle": float(slope_est),
                "elevation_m": float(elevation),
                "distance_to_river_m": float(river_dist_est),
                "seismic_activity_score": float(seismic_auto),
                "vegetation_ndvi": float(ndvi_est),
                "soil_weakness_score": float(weakness_est)
            }
            res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()

            st.divider()
            if res["risk_level"] == "HIGH":
                st.error(f"🚨 HIGH RISK for {name or 'user'} at {district}, {state} — Confidence: {res['confidence_percent']}%")
                st.write(f"⚠️ Alert would be sent to: {contact}")
            else:
                st.success(f"✅ LOW RISK for {name or 'user'} at {district}, {state} — Confidence: {res['confidence_percent']}%")
    except Exception as e:
        st.error(f"Could not fetch data: {e}")