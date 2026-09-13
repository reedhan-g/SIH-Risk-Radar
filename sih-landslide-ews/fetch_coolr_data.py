import requests
import json

url = "https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Events_Points/FeatureServer/0/query"
params = {
    "where": "country_name='India'",
    "outFields": "*",
    "f": "json"
}

response = requests.get(url, params=params, timeout=20)
print("Status code:", response.status_code)

result = response.json()
features = result.get("features", [])
print(f"Total India landslide events found: {len(features)}")

# Save to a file so we can inspect it
with open("coolr_india_events.json", "w") as f:
    json.dump(features, f, indent=2)

# Preview first 5
for feat in features[:5]:
    a = feat["attributes"]
    print(a.get("event_date"), "|", a.get("latitude"), a.get("longitude"), "|", a.get("event_title"))