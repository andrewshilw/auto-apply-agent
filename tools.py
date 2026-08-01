"""Weather lookup tool backed by the free Open-Meteo API (no key required)."""

import requests
from langchain_core.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


@tool
def get_weather(city: str) -> str:
    """Fetch the current real-time weather for a given city name."""
    geo = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10).json()
    results = geo.get("results")
    if not results:
        return f"Could not find a location named '{city}'."

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    label = ", ".join(filter(None, [place.get("name"), place.get("admin1"), place.get("country")]))

    forecast = requests.get(
        FORECAST_URL,
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()
    current = forecast.get("current_weather")
    if not current:
        return f"Weather data unavailable for {label}."

    condition = WEATHER_CODES.get(current["weathercode"], "unknown conditions")
    return (
        f"Current weather in {label}: {current['temperature']}°C, "
        f"wind {current['windspeed']} km/h, {condition}."
    )
