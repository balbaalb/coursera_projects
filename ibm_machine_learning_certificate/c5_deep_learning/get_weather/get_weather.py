import pandas as pd
import requests
from pathlib import Path
import json
import numpy as np

# https://open-meteo.com/en/docs

cities = {
    "austin": {"name": "Austin", "latitude": 30.2672, "longitude": 97.7431},
    "calgary": {"name": "Calgary", "latitude": 51.1217, "longitude": 114.0081},
    "los_angeles": {
        "name": "LosAngeles",
        "latitude": 33.9422,
        "longitude": 118.4036,
    },
    "montreal": {"name": "Montreal", "latitude": 45.4578, "longitude": 73.7492},
    "toronto": {"name": "Toronto", "latitude": 43.6532, "longitude": 79.3832},
    "lincoln": {"name": "Lincoln_NB", "latitude": 40.8137, "longitude": 96.7026},
    "lisbon": {"name": "Lisbon", "latitude": 38.7223, "longitude": 9.1393},
    "paris": {"name": "Paris", "latitude": 48.8575, "longitude": 2.3514},
}


def get_weather(city: dict, year: int) -> None:
    this_path = Path(__file__).parent.resolve()
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        + f"latitude={city['latitude']}"
        + f"&longitude={city['longitude']}"
        + f"&start_date={year}-01-01"
        + f"&end_date={year}-12-31"
        + "&hourly=temperature_2m"
        + ",relative_humidity_2m"
        + ",precipitation"
        + ",rain"
        + ",showers"
        + ",snowfall"
        + ",snow_depth"
        + ",dew_point_2m"
        + ",wind_speed_10m"
        + ",wind_direction_10m"
        + ",wind_gusts_10m"
        + ",cloud_cover"
        + ",visibility"
        + ",surface_pressure"
        + ",soil_moisture_0_to_1cm"
    )

    response = requests.get(url=url).json()
    with open(this_path / f"weather_data/{city['name']}.{year}.txt", "w") as file:
        file.write(str(response))
    print("-------------DONE----------------")


def add_weather_data(city: dict, year: int, df: pd.DataFrame | None) -> None:
    """
    Adds the weather data from a weather API
    """
    this_path = Path(__file__).parent.resolve()
    with open(this_path / f"weather_data/{city['name']}.{year}.txt", "r") as file:
        data_json = json.load(file)
        df_weather = pd.DataFrame()
        timestamp_raw = data_json["hourly"]["time"]
        df_weather["timestamp"] = pd.to_datetime(
            timestamp_raw, format="mixed", dayfirst=True
        )
        fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "snow_depth",
            "dew_point_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "surface_pressure",
        ]
        for field in fields:
            df_weather[field] = data_json["hourly"][field]
        print(f"df_weather.shape = {df_weather.shape}")
        if df is None:
            df = df_weather
        else:
            df = pd.concat([df, df_weather], axis=0)
        df.to_csv(this_path / f"weather_data/{city['name']}_weather.csv", index=False)
        return df


def main():
    get_weather(city=cities["paris"], year=2023)
    for city in ["austin", "calgary", "lisbon", "los_angeles", "paris"]:
        df = add_weather_data(city=cities[city], year=2023, df=None)
        df = add_weather_data(city=cities[city], year=2024, df=df)


def main_2025_08_30():
    city = "austin"
    year = 1989
    get_weather(city=cities[city], year=year)


def main_2025_08_30_pt2():
    city = "austin"
    df = None
    years = np.arange(1992, 2025)
    for year in years:
        df = add_weather_data(city=cities[city], year=year, df=df)


if __name__ == "__main__":
    # main()
    # main_2025_08_30()
    main_2025_08_30_pt2()
    pass

# py -m coursera.IBM_ML_C5_DeepLearning_and_RL.project.get_weather.get_weather
