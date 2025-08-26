import pandas as pd
import requests
from pathlib import Path

# https://open-meteo.com/en/docs

def main():
    this_path = Path(__file__).parent.resolve()
    latitude = 52.63165629193071
    longitude = -1.1376761836683829
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        + f"latitude={latitude}"
        + f"&longitude={longitude}"
        + "&start_date=2010-02-01"
        + "&end_date=2012-08-01"
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
    with open(this_path / "weather.2010.02.01-2012.08.01.txt" , "w") as file:
        file.write(str(response))
    print("-------------DONE----------------")


if __name__ == "__main__":
    main()

# py -m coursera.IBM_ML_C3_Classification.project.get_weather.get_weather
