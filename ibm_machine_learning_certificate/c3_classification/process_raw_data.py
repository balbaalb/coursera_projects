import pandas as pd
import numpy as np
from pathlib import Path
import json

THIS_PATH = Path(__file__).parent.resolve()
PROJ_DATA_PATH = THIS_PATH / "project_data"


def process_compressor_data() -> pd.DataFrame:
    """
    Read the compressor data and process it into a new pd.DataFrame with cleaned data.
    """
    df0 = pd.read_csv(PROJ_DATA_PATH / "DMU_Compressor_Power.csv")
    timestamp = pd.to_datetime(
        df0["Date and time"], format="mixed", dayfirst=True
    )  # 15/02/2010 20
    df1 = pd.DataFrame()
    df1["timestamp"] = timestamp
    df1["cooling_kw"] = df0["Cooling"]
    df1["heating_kw"] = df0["Heating"]
    df1.dropna(inplace=True)
    return df1


def add_temperature_to_data(df: pd.DataFrame) -> None:
    """
    Adds the temperature measurements to the input.
    """
    temp_df = pd.read_csv(PROJ_DATA_PATH / "DMU_Air_Temperatures.csv")
    temp_timestamp = pd.to_datetime(
        temp_df["Date & Time"], format="mixed", dayfirst=False
    )  # 2010-01-15 0:00
    xp = temp_timestamp.astype(np.int64) // 10**9
    fp = temp_df["Air Temperature_C"].values
    x = df["timestamp"].astype(np.int64) // 10**9
    f = np.interp(x=x, xp=xp, fp=fp)
    df["dmu_air_temperature_c"] = f


def add_weather_data(df: pd.DataFrame) -> None:
    """
    Adds the weather data from a weather API
    """
    with open(PROJ_DATA_PATH / "weather.2010.02.01-2012.08.01.json", "r") as file:
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
        df_weather.to_csv(
            PROJ_DATA_PATH / "weather.2010.02.01-2012.08.01.csv", index=False
        )
        xp = df_weather["timestamp"].astype(np.int64) // 10**9
        x = df["timestamp"].astype(np.int64) // 10**9
        for field in fields:
            df[field] = np.interp(x=x, xp=xp, fp=df_weather[field].values)


if __name__ == "__main__":
    df = process_compressor_data()
    add_temperature_to_data(df)
    add_weather_data(df)
    df.to_csv(PROJ_DATA_PATH / "data.csv", index=False)


# py -m coursera.IBM_ML_C3_Classification.project.process_raw_data
