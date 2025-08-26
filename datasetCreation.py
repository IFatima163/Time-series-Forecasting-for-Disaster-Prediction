import pandas as pd
import numpy as np
import glob
import os
import xarray as xr

data_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\RawDatasets"
save_path = r"C:\Users\bifq0\OneDrive\Desktop\VS Code\ModelTraining\IndividualDisasterDatasets"

#----------------------------------------------------------------------------------
# Merge all NOAA flood CSVs
flood_files = glob.glob(os.path.join(data_path,"StormEvents_details-ftp_v1.0_*.csv"))
flood_list = [pd.read_csv(f) for f in flood_files]
floods = pd.concat(flood_list, ignore_index=True)
print("Floods merged:", len(floods), "rows")

#----------------------------------------------------------------------------------
# Merge all NASA FIRMS wildfire CSVs
fire_files = glob.glob(os.path.join(data_path,"modis_*.csv"))
fire_list = [pd.read_csv(f) for f in fire_files]
fires = pd.concat(fire_list, ignore_index=True)
print("Wildfires merged:", len(fires), "rows")

#----------------------------------------------------------------------------------
# Open temperature NetCDF and convert decimal years to datetime
nc_file = os.path.join(data_path, "Global_TAVG_Gridded_1deg.nc")
ds = xr.open_dataset(nc_file)
time_years = ds["time"].values  # float years, e.g., 1850.041667

def decimal_year_to_datetime(years):
    dates = []
    for y in years:
        year = int(np.floor(y))
        rem = y - year
        day_of_year = int(rem * 365.25)  # approximate
        date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_of_year)
        dates.append(date)
    return np.array(dates)

dates = decimal_year_to_datetime(time_years)

# Assign proper datetime index to temperature
temp = ds["temperature"].assign_coords(time=("time", dates))

# Compute daily max/min across lat/lon
daily_max = temp.max(dim=["latitude", "longitude"])
daily_min = temp.min(dim=["latitude", "longitude"])

# Convert to DataFrame
temps_max = daily_max.to_dataframe().reset_index().rename(columns={"time": "date", "temperature": "tmax"})
temps_min = daily_min.to_dataframe().reset_index().rename(columns={"time": "date", "temperature": "tmin"})

# Merge max and min
temps = pd.merge(temps_max, temps_min, on="date")

# Keep only the date part
temps["date"] = pd.to_datetime(temps["date"]).dt.date

# Filter for 2000-01-01 to 2024-12-31
start_filter = pd.Timestamp("2000-01-01").date()
end_filter = pd.Timestamp("2024-12-31").date()
temps = temps[(temps["date"] >= start_filter) & (temps["date"] <= end_filter)]

# Compute heatwave/coldwave thresholds AFTER filtering
heat_thresh = temps["tmax"].quantile(0.9)
cold_thresh = temps["tmin"].quantile(0.1)
print("Heatwave threshold:", heat_thresh)
print("Coldwave threshold:", cold_thresh)

temps["Heatwave"] = (temps["tmax"] >= heat_thresh).astype(int)
temps["Coldwave"] = (temps["tmin"] <= cold_thresh).astype(int)
heatcold_daily = temps.groupby("date")[["Heatwave","Coldwave"]].max().reset_index()
heatcold_daily.to_csv(os.path.join(save_path, "heatcold_daily.csv"), index=False)
print("heatcold_daily.csv saved!")

# Save filtered temperature CSV
temps.to_csv(os.path.join(data_path, "temperature_daily.csv"), index=False)
print("Temperature CSV created with proper dates and heat/cold wave flags")

#----------------------------------------------------------------------------------
# Prepare floods
floods["BEGIN_DATE_TIME"] = pd.to_datetime(floods["BEGIN_DATE_TIME"], errors='coerce')
floods["date"] = floods["BEGIN_DATE_TIME"].dt.date
floods_daily = floods.groupby("date").size().reset_index(name="Flood_Count")
floods_daily.to_csv(os.path.join(save_path, "floods_daily.csv"), index=False)
print("floods_daily.csv saved!")

#----------------------------------------------------------------------------------
# Prepare droughts
drought = pd.read_csv(os.path.join(data_path, "dm_export_20000101_20241231.csv"))
drought["ValidStart"] = pd.to_datetime(drought["ValidStart"], errors='coerce')
drought["date"] = drought["ValidStart"].dt.date
drought["Drought_Index"] = drought[["D0","D1","D2","D3","D4"]].max(axis=1)

drought_daily = drought.groupby("date")["Drought_Index"].mean().reset_index()

# Resample to fill missing dates
drought_daily["date"] = pd.to_datetime(drought_daily["date"])
drought_daily = drought_daily.set_index("date").resample("D").ffill().reset_index()
drought_daily.to_csv(os.path.join(save_path, "drought_daily.csv"), index=False)
print("drought_daily.csv saved!")

#----------------------------------------------------------------------------------
# Prepare wildfires
fires["acq_date"] = pd.to_datetime(fires["acq_date"], errors='coerce').dt.date
fires_daily = fires.groupby("acq_date").size().reset_index(name="Wildfire_Count")
fires_daily.rename(columns={"acq_date": "date"}, inplace=True)
fires_daily.to_csv(os.path.join(save_path, "fires_daily.csv"), index=False)
print("fires_daily.csv saved!")