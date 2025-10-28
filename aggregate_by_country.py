import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask


# mask = regionmask.Regions(europe["geometry"], names=europe["NAME_EN"], abbrevs=europe["ISO_A3"])
# # mask_3D = mask.mask_3D(ds, lat_name="latitude", lon_name="longitude")
# # mask_3D = mask.mask_3D(ds, lat="latitude", lon_or_obj="longitude")
# ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
# mask_3D = mask.mask_3D(ds["lon"], ds["lat"])



ds = xr.open_dataset("ERA5_full_2000_2024.nc")

# Rename coords for regionmask compatibility
ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

world = gpd.read_file("countries/ne_110m_admin_0_countries.shp")
extra_countries = ["Georgia", "Moldova", "Albania", "Serbia", "Kosovo", "Bosnia and Herzegovina", "Montenegro"]
europe = world[(world["REGION_UN"] == "Europe") | (world["NAME_EN"].isin(extra_countries))]

# Create region mask
mask = regionmask.Regions(europe["geometry"], names=europe["NAME_EN"], abbrevs=europe["ISO_A3"])
mask_2D = mask.mask(ds)


# Compute mean for each country (region)
results = []
for i, region_name in enumerate(mask.names):
    # select grid points within this region
    region_data = ds.where(mask_2D == i)
    
    # take spatial mean (lat/lon) for all variables
    mean_ds = region_data.mean(dim=["lat", "lon"], skipna=True)
    
    # convert to dataframe and add region info
    df_region = mean_ds.to_dataframe().reset_index()
    df_region["country"] = region_name
    results.append(df_region)

# Combine all country-level data
df_all = pd.concat(results, ignore_index=True)


df_all.to_csv("ERA5_country_monthly_means.csv", index=False)
# print("Saved country-level monthly averages to ERA5_country_monthly_means.csv")