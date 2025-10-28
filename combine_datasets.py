import xarray as xr

xr.set_options(use_flox=False, use_bottleneck=True)

# ds1 = xr.open_mfdataset(
#     "./MONTHLY_popular_2000_2011/*.nc",
#     combine='by_coords',
#     join='outer',
#     compat='override'
# )
# ds2 = xr.open_mfdataset(
#     "./MONTHLY_popular_2012_2024/*.nc",
#     combine='by_coords',
#     join='outer',
#     compat='override'
# )

# ds_combined = xr.concat([ds1, ds2], dim="valid_time")
# ds_combined.to_netcdf("ERA5_popular_2000_2024.nc")
# print(ds_combined)

ds_combined = xr.open_mfdataset("./ERA5_popular_2000_2024.nc")

ds_radiation = xr.open_mfdataset("./MONTHLY_radiation_2000_2024.nc", combine='by_coords')


ds_final = xr.merge([ds_combined, ds_radiation])

ds_final.to_netcdf("ERA5_full_2000_2024.nc")
print(ds_final)