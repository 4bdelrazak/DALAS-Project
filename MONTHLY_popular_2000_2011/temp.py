import xarray as xr

# Load both datasets
ds_avgad = xr.open_dataset("data_stream-moda_stepType-avgad.nc")
ds_avgua = xr.open_dataset("data_stream-moda_stepType-avgua.nc")


# for var in ds_avgua.data_vars:
#     print(f'- variable name: {var},\n- attributes: {ds_avgua[var].attrs}\n')

print(ds_avgad)
print(ds_avgua)


# ds = xr.merge([ds_avgad, ds_avgua])

# print(ds)

ds = ds_avgua

# # Convert a slice (e.g., first month) to a DataFrame
# df_sample = ds.isel(valid_time=0).to_dataframe().reset_index()

# print(df_sample.head())


# import matplotlib.pyplot as plt

# # ds['t2m'].isel(valid_time=0).plot(cmap='coolwarm')
# # plt.title("2m Temperature (Jan 2000)")


# paris = ds.sel(latitude=48.8, longitude=2.3, method='nearest')

# paris['t2m'].plot()
# plt.title("Monthly mean temperature near Paris (2000–2011)")
# plt.show()