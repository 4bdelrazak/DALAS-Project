import pandas as pd

era_df = pd.read_csv('ERA5_aggregated_full.csv')
cams_df = pd.read_csv('CAMS_aggregated.csv')

era_df['year'] = era_df['year'].astype(int)
cams_df['year'] = cams_df['year'].astype(int)

# Perform a LEFT JOIN starting with the longer dataset (ERA5)
# This keeps 1980-2023 and adds CAMS data where it exists (2003-2023)
final_panel = pd.merge(
    era_df, 
    cams_df, 
    on=['country_name', 'year', 'month'], 
    how='left'
)

final_panel = final_panel.sort_values(['country_name', 'year', 'month'])


output_path = 'Climate_Data_Final.csv'
final_panel.to_csv(output_path, index=False)



# merged = pd.merge(
#     cams,
#     era_combined,
#     on=["country_name", "year", "month"],
#     how="inner",         
#     suffixes=("_cams", "_era")
# )

