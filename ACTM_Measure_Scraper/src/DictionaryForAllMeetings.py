import pandas as pd

df = pd.read_csv('data/ATCM_IPWP_WithCats.csv')
print(df.info())
meeting_year_dict = pd.Series(df.Year.values, index=df.Meeting_Number).to_dict()
# Convert dictionary to dataframe
df_dict = pd.DataFrame(list(meeting_year_dict.items()), columns=['Meeting_Number', 'Year'])
# Save to CSV
df_dict.to_csv('data/meeting_year_dictionary.csv', index=False)