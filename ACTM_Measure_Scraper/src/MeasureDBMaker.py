import pandas as pd

df = pd.read_csv('data/MeasureCorpus_Latest.csv')

# df = df[df['Document_Number'] <= 750]

df.drop(columns=['Year','Content'], inplace=True)

df2 = pd.read_csv('data/MeasureCorpus_Latest_2.csv')

df2.drop(columns=['Content'], inplace=True)

df = pd.concat([df, df2], ignore_index=True)

df['Adoption_Year'] = pd.to_datetime(
    df['Status'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0], 
    format='%d/%m/%Y', 
    errors='coerce'
).dt.year


df['ATCM_Year'] = df['Title'].str.extract(r'\((\d{4})\)|ATCM .+?\(.*?,\s*(\d{4})\)|.*?\(.*?,\s*(\d{4})\)').fillna('').replace('', None).bfill(axis=1).iloc[:, 0]

def roman_to_int(roman):
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
    if not roman or not isinstance(roman, str):
        return None
    result = 0
    for i in range(len(roman)):
        if i > 0 and roman_values[roman[i]] > roman_values[roman[i-1]]:
            result += roman_values[roman[i]] - 2 * roman_values[roman[i-1]]
        else:
            result += roman_values[roman[i]]
    return result

df['ATCM_Number'] = df['Title'].str.extract(r'ATCM\s+([IVXL]+)').iloc[:, 0].apply(roman_to_int)
meeting_dict = pd.read_csv('data/meeting_year_dictionary.csv')
meeting_dict_map = dict(zip(meeting_dict['Meeting_Number'], meeting_dict['Year']))

df['ATCM_Year'] = df.apply(lambda x: meeting_dict_map.get(x['ATCM_Number']) if pd.isna(x['ATCM_Year']) and x['ATCM_Number'] in meeting_dict_map else x['ATCM_Year'], axis=1)
df['Type'] = df['Title'].str.extract(r'^(Resolution|Decision|Measure|Recommendation)').iloc[:, 0]
df.to_csv('data/MeasureCorpus_withAdoptionDates_andTypes.csv', index=False)
