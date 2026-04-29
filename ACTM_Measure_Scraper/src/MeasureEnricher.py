import pandas as pd

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

def enrich_measure_data(measures_df_path = 'data/MeasureCorpus.csv', meeting_year_dict_path = 'data/meeting_year_dictionary.csv', output_path = 'data/MeasureCorpusEnriched.csv'):
    df = pd.read_csv(measures_df_path)

    df['Adoption_Year'] = pd.to_datetime(
        df['Status'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0], 
        format='%d/%m/%Y',
        errors='coerce'
    ).dt.year


    df['ATCM_Year'] = df['Title'].str.extract(r'\((\d{4})\)|ATCM .+?\(.*?,\s*(\d{4})\)|.*?\(.*?,\s*(\d{4})\)').fillna('').replace('', None).bfill(axis=1).iloc[:, 0]

    df['ATCM_Number'] = df['Title'].str.extract(r'ATCM\s+([IVXL]+)').iloc[:, 0].apply(roman_to_int)
    meeting_dict = pd.read_csv(meeting_year_dict_path)
    meeting_dict_map = dict(zip(meeting_dict['Meeting_Number'], meeting_dict['Year']))
    meeting_to_year = lambda meeting_number: meeting_dict_map.get(meeting_number) if meeting_number in meeting_dict_map else meeting_number - 18 + 1994

    df['ATCM_Year'] = df.apply(lambda x: meeting_to_year(x['ATCM_Number']) if pd.isna(x['ATCM_Year']) and x['ATCM_Number'] in meeting_dict_map else x['ATCM_Year'], axis=1)
    df['Type'] = df['Title'].str.extract(r'^(Resolution|Decision|Measure|Recommendation)').iloc[:, 0]

    df['Meeting_Type'] = df['Title'].apply(lambda x: 
        'SATCM' if any(term in str(x) for term in ['SATCM', 'CCAMLR']) else 
        ('ATCM' if any(term in str(x) for term in ['ATCM', 'CEP', 'Antarctic Conference', 'ATIP']) else 
        ('CCAS' if 'CCAS' in str(x) else 'Unknown')))

    df.to_csv(output_path, index=False)

def enrich_if_not_exists(measures_df_path = 'data/MeasureCorpus.csv', meeting_year_dict_path = 'data/meeting_year_dictionary.csv', output_path = 'data/MeasureCorpusEnriched.csv'):
    try:
        pd.read_csv(output_path)
        print(f"{output_path} already exists. Skipping enrichment.")
    except FileNotFoundError:
        enrich_measure_data(measures_df_path, meeting_year_dict_path, output_path)

if __name__ == "__main__":
    enrich_if_not_exists()