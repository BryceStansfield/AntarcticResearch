import pandas as pd

df = pd.read_csv('data/MeasureCorpus_withAdoptionDates_andTypes.csv', low_memory=False)

print(df.info())
df['Meeting_Type'] = df['Title'].apply(lambda x: 
    'SATCM' if any(term in str(x) for term in ['SATCM', 'CCAMLR']) else 
    ('ATCM' if any(term in str(x) for term in ['ATCM', 'CEP', 'Antarctic Conference', 'ATIP']) else 
    ('CCAS' if 'CCAS' in str(x) else 'Unknown')))
print(df['Type'].isna().sum())
df.to_csv('data/MeasureCorpus_withMeetingType3.csv', index=False)