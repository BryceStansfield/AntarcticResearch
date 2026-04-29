import pandas as pd

def construct_meeting_year_pairs(ipwp_input_path = 'data/ATCM_IPWP_WithCats.csv', output_path = 'data/meeting_year_dictionary.csv'):
    df = pd.read_csv(ipwp_input_path)
    
    meeting_year_dict = pd.Series(df.Year.values, index=df.Meeting_Number).to_dict()

    # Convert dictionary to dataframe
    df_dict = pd.DataFrame(list(meeting_year_dict.items()), columns=['Meeting_Number', 'Year'])

    # Save to CSV
    df_dict.to_csv(output_path, index=False)

def construct_meeting_year_pairs_if_not_exists(ipwp_input_path = 'data/ATCM_IPWP_WithCats.csv', output_path = 'data/meeting_year_dictionary.csv'):
    try:
        df = pd.read_csv(output_path)
        print(f"Meeting-year pairs already exist in {output_path}. No new file created.")
    except FileNotFoundError:
        construct_meeting_year_pairs(ipwp_input_path, output_path)
        print(f"Meeting-year pairs constructed and saved to {output_path}.")

if __name__ == "__main__":
    construct_meeting_year_pairs_if_not_exists()