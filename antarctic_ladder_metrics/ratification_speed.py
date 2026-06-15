import requests
from ACTM_Measure_Scraper.src.Pipeline import scrape_and_enrich_measures
import pandas as pd
import datetime
import country_meta_info

class RatificationSpeed():
    def __init__(self) -> None:
        scrape_and_enrich_measures("data/MeasureCorpus.csv", "data/MeasureCorpusEnriched.csv")

        measures = pd.read_csv("data/MeasureCorpusEnriched.csv")
        measures = measures[(measures["Meeting_Type"] == "ATCM")
                            & (measures["ATCM_Year"] >= 1995)
                            & (measures["ATCM_Year"] <= 2023)
                            & (measures["Type"] == "Measure")
                            & ~measures["Approvals"].str.contains("Fast Approval", na=False)
                            & (measures["Approvals"] != "")]
                            #& ~measures["Approvals"].isna()]

        # TODO: Try using exact ATCM and approval dates.
        # TODO: Choose how to deal with never going to be approved measures.
        self.country_approval_times = country_meta_info.CaseInsensitiveDict()
        def add_approval(country, approval_delay_years):
            if country[-1] == "*":
                country = country[:len(country)-2]

                if country in self.country_approval_times:
                    self.country_approval_times[country].append(approval_delay_years)
                else:
                    self.country_approval_times[country] = [approval_delay_years]

        # All measures of form "... (year)", "Not yet effective", or Effective dd/mm/YYYY.
        def extract_end_year(status):
            if status == "Not yet effective":
                return 2023
            if "Effective" in status:
                return int(status[-4:])
            if status.endswith(')'):
                return int(status[status.rfind('(')+1:-1])
            raise ValueError(f"Cannot extract year from status: {status}")

        for row in measures.itertuples():
            approval_list = list(filter(lambda s: s != '', [s.strip() for s in row.Approvals.split('\n')]))
            atcm_year = row.ATCM_Year
            status = row.Status

            country_approval_pairs = []

            last_country = ""
            for e in approval_list:
                if e[0] not in "0123456789":
                    if last_country != "":
                        country_approval_pairs.append((last_country, None))

                    last_country = e
                else:
                    country_approval_pairs.append((last_country, e[len(e)-4:]))
                    last_country = ""
            if last_country != "":
                country_approval_pairs.append((last_country, None))
            
            for pair in country_approval_pairs:
                if pair[1] != None:
                    add_approval(pair[0], int(pair[1])-atcm_year)
                else:
                    if "Effective" in status:
                        add_approval(pair[0], int(status[len(status)-4:])-atcm_year)
                    else:
                        add_approval(pair[0], extract_end_year(row.Status)-atcm_year)

        for country in self.country_approval_times:
            self.country_approval_times[country] = sum(self.country_approval_times[country])/len(self.country_approval_times[country])

    def country_dict(self) -> dict:
        return dict(self.country_approval_times)

    def figure_title(self) -> str:
        return "Ratification Delay"
    
if __name__ == "__main__":
    print(RatificationSpeed().country_dict())