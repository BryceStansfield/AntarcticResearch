from ACTM_Measure_Scraper.src.Pipeline import scrape_and_enrich_measures
import pandas as pd
import country_meta_info

class RatificationSpeed():
    def __init__(self) -> None:
        scrape_and_enrich_measures("data/MeasureCorpus.csv", "data/MeasureCorpusEnriched.csv")

        measures = pd.read_csv("data/MeasureCorpusEnriched.csv")
        measures = measures[(measures["Meeting_Type"] == "ATCM")
                            & (measures["ATCM_Year"] >= 1995)
                            & (measures["ATCM_Year"] <= 2024)
                            & (measures["Type"] == "Measure")
                            & ~measures["Approvals"].str.contains("Fast Approval", na=False)
                            & (measures["Approvals"] != "")
                            & ~measures["Approvals"].isna()]

        # Delays are bucketed by the measure's ATCM year -- when the instrument was
        # tabled -- rather than by the year the country happened to ratify it. A
        # year's figure therefore reads as "how quickly was what we adopted that
        # year ratified", which keeps a slow ratification attributed to the measure
        # that caused it instead of smearing it across the years it took.
        # Country -> {atcm_year: [delay_years, ...]}.
        self.country_approval_times_by_year = country_meta_info.CaseInsensitiveDict()
        def add_approval(country, atcm_year, approval_delay_years):
            # The trailing " *" marks exactly the Consultative Parties as of that
            # resolution, so this guard defines the population rather than cleaning
            # the data: only Consultative Parties are meaningful to measure
            # Antarctic power for. Unstarred parties are dropped from the metric
            # entirely, at every year, by design -- do not widen this filter.
            if country[-1] == "*":
                country = country[:len(country)-2]

                if country not in self.country_approval_times_by_year:
                    self.country_approval_times_by_year[country] = {}
                by_year = self.country_approval_times_by_year[country]
                by_year.setdefault(atcm_year, []).append(approval_delay_years)

        # All measures of form "... (year)", "Not yet effective", or Effective dd/mm/YYYY.
        def extract_end_year(status):
            if status == "Not yet effective":
                return 2024
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
                    add_approval(pair[0], atcm_year, int(pair[1])-atcm_year)
                else:
                    if "Effective" in status:
                        add_approval(pair[0], atcm_year, int(status[len(status)-4:])-atcm_year)
                    else:
                        add_approval(pair[0], atcm_year, extract_end_year(row.Status)-atcm_year)

        # The headline figure averages every delay for the country, not the average
        # of its per-year averages, so it is unaffected by the yearly bucketing.
        self.country_approval_times = country_meta_info.CaseInsensitiveDict()
        for country, by_year in self.country_approval_times_by_year.items():
            delays = [d for year_delays in by_year.values() for d in year_delays]
            self.country_approval_times[country] = sum(delays)/len(delays)

    def country_dict(self) -> dict:
        return dict(self.country_approval_times)

    def figure_title(self) -> str:
        return "Ratification Delay"

    def save_full_figures(self, path: str):
        # value is the mean delay across the measures adopted in that ATCM year.
        # A country-year exists iff that country was listed with the trailing "*"
        # on some qualifying measure tabled that year, so a gap means "no measure
        # to ratify, or not a starred party" -- a coverage fact, not a behavioural
        # one. Failing to ratify does not produce a gap: extract_end_year censors
        # those at the withdrawal year (or 2024 for "Not yet effective"), so a
        # country that ratified nothing surfaces as a long delay, not a missing row.
        yearly_figures = sorted(
            ({"year": int(year), "country": country, "value": sum(delays)/len(delays)}
             for country, by_year in self.country_approval_times_by_year.items()
             for year, delays in by_year.items()),
            key=lambda r: (r["year"], r["country"]))
        pd.DataFrame(yearly_figures).to_csv(path)


if __name__ == "__main__":
    print(RatificationSpeed().country_dict())