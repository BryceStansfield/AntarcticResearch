import re

from ACTM_Measure_Scraper.src.Pipeline import scrape_and_enrich_measures
import pandas as pd
import country_meta_info

from antarctic_ladder_metrics.constants import END_YEAR


# The year a measure stopped being ratifiable, for countries that never
# ratified it. Matched on shape rather than on substring presence, because a
# single status can carry several year-like fragments:
#   "Not yet effective"
#   "Effective 11/05/2016"
#   "Effective 19/12/2002 (Fast Approval)"        -- parens hold a label
#   "Effective 30/04/1962. No longer current:D 1 (2014)"
#   "Did not enter into effect. Withdrawn:M 3 (2012)"
#
# Module-level (not nested in RatificationSpeed.__init__) so it can be
# exercised directly in tests without running the whole scrape-and-enrich
# pipeline. Its only free names are `re` and `END_YEAR`, both imported above.
def extract_end_year(status):
    status = status.strip()
    # A measure that never took effect is right-censored at the edge of the
    # observation window, so its delay reads as "at least this long" rather
    # than as a completed ratification.
    if status == "Not yet effective":
        return END_YEAR
    # A trailing "(yyyy)" is the year the measure was withdrawn or ceased to
    # be current, and it takes precedence: it closes the ratification window
    # later than the effective date does. Requiring four digits is what keeps
    # "(Fast Approval)" from being read as a year -- slicing the parenthesis
    # unconditionally raises ValueError on those.
    withdrawn = re.search(r"\((\d{4})\)\s*$", status)
    if withdrawn:
        return int(withdrawn.group(1))
    # Otherwise fall back to the date it took effect: a country that had not
    # ratified by then had already missed its window.
    effective = re.search(r"Effective\s+\d{2}/\d{2}/(\d{4})", status)
    if effective:
        return int(effective.group(1))
    raise ValueError(f"Cannot extract year from status: {status}")


# Splits a measure's raw Approvals cell into (country, approval_year_or_None)
# pairs. A line starting with a digit is a year continuation of the previous
# country line; a country line with no following digit line pairs with None,
# meaning "never ratified". Module-level so it is testable without a corpus
# load -- see extract_end_year's docstring comment above for why that matters.
def parse_approval_pairs(approvals_text):
    approval_list = list(filter(lambda s: s != '',
                                 [s.strip() for s in approvals_text.split('\n')]))

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

    return country_approval_pairs


# Turns one measure row's Approvals cell into (country, delay_years) pairs,
# where delay_years is years-since-ATCM-adoption for a ratifying country, or
# a censored delay (extract_end_year(status) - atcm_year) for a country that
# never ratified. Deliberately does NOT do the " *" Consultative Party
# population filter -- that stays in RatificationSpeed.add_approval, which is
# also the only place that strips the suffix and accumulates the result.
def compute_row_delays(approvals_text, atcm_year, status):
    delays = []
    for country, approval_year in parse_approval_pairs(approvals_text):
        if approval_year is not None:
            delays.append((country, int(approval_year) - atcm_year))
        else:
            # No date against the country: it never ratified, so censor at the
            # year the measure stopped being ratifiable. extract_end_year owns
            # every status shape, including "Effective dd/mm/YYYY" -- handling
            # that one inline here as well only duplicated the parsing.
            delays.append((country, extract_end_year(status) - atcm_year))
    return delays


class RatificationSpeed():
    # This figure is a mean delay, not a count, so a country missing from the result
    # is not a zero-delay ratifier -- it was never a Consultative Party across the
    # window (Czechia only became one in 2014, after the last qualifying measure),
    # and so has no ratification behaviour to measure. Defaulting it to 0 would rank
    # it the fastest of all 28; NaN keeps it out of the comparison entirely.
    MISSING_VALUE = float("nan")

    def __init__(self) -> None:
        scrape_and_enrich_measures("data/MeasureCorpus.csv", "data/MeasureCorpusEnriched.csv")

        measures = pd.read_csv("data/MeasureCorpusEnriched.csv")
        # The lower bound is the 1995 ATCM reform, not the ladder's START_YEAR: it is
        # the point from which "Measure" means the modern ratifiable instrument, so
        # it is a definitional bound rather than a window choice.
        measures = measures[(measures["Meeting_Type"] == "ATCM")
                            & (measures["ATCM_Year"] >= 1995)
                            & (measures["ATCM_Year"] <= END_YEAR)
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

        for row in measures.itertuples():
            atcm_year = row.ATCM_Year
            for country, approval_delay_years in compute_row_delays(row.Approvals, atcm_year, row.Status):
                add_approval(country, atcm_year, approval_delay_years)

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
        # those at the withdrawal year (or END_YEAR for "Not yet effective"), so a
        # country that ratified nothing surfaces as a long delay, not a missing row.
        yearly_figures = sorted(
            ({"year": int(year), "country": country, "value": sum(delays)/len(delays)}
             for country, by_year in self.country_approval_times_by_year.items()
             for year, delays in by_year.items()),
            key=lambda r: (r["year"], r["country"]))
        pd.DataFrame(yearly_figures).to_csv(path)


if __name__ == "__main__":
    print(RatificationSpeed().country_dict())
