import pandas as pd

import country_meta_info
from antarctic_ladder_metrics.infrastructure_figures import FacilityFigures, VesselCrewFigures
from antarctic_ladder_metrics.final_report_metrics import FinalReportMentionFigures, FinalReportInterventionFigures
from antarctic_ladder_metrics.scar_leadership_figures import ScarLeadershipFigures
from antarctic_ladder_metrics.scopus_figures import ScopusFigures
from antarctic_ladder_metrics.ratification_speed import RatificationSpeed
from antarctic_ladder_metrics.working_paper_metrics import WorkingPaperAuthorship, WPCollaborationGraphCentrality
from antarctic_ladder_metrics.topic_introduction import TopicIntroduction, TopicDiversity
from antarctic_ladder_metrics.measure_wp_introduction import MeasureWPIntroducers
from antarctic_ladder_metrics.information_paper_metrics import InformationPaperAuthorship
from downloaders.download_all import download_and_extract_all

import pathlib

def aggregate_all_figures():
    download_and_extract_all()
    countries = ["Argentina", "Australia", "Belgium", "Brazil", "Bulgaria", "Chile", "China", "Czechia", "Ecuador", "Finland", "France", "Germany", "India", "Italy", "Japan", "Republic of Korea", "Netherlands", "New Zealand", "Norway", "Peru", "Poland", "Russia", "South Africa", "Spain", "Sweden", "United Kingdom", "United States", "Uruguay"]
    collaboration_centrality = WPCollaborationGraphCentrality()
    figures = [FacilityFigures(), VesselCrewFigures(), FinalReportMentionFigures(), FinalReportInterventionFigures(), ScarLeadershipFigures(), ScopusFigures(), RatificationSpeed(), WorkingPaperAuthorship(), collaboration_centrality, TopicIntroduction(), TopicDiversity(), MeasureWPIntroducers(), InformationPaperAuthorship()]
    figure_dicts = [figure.country_dict() for figure in figures]

    results = pd.DataFrame(columns=["Country"] + [figure.figure_title() for figure in figures])
    for country in countries:
        row = {"Country": country}
        for figure, cdict in zip(figures, figure_dicts):
            # Figures default to 0 for a country they never saw; those reporting an
            # average override MISSING_VALUE, since 0 is a real score there.
            row[figure.figure_title()] = country_meta_info.get_country_value_from_dict(
                cdict, country, getattr(figure, "MISSING_VALUE", 0))
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    print("\n--- Coverage Check ---")
    for figure, cdict in zip(figures, figure_dicts):
        unused, not_found = country_meta_info.check_dict_coverage(cdict, countries)
        print(f"\n{figure.figure_title()}:")
        print(f"  Unused dict keys:    {unused}")
        print(f"  Countries not found: {not_found}")

    results.to_csv("data/ladder_results.csv")

    full_figure_dir = pathlib.Path("data") / "full_figures"
    full_figure_dir.mkdir(parents=True, exist_ok=True)
    for f in figures:
        # FacilityFigures and VesselCrewFigures have no yearly breakdown to save, so they
        # legitimately define no save_full_figures. Skip exactly those; a figure that has the
        # method and then fails must raise, rather than silently leaving the previous run's
        # CSV on disk next to a freshly-written ladder_results.csv.
        if not hasattr(f, "save_full_figures"):
            continue
        f.save_full_figures(full_figure_dir / (f.figure_title() + ".csv"))

    collaboration_centrality.save_collaboration_graphs(pathlib.Path("data") / "collaboration_graphs")
    return results

from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    print(aggregate_all_figures())