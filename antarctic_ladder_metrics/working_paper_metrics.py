import pandas as pd
from antarctic_ladder_metrics.constants import *
import country_meta_info

import networkx

from utils import split_parties

class WorkingPaperAuthorship():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_type"] == "ATCM") & (wp_authorship_table["party_type"] == "wp")][["parties", "meeting_year", "paper_id"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= START_YEAR) & (wp_authorship_table["meeting_year"] <= END_YEAR)]
        wp_authorship_table = wp_authorship_table.drop_duplicates(subset="paper_id", keep="first")

        self.country_authorships_by_year = {}
        for i in range(START_YEAR, END_YEAR+1):
            authors = list(wp_authorship_table[wp_authorship_table["meeting_year"] == i]["parties"].map(split_parties))
            
            for pl in authors:
                for p in pl:
                    p = country_meta_info.normalize_country_name(p)
                    if (i, p) in self.country_authorships_by_year:
                        self.country_authorships_by_year[(i, p)] += 1
                    else:
                        self.country_authorships_by_year[(i, p)] = 1
        
        self.country_authorships = {}
        for k in self.country_authorships_by_year:
            if k[1] in self.country_authorships:
                self.country_authorships[k[1]] += self.country_authorships_by_year[k]
            else:
                self.country_authorships[k[1]] = self.country_authorships_by_year[k]
            
    def country_dict(self) -> dict:
        return dict(self.country_authorships)

    def figure_title(self) -> str:
        return "Working Paper Authorship"
    
    def save_full_figures(self, path: str):
        yearly_figures = [{"year": k[0], "country": k[1], "value": v} for k,v in self.country_authorships_by_year.items()]
        pd.DataFrame(yearly_figures).to_csv(path)

class WPCollaborationGraphCentrality():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_type"] == "ATCM") & (wp_authorship_table["party_type"] == "wp")][["parties", "meeting_year", "paper_id"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= START_YEAR) & (wp_authorship_table["meeting_year"] <= END_YEAR)]
        wp_authorship_table = wp_authorship_table.drop_duplicates(subset="paper_id", keep="first")
        
        # Author sets are kept with their meeting year so the graph can be rebuilt
        # over any window. Centrality is not additive, so a window's value has to be
        # recomputed from that window's graph rather than derived from the overall one.
        self._author_sets_by_year = [(row.meeting_year, split_parties(row.parties))
                                     for row in wp_authorship_table.itertuples()]

        self.centrality = self._centrality_within()

    def _centrality_within(self, min_year: int | None = None, max_year: int | None = None) -> dict:
        """Katz centrality of the collaboration graph, optionally restricted to a closed year range."""
        author_sets = [parties for year, parties in self._author_sets_by_year
                       if (min_year is None or year >= min_year) and (max_year is None or year <= max_year)]

        party_set = set()
        for s in author_sets:
            for c in s:
                party_set.add(c)

        # An empty window has no graph to run centrality over.
        if not party_set:
            return {}

        collaboration_graph = networkx.Graph()
        # Add nodes in a deterministic order (country name descending) so the matrix
        # layout used by the centrality computation is reproducible run-to-run.
        collaboration_graph.add_nodes_from(sorted(party_set, reverse=True))

        edge_weights = {}

        for author_set in author_sets:
            for i in range(len(author_set)):
                for j in range(i+1, len(author_set)):

                    if author_set[i] > author_set[j]:
                        c1 = author_set[j]
                        c2 = author_set[i]
                    else:
                        c1 = author_set[i]
                        c2 = author_set[j]
                    
                    if (c1, c2) in edge_weights:
                        edge_weights[(c1, c2,)] += 1
                    else:
                        edge_weights[(c1, c2,)] = 1
        
        # Normalizing our graph decreases our eigenvalues
        if edge_weights:
            max_edge_weight = max(edge_weights.values())
            for c in edge_weights:
                edge_weights[c] /= max_edge_weight

        # Add edges in a deterministic order (edge country-name pair descending).
        for parties, weight in sorted(edge_weights.items(), reverse=True):
            collaboration_graph.add_edge(parties[0], parties[1], weight=weight)

        return networkx.centrality.katz_centrality_numpy(collaboration_graph, alpha=0.1, weight="weight")

    def country_dict(self) -> dict:
        return dict(self.centrality)

    def figure_title(self) -> str:
        return "WP Collaboration Graph Centrality"

    def save_full_figures(self, path: str):
        # Recomputed per decade rather than summed: see DECADE_BUCKETS. Centrality is
        # normalised within its own graph, so values are comparable across countries
        # inside a decade but NOT across decades, and they do not relate to
        # country_dict() by any sum.
        period_figures = []
        for label, min_year, max_year in DECADE_BUCKETS:
            for country, centrality in sorted(self._centrality_within(min_year, max_year).items()):
                period_figures.append({"period": label, "country": country, "value": centrality})
        pd.DataFrame(period_figures).to_csv(path)


class WPCollaborationDiversity():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_type"] == "ATCM") & (wp_authorship_table["party_type"] == "wp")][["parties", "meeting_year", "paper_id"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= START_YEAR) & (wp_authorship_table["meeting_year"] <= END_YEAR)]
        wp_authorship_table = wp_authorship_table.drop_duplicates(subset="paper_id", keep="first")
        
        author_sets = []

        for row in wp_authorship_table.itertuples():
            parties = row.parties
            author_sets.append(split_parties(parties))
        
        collaborations = dict()
        for s in author_sets:
            for i in s:
                for j in s:
                    if i != j:
                        if i not in collaborations:
                            collaborations[i] = set([j])
                        else:
                            collaborations[i].add(j)

        # Note this includes collaboration with agencies.
        self.diversity = {k: len(v) for k, v in collaborations.items()}

    def country_dict(self) -> dict:
        return dict(self.diversity)

    def figure_title(self) -> str:
        return "WP Collaboration Diversity"


if __name__ == "__main__":
    WorkingPaperAuthorship().save_full_figures("test.csv")
    WPCollaborationGraphCentrality()
    print(WPCollaborationDiversity().country_dict())