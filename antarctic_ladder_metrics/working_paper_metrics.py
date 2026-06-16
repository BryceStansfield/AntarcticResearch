import pandas as pd
from collections import Counter
from antarctic_ladder_metrics.constants import *

import networkx

from utils import split_parties

class WorkingPaperAuthorship():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_type"] == "ATCM") & (wp_authorship_table["party_type"] == "wp")][["parties", "meeting_year", "paper_id"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= START_YEAR) & (wp_authorship_table["meeting_year"] <= END_YEAR)]
        wp_authorship_table = wp_authorship_table.drop_duplicates(subset="paper_id", keep="first")

        authors = list(wp_authorship_table["parties"].map(split_parties))
        
        self.country_authorships = {}
        for pl in authors:
            for p in pl:
                if p in self.country_authorships:
                    self.country_authorships[p] += 1/len(pl)
                else:
                    self.country_authorships[p] = 1/len(pl)
            
    def country_dict(self) -> dict:
        return dict(self.country_authorships)

    def figure_title(self) -> str:
        return "Working Paper Authorship"

class WPCollaborationGraphCentrality():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_type"] == "ATCM") & (wp_authorship_table["party_type"] == "wp")][["parties", "meeting_year", "paper_id"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= START_YEAR) & (wp_authorship_table["meeting_year"] <= END_YEAR)]
        wp_authorship_table = wp_authorship_table.drop_duplicates(subset="paper_id", keep="first")
        
        author_sets = []

        for row in wp_authorship_table.itertuples():
            parties = row.parties
            author_sets.append(split_parties(parties))
        
        party_set = set()
        for s in author_sets:
            for c in s:
                party_set.add(c)
        
        collaboration_graph = networkx.Graph()
        collaboration_graph.add_nodes_from(party_set)

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
                        edge_weights[(c1, c2,)] += 1/len(author_set)
                    else:
                        edge_weights[(c1, c2,)] = 1/len(author_set)
        
        # Normalizing our graph decreases our eigenvalues and allows us to use higher attenuation factors.
        max_edge_weight = max(edge_weights.values())
        for c in edge_weights:
            edge_weights[c] /= max_edge_weight
        
        for parties, weight in edge_weights.items():
            collaboration_graph.add_edge(parties[0], parties[1], weight=weight)
        
        self.centrality = networkx.centrality.katz_centrality_numpy(collaboration_graph, alpha=0.1, weight="weight")

    def country_dict(self) -> dict:
        return dict(self.centrality)

    def figure_title(self) -> str:
        return "WP Collaboration Graph Centrality"
    
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
        return "WP Collaboration Graph Centrality"


if __name__ == "__main__":
    WorkingPaperAuthorship()
    WPCollaborationGraphCentrality()
    print(WPCollaborationDiversity().country_dict())