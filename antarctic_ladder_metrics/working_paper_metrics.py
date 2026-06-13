import pandas as pd
from collections import Counter

import networkx

class WorkingPaperAuthorship():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[wp_authorship_table["meeting_type"] == "ATCM"][["parties", "meeting_year"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= 2000) & (wp_authorship_table["meeting_year"] <= 2024)]
        wp_authorship_table["First Author"] = wp_authorship_table["parties"].map(lambda l: l[0])
        
        first_authors = list(wp_authorship_table["First Author"])
        first_authors = list(map(lambda s: s.lower(), filter(lambda s: '|' not in s, first_authors)))     # Entries containing a | are sorted in alphabetical author and have no real first authorship information.
        
        self.first_authorships = Counter(first_authors)
    
    def country_dict(self) -> dict:
        return dict(self.first_authorships)

    def figure_title(self) -> str:
        return "Working Papers"

class WPCollaborationGraphCentrality():
    def __init__(self) -> None:
        wp_authorship_table = pd.read_parquet("data/antarctic-db/processed/document-summary.parquet")
        wp_authorship_table = wp_authorship_table[wp_authorship_table["meeting_type"] == "ATCM"][["parties", "meeting_year"]]
        wp_authorship_table = wp_authorship_table[(wp_authorship_table["meeting_year"] >= 2000) & (wp_authorship_table["meeting_year"] <= 2024)]
        
        author_sets = []

        for row in wp_authorship_table.itertuples():
            parties = row.parties
            author_sets.append([s.strip().lower() for p in parties for s in p.split('|')])
        
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

if __name__ == "__main__":
    WorkingPaperAuthorship()
    WPCollaborationGraphCentrality()