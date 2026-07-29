START_YEAR = 2000
END_YEAR = 2025

# Some metrics are meaningless over a single year and are reported per decade
# instead: topic diversity degenerates into a publication count, and collaboration
# graph centrality degenerates into centrality on a near-unconnected graph. These
# are recomputed from scratch within each window rather than summed across years,
# because neither is additive. The last bucket is labelled incomplete because
# END_YEAR falls mid-decade.
DECADE_BUCKETS = [
    ("2000s", 2000, 2009),
    ("2010s", 2010, 2019),
    ("2020s (incomplete)", 2020, END_YEAR),
]