# Requirements
Dependencies handled with UV.
Golang 1.23+ required for antarctic-database-go usage.

Secrets managed with secrets.json. See secrets.example.json for an example. Only "OPENROUTER_API_KEY" is required for antarctic ladder usage.

Some parts of this pipeline require a complete and OCRd antarctic-database-go database. If you want to do this yourself, an OCR api key will be required (and can be placed in secrets.json). Otherwise, if you have an archived copy, unzip to data/antarctic-db.

If you have an archived copy of the embeddings, place it in data/document_embeddings.sqlite3

# How to run
uv run antarctic_ladder_figure_aggregator.py will run the full antarctic ladder.


# Outputs
Antarctic Ladder figures will persist to data/ladder_results.csv

Full yearly figures saved to data/full_figures

Un-normalized collaboration graphs saved to data/collaboration_graphs

# Methodology
Documentation on Antarctic Ladder methods is available in Methods/