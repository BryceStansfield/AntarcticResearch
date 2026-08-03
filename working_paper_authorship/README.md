Files related to authorship classification on WPs.

authorship_performance_figures.py - Produces charts of the classifiers performance.
country_authorship_classifier.py - The main classifier, and hyperparam tuning code. This file has a special flag for reporting statistics on the test distribution, but I've not run it to ignore any meta-data-leaks.
country_signal_projection.py/direct_country_signal_probe.py - Orthogonal decomposition of direct country authorship information.
feature_importance_report.py - A report on which feature dimensions were most important for authorship classification.
low_dim_pca_experiment.py - An experiment in making a better generalizing classifier by stripping out more dimensions (in particular aimed at measure prediction). Failure (probably due to hyper-param tuned pca in base classifier).


----------
HISTORY:
This directory used to contain a few experiments which didn't pan out. If you want to see details I would recommend looking at the git history of this repo (or to try and re-implement them yourself!)

1) LLM Finetuning - LLMs massively overfit on the data.
2) Measure ratification prediction - Super OOD, and a massive failure. Measure ratification prediction was several times worse than *random guesses* in terms of average Binary Cross Entropy.
3) Sentence by sentence classification. Not enough signal in even quite large document chunks. NOTE: If you're doing more work on this, I think this is where you'll find the most luck extending my work; I think a better method of filtering sentences could help here.
