This folder contains all of the code related to both censoring and embedding working papers.

bertopic_backend.py - Contains all of the code for properly integrating our embeddings with BERTOPIC and generating topic labels.
document_embeddings.py - Contains the code responsible for dispatching embedding tasks to openrouter, and for caching results (otherwise runs cost ~$0.6aud each)
embed_all_documents.py - If run, embeds all WPs/IPs/Measures. For WPs, this is both raw and naively censored (LLM censorship is expensive so it will only initially run when authorship classification is run)
working_paper_censorship.py - Code for censoring working papers.

Code for orthogonal decomposition of embedding space lives in working_paper_authorship.