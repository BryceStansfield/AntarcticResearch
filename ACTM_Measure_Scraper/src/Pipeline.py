def scrape_and_enrich_measures(measure_corpus_output_path, enriched_output_path, failure_list_file = ''):
    from MeasureScraper import scrape_data_if_not_exists
    scrape_data_if_not_exists(output_file=measure_corpus_output_path, failure_list_file=failure_list_file)

    from MeasureEnricher import enrich_if_not_exists
    enrich_if_not_exists(measures_df_path=measure_corpus_output_path, output_path=enriched_output_path)
