if __name__ == "__main__":
    from MeasureScraper import scrape_data_if_not_exists
    scrape_data_if_not_exists(failure_list_file='data/scraping_failure_list.txt')

    from MeasureEnricher import enrich_if_not_exists
    enrich_if_not_exists()