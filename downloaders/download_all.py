import downloaders.scrape_final_reports
import downloaders.run_antarctic_db_go_pipeline
import downloaders.map_all_wp_ip_locations
import embeddings.embed_all_documents

def download_and_extract_all():
    downloaders.scrape_final_reports.run_final_report_downloading_pipeline()
    
    if not downloaders.run_antarctic_db_go_pipeline.SENTINEL.exists():
        print(downloaders.run_antarctic_db_go_pipeline.SENTINEL)
        print("Please manually run `uv run python -m downloaders.run_antarctic_db_go_pipeline.py`")
        quit()
    
    downloaders.map_all_wp_ip_locations.map_all_wp_ip_file_locations()
    embeddings.embed_all_documents.embed_all()