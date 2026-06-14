from bs4 import BeautifulSoup
import requests
import pathlib
import json
import downloaders.final_report_ocr as final_report_ocr

def download_pdf_to_dir(dump_directory: pathlib.Path, url: str):
    result = requests.get(url)

    if result.status_code != 200:
        raise Exception(f"Bad status code {result.status_code}")

    url_final_comp = url.split("/")[-1]
    with open(dump_directory / url_final_comp, "wb") as f:
        f.write(result.content)
    
    return url_final_comp

def download_final_reports_from_vol_2_page(dump_directory: pathlib.Path, vol_2_page_url):
    vol_2_page = requests.get(f"https://www.ats.aq{vol_2_page_url}").content
    vol_2_page = BeautifulSoup(vol_2_page, features="html.parser")

    data_table = vol_2_page.find_all(class_="table__data")[0]
    links = data_table.find_all("a")
    
    pdfs = []
    for link in links:
        if "documents.ats.aq" in link["href"]:
            pdfs.append(download_pdf_to_dir(dump_directory, link["href"]))
    
    return pdfs

def download_final_reports(dump_directory: pathlib.Path):
    final_reports_request = requests.get("https://www.ats.aq/devAS/Info/FinalReports?lang=e")
    final_report_pdf_to_atcm = {}

    if final_reports_request.status_code == 200:
        reports_landing_page_soup = BeautifulSoup(final_reports_request.content, features="html.parser")

        results_table_body = reports_landing_page_soup.find_all(class_="table__results")[0]
        results_table_rows = results_table_body.find_all("tr")
        for row in results_table_rows:
            children = list(row.find_all("td"))
            meeting = children[0].text.split("-")[0].strip()
            if "SATCM" in meeting or "ATCM" not in meeting:
                continue

            print(f"Downloading reports for {meeting}")

            english_docs = children[3]
            link_divs = english_docs.find_all("div", recursive=True)

            for link_div in link_divs:
                name_span = link_div.find("span")
                doc_name = str(name_span.contents[0]).lower()
                link = link_div.find("a")
                
                # Lots of special casing to avoid redundant downloads.
                if "10 mb" in doc_name:
                    continue
                elif "(published version)" in doc_name:
                    continue

                if "documents.ats.aq" in link['href']:
                    pdf_name = download_pdf_to_dir(dump_directory, link['href'])
                    final_report_pdf_to_atcm[pdf_name] = meeting
                elif "/devAS/" in link['href']:
                    if "ATCM XXXVI" in meeting:
                        continue
                    
                    pdfs = download_final_reports_from_vol_2_page(dump_directory, link["href"])
                    for pdf in pdfs:
                        final_report_pdf_to_atcm[pdf] = meeting
    
    with open(dump_directory / "pdf_to_atcm.json", "w") as f:
        json.dump(final_report_pdf_to_atcm, f)

def run_final_report_downloading_pipeline(dump_directory = pathlib.Path("data/final_reports")):
    if not (dump_directory / "download.complete").exists():
        download_final_reports(dump_directory)
        (dump_directory / "download.complete").touch()

    if not (dump_directory / "ocr.complete").exists():
        final_report_ocr.ocr_full_directory(dump_directory)

if __name__ == "__main__":
    run_final_report_downloading_pipeline()