import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_data(output_file = 'data/MeasureCorpus.csv', failure_list_file = ''):
    data = pd.DataFrame(columns=['Document_Number', 'Subject', 'Status', 'Category', 'Topics', 'Title', 'Content', 'Approvals'])

    base_url = 'https://www.ats.aq/devAS/Meetings/Measure/'
    failure_list = []

    i = 1
    failed_in_row = 0
    while True and failed_in_row < 5:
        print(f"Scraping measure #{i}")
        url = f"{base_url}{i}"

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            if "An error occurred while processing your request." in soup.get_text():
                print(f"Scraped all measures.")
                break

            elements = soup.find_all('span', class_='characteristics__item__text__text')
            approval_container = soup.find("aside")
            content_div = soup.find_all('div', class_='text-container')
            Title = soup.find('h1', class_='title')

            if len(elements) >= 0:
                subject = elements[0].get_text(strip=True) if len(elements) > 0 else None
                status = elements[1].get_text(strip=True) if len(elements) > 1 else None
                category = elements[2].get_text(strip=True) if len(elements) > 2 else None
                topics = list(filter(lambda s: s is not None and s != '', [t.strip() for t in elements[3].get_text(strip=True).split('-')])) if len(elements) > 3 else None
                content = content_div[1].get_text(strip=True) if len(content_div) > 1 else None
                title = Title.get_text(strip=True) if Title else None

                if approval_container != None:
                    approvals = approval_container.find_all("tr")

                    if approvals == None or approvals == []:
                        approvals_text = approval_container.find("p").text
                    else:
                        approvals = list(filter(lambda tr: tr.find("th") != None and tr.find("td") != None, approvals))
                        approvals_text = "\n".join(map(lambda tr: tr.find("th").text + " " + tr.find("td").text, approvals))
                else:
                    approvals_text = ""
                
                # NOTE: Currently attachements are unscraped.
                data.loc[len(data)] ={
                    'Document_Number': i,
                    'Subject': subject,
                    'Status': status,
                    'Category': category,
                    'Topics': topics,
                    'Title': title,
                    'Content': content,
                    'Approvals': approvals_text
                }
            else:
                print(f"Couldn't find enough information on page {i}")
                failure_list.append(url)

            failed_in_row = 0
        
        except Exception as e:
            failed_in_row += 1
            failure_list.append(url)
            if isinstance(e, requests.HTTPError):
                print(f"Failed to retrieve page {i}: {e}")
            elif isinstance(e, requests.RequestException):
                print(f"A request error occurred: {e}")
            else:
                raise e
        i += 1

    data.to_csv(output_file, index=False)

    if failure_list_file != '':
        with open('data/scraping_failure_list.txt', 'w') as f:
            for url in failure_list:
                f.write(f"{url}\n")

def scrape_data_if_not_exists(output_file = 'data/MeasureCorpus.csv', failure_list_file = ''):
    try:
        data = pd.read_csv(output_file)
        print(f"Data already exists in {output_file}. Skipping scraping.")
    except FileNotFoundError:
        print(f"{output_file} not found. Starting scraping process.")
        scrape_data(output_file, failure_list_file)

if __name__ == "__main__":
    scrape_data_if_not_exists()