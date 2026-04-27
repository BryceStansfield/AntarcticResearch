import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
data = pd.DataFrame(columns=['Document_Number', 'Subject', 'Status', 'Category', 'Topics', 'Title', 'Content'])

base_url = 'https://www.ats.aq/devAS/Meetings/Measure/'

for i in range(853, 750, -1):
    url = f"{base_url}{i}"
    print(i)

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        elements = soup.find_all('span', class_='characteristics__item__text__text')
        content_div = soup.find_all('div', class_='text-container')
        Title = soup.find('h1', class_='title')

        if len(elements) >= 0:
            subject = elements[0].get_text(strip=True) if len(elements) > 0 else None
            status = elements[1].get_text(strip=True) if len(elements) > 1 else None
            category = elements[2].get_text(strip=True) if len(elements) > 2 else None
            topics = elements[2].get_text(strip=True) if len(elements) > 3 else None
            content = content_div[1].get_text(strip=True) if len(content_div) > 1 else None
            title = Title.get_text(strip=True) if content_div[0] else None
            data.loc[len(data)] ={
                'Document_Number': i,
                'Subject': subject,
                'Status': status,
                'Category': category,
                'Topics': topics,
                'Title': title,
                'Content': content
            }
        else:
            print(f"Couldn't find enough information on page {i}")

        time.sleep(1)

    except requests.HTTPError as e:
        print(f"Failed to retrieve page {i}: {e}")
    except requests.RequestException as e:
        print(f"A request error occurred: {e}")

data.to_csv('data/MeasureCorpus_Latest_2.csv', index=False)