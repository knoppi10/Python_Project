import numpy as np
import pandas as pd 
#pip install matplotlib
#import matplotlib as matplotlib
from bs4 import BeautifulSoup
import requests

#pip install pyyaml
import yaml

with open("config.yaml", "r", encoding = "utf-8") as config_file:
    config = yaml.safe_load(config_file) 
    
url = config ['webscrapper']['url']
headers = config ['webscrapper']['headers']
print(url)

responds = requests.get(url, headers=headers)
print (responds)
responds.raise_for_status() #check if request was successful

soup = BeautifulSoup(responds.text, "html")
print (soup)
pretty_soup = soup.prettify()
print (pretty_soup)

#find and find_all methods
soup.find("div)")
all = soup.find_all("div")
table_all = soup.find_all("table")
table = soup.find("table")
print (table)
print (table_all)
print (len (table_all))

print (table.find_all("tr"))

rows = table.find_all("tr")
data = []
for row in rows:
    cols = row.find_all("td")
    cols = [td.text.strip() for td in cols]
    data.append(cols)
#umwandeln in dataframe = Tabellenform
    df = pd.DataFrame(data)
    header = df.iloc
    df = df[1:]
    df.columns = header
    print (df)