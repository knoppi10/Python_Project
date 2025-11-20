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
    print(url)

requests.get(url)

