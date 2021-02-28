# import cv2
import pandas as pd
import json as js
from selenium import webdriver
from bs4 import BeautifulSoup
# import datetime
import time
import os

from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

site = "https://www.myfitnesspal.com/nutrition-facts-calories/indian-food"

driver = webdriver.Chrome(ChromeDriverManager().install())
# driver.get(site)

# print(l,len(l))

mainList = []

csvList=[]
for number in range(1,11):
    driver.get(f"{site}/{number}")
    l= driver.find_elements_by_class_name("jss60")
    for element in l:
        json = {}
        a=element.find_element_by_tag_name('a')
        link=a.get_attribute('href')
        # print(a.get_attribute('href'),a.text)
        sub=element.find_element_by_class_name('jss65').text
        info=element.find_element_by_class_name('jss70').text
        name,serving=sub.split(",")[:2]

        if "Indian Food" in name:
            title=a.text
        else:
            title=name
        nutrients={}
        n=info.split("•")
        for item in n:
            item=item.strip()
            nut,qty=item.split(":")
            nutrients[nut]=qty.strip()
            # d={nut:qty.strip()}
            # nutrients.append(d)
        csvList.append(title)
        json["title"]=title
        json["link"]=link
        json["serving"]=serving
        json["nutrients"]=nutrients
        mainList.append(json)         
# print(mainList)
df = pd.DataFrame(csvList)
df.to_csv('data.csv', index=False)
jsonstring=js.dumps(mainList)
with open('data.json',"w") as f:
    f.write(jsonstring)