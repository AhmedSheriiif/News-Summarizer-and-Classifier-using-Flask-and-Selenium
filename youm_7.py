import pandas as pd
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv 
import random
import threading as th
from selenium.webdriver.common.keys import Keys
import functions as func
import pickle
#driver path 
driverpath = r"D:\ITI\Phase II\NLP Project\scrap\chromedriver.exe"
#initialize driver
option = webdriver.ChromeOptions()
option.add_argument("-disk-chache-size=3000000000")
driver = webdriver.Chrome(executable_path=driverpath)
#open the page
driver.get("https://www.youm7.com/")

driver.find_element_by_css_selector("li[class='homeIco']").click()
driver.find_element_by_css_selector("body > header > div.row.marigin0.headerNewNew > div > nav > div > ul > li:nth-child(2) > a > h2").click()
title = []
date = []
topic = []
test = driver.find_elements_by_css_selector("div[class='col-xs-12 bigOneSec']")

#geting links in array
links_1 = []
for i in test:
    links = i.find_element_by_css_selector("a").get_attribute('href')
    links_1.append(links)
    print(links,'\n')

#opening each link in the array  
for i in links_1:

    driver.get(i)
    
    title.append(driver.find_element_by_css_selector('div[class="articleHeader"] > h1').text)
    date.append(driver.find_element_by_css_selector('span[class="newsStoryDate"]').text)
    topic.append(driver.find_element_by_css_selector('div[id="articleBody"]').text)
#     print(title[i],'\n')
#     print(date[i],'\n')
#     print(topic[i],'\n')
#     print("....................................................................")
    
doc_ = {'date':date,
       'Subject':title,
       'Content':topic}
# df = pd.DataFrame(date,title,topic,columns=['date','title','topic'])
df = pd.DataFrame(doc_)

model = pickle.load(open('models/ar_kmeans.pkl', 'rb'))
