from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

############################################################################

# PATH = "C:\Program Files (x86)\chromedriver.exe"
service = Service()
options = webdriver.ChromeOptions()
driver = uc.Chrome(service=service, options=options)

############################################################################

# Scraping PrizePicks
driver.get("https://app.prizepicks.com/")

# Waiting and closes popup
wait = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "close")))
driver.find_element(By.XPATH, "/html/body/div[3]/div[3]/div/div/div[1]").click()
time.sleep(1)

# Creating tables for players
ppPlayers = []

# Now it will click on the LoL tab
driver.find_element(By.XPATH, "//div[@class='name'][normalize-space()='VAL']").click()
time.sleep(2)

# Waits until stat container element is viewable
stat_container = WebDriverWait(driver, 1).until(EC.visibility_of_element_located((By.CLASS_NAME, "stat-container")))

# Finding all the props: Map 1 Kills, etc.
categories = driver.find_element(By.CSS_SELECTOR, ".stat-container").text.split('\n')

# Collecting categories
for category in categories:
    driver.find_element(By.XPATH, f"//div[text()='{category}']").click()

    projectionsPP = WebDriverWait(driver, 5).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".projection")))

    for projections in projectionsPP:
        names = projections.find_element(By.CLASS_NAME, "name").text
        pts = projections.find_element(By.CLASS_NAME, "presale-score").get_attribute('innerHTML')
        proptype = projections.find_element(By.CLASS_NAME, "text").get_attribute('innerHTML')

        players = {
            'Name': names,
            'Points': pts,
            'Prop': proptype.replace("<wbr>", "")
        }
        ppPlayers.append(players)

dfProps = pd.DataFrame(ppPlayers)
dfProps.to_csv('test.csv')

print("These are all of the props offered by PP.", '\n')
print(dfProps)
print('\n')