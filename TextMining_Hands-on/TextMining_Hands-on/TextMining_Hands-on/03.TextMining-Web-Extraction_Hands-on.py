# pip install tweepy

# Twitter Extraction
import pandas as pd
import tweepy
from tweepy import OAuthHandler
 
# Your Twitter App Credentials
# https://apps.twitter.com -> https://developer.twitter.com

consumer_key = "0hBc79kiuvx6VC5DmhPHirNyB"
consumer_secret = "JSv4w63E8h2cD8CX62J9uGDvqv0n2Hzx27HSOtT1ZVZqEjGDzx"
access_token = "1749465287576141824-Us8GwiAz8vzGgrsZqeR7ntoOXWzp3B"
access_token_secret = "g0yMP6Wu9jdXQN6BoanHbcAIMh6y4wTLkhKzyA6yFdm8a"

# OAuth 2.0 Client ID and Client Secret
# cG9VMGF0cndPWkp4WFZYdU81S206MTpjaQ
# WbOyYsgCP8AiA_xKFPfEPaBUdL0PhrkZsAvBzHGZsoouiIngGt


# Calling API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# Provide the keyword related to which you want to pull the data e.g. "Python".
keyword = "T20WorldCup"

# Fetching tweets
tweets_keyword = api.search_tweets(keyword, count = 100, lang = 'en',
                            exclude = 'retweets', tweet_mode = 'extended') #Changed
 
for item in tweets_keyword:
    print(item)
    
tweets_for_csv = [tweet.full_text for tweet in tweets_keyword] 


# 200 tweets to be extracted 
tweets_user = api.user_timeline(screen_name = "ShashiTharoor", count = 200)


for item in tweets_user:
    print(item)
# Create array of tweet information: username, tweet id, date/time, text 
tweets_for_csv1 = [tweet.text for tweet in tweets_user] 

# Saving the tweets onto a CSV file
# convert 'tweets' list to pandas DataFrame
tweets_df = pd.DataFrame(tweets_for_csv1, columns = ['Value'])

tweets_df.to_csv('tweets.csv')

import os
os.getcwd()
#########################################

# pip install ntscraper

from ntscraper import Nitter
import json

# https://www.youtube.com/watch?v=qSEa5lMVoI4&ab_channel=PrinceIdris
# from ntscraper import Nitter
# tweets = Nitter().get_tweets("elonmusk", mode='user', number=10)
# print(tweets)
# Here is the link to the Python code on Github with the option to save the data as a json file:
# https://t.co/NRqTVqFC32
# ---------------------------------------------------------------

# scraper = Nitter(log_level=1, skip_instance_check=False)

# github_hash_tweets = scraper.get_tweets("github", mode='hashtag')

# bezos_tweets = scraper.get_tweets("JeffBezos", mode='user')

# terms = ["github", "bezos", "musk"]

# results = scraper.get_tweets(terms, mode='term')

# bezos_information = scraper.get_profile_info("JeffBezos")

# usernames = ["x", "github"]

# results = scraper.get_profile_info(usernames)

# random_instance = scraper.get_random_instance()

tweets = Nitter().get_tweets("JeffBezos", mode='user', number = 100)

help(Nitter)

with open ("jeffbezos","w") as file:
    json.dump(tweets, file, indent = 4)

# Open the JSON file
with open("jeffbezos", "r") as file:
    data = json.load(file)

# Extract text values from each tweet
texts = [tweet["text"] for tweet in data["tweets"]]

# Print or do whatever you want with the extracted text
for text in texts:
    print(text)


#########################################
# Scraping Data from local HTML file

# pip install bs4
from bs4 import BeautifulSoup

soup = BeautifulSoup(open(r'C:/Users/Lenovo/Downloads/Study material/Data Science/TextMining_Hands-on/TextMining_Hands-on/TextMining_Hands-on/dictionaries/sample_doc.html'), 'html.parser')

soup.text

soup.contents

# Look for tag address
soup.find('address')

soup.find_all('address')


# Look for tag 'q' (this denote quotes)
soup.find_all('q')

# Look for tag 'b' (this denote texts in bold font)
soup.find_all('b')

# Look for tag 'table'
table = soup.find('table')
table

for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)

table.find_all('tr')[3].find_all('td')[2]


# Amazon has blocked scrapping by captcha and other automation blocking methods, have to find an alternate way
# import requests
# # Importing requests to extract content from a url
# from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scraping...used to scrap specific content 
# import re
# # pip install wordcloud
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # creating empty reviews list oneplus_reviews
# iqoo = []
# for i in range(1, 21): 
#     ip = []
#     url = "https://www.amazon.in/iQOO-MediaTek-Dimesity-Processor-Smartphone/product-reviews/B07WGPJPR3/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
#     url = "https://www.amazon.in/iQOO-MediaTek-Dimesity-Processor-Smartphone/product-reviews/B07WGPJPR3/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"
#     response = requests.get(url)
#     soup = bs(response.content, "html.parser") # creating soup object to iterate over the extracted content 
#     reviews = soup.find_all("span", attrs = {"class", "a-size-base cr-lightbox-review-body"}) # Ext 

#     for i in range(len(reviews)):
#         ip.append(reviews[i].text)
    
#     iqoo = iqoo + ip # adding the reviews of one page to empty list which in future
    
    
    
    