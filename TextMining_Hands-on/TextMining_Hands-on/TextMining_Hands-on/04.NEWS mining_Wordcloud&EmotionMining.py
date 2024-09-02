# pip install newspaper3k
'''
Features of Newspaper3k

Multi-threaded article download framework
News URL identification
Text extraction from HTML
Top image extraction from HTML
All image extraction from HTML
Keyword extraction from text
Summary extraction from text
Author extraction from text
Google trending terms extraction
Works in 10+ languages (English, Chinese, German, Arabic,…)
'''


# import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

    # pip install newspaper
# pip3 install newspaper3k

import newspaper

from newspaper import Article

# Support on different languages
newspaper.languages()

# Documentation of the newspaper3k module 
# Newspaper is an amazing python library for extracting & curating articles
# https://newspaper.readthedocs.io/en/latest/
# https://pypi.org/project/newspaper3k/

# Data Extraction from Times of India e-portal
url = 'https://timesofindia.indiatimes.com/sports/cricket/india-in-west-indies/ravichandran-ashwin-indias-greatest-match-winner-since-anil-kumble/articleshow/101793774.cms'

# https://timesofindia.indiatimes.com/world/rest-of-world/singapore-hangs-indian-origin-man-over-1-kg-of-cannabis/articleshow/99800442.cms
# https://economictimes.indiatimes.com/markets/stocks/news/facebook-owner-meta-touts-ai-might-as-digital-ads-boost-outlook-shares-jump/articleshow/99801441.cms
# BBC: https://www.bbc.com/news/technology-65410293


# If no language is specified, Newspaper library will attempt to auto-detect a language.

# Scrap data from a given URL Download the content
article_name = Article(url, language="en")

article_name.download() 

# Parse the content from the html document
article_name.parse() 

# HTML content extracted
article_name.html

# Keyword extraction wrapper
article_name.nlp()

print("Article Title:") 
print(article_name.title) # prints the title of the article
print("\n") 

print("Article Text:") 
print(article_name.text) # prints the entire text of the article
print("\n") 

print("Article Summary:") 
print(article_name.summary) # prints the summary of the article
print("\n") 

print("Article Keywords:")
print(article_name.keywords) # prints the keywords of the article


# Write the extracted data into text file
file1 = open("News1.txt", "w+")
file1.write("Title:\n")
file1.write(article_name.title)

file1.write("\n\nArticle Text:\n")
file1.write(article_name.text)

file1.write("\n\nArticle Summary:\n")
file1.write(article_name.summary)

file1.write("\n\n\nArticle Keywords:\n")
keywords = '\n'.join(article_name.keywords)

file1.write(keywords)
file1.close()

# Read the text from the file
with open("News1.txt", "r") as file2:
    text = file2.read()
    
TOInews = re.sub("[^A-Za-z" "]+", " ", text).lower()

# Tokenize
TOInews_tokens = TOInews.split(" ")

with open("C:/Users/Lenovo/Downloads/Study material/Data Science/TextMining_Hands-on/TextMining_Hands-on/TextMining_Hands-on/dictionaries/stop.txt", "r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

# Cleaned tokens
tokens = [w for w in TOInews_tokens if not w in stop_words]

from collections import Counter
tokens_frequencies = Counter(tokens)

# tokens_frequencies = tokens_frequencies.loc[tokens_frequencies.text != "", :]
# tokens_frequencies = tokens_frequencies.iloc[1:]

# Sorting
tokens_frequencies = sorted(tokens_frequencies.items(), key = lambda x: x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in tokens_frequencies]))
words = list(reversed([i[0] for i in tokens_frequencies]))

# Barplot of top 10 
# import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = ['red','green','black','yellow','blue','pink','violet'])

plt.xticks(list(range(0, 11), ), words[0:11])
plt.xlabel("Tokens")
plt.ylabel("Count")
plt.show()
##########


# Joinining all the tokens into single paragraph 
cleanstrng = " ".join(words)

wordcloud_ip = WordCloud(background_color = 'White',
                      width = 2800, height = 2400).generate(cleanstrng)
plt.axis("off")
plt.imshow(wordcloud_ip, interpolation='bilinear')


# positive words
with open("C:/Users/Lenovo/Downloads/Study material/Data Science/TextMining_Hands-on/TextMining_Hands-on/TextMining_Hands-on/dictionaries/positive-words.txt", "r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
pos_tokens = " ".join ([w for w in words if w in poswords])

wordcloud_positive = WordCloud(background_color = 'White', width = 1800,
                               height = 1400).generate(pos_tokens)
plt.figure(2)
plt.axis("off")
plt.imshow(wordcloud_positive, interpolation='bilinear')


# Negative words
with open("C:/Users/Lenovo/Downloads/Study material/Data Science/TextMining_Hands-on/TextMining_Hands-on/TextMining_Hands-on/dictionaries/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# Negative word cloud
# Choosing the only words which are present in negwords
neg_tokens = " ".join ([w for w in words if w in negwords])

wordcloud_negative = WordCloud(background_color = 'black', width = 1800,
                               height=1400).generate(neg_tokens)
plt.figure(3)
plt.axis("off")
plt.imshow(wordcloud_negative)


'''Bi-gram Wordcloud'''
# Word cloud with 2 words together being repeated
import nltk
nltk.download('punkt')

# Generate 2 work tokens
bigrams_list = list(nltk.bigrams(words))

dictionary2 = [' '.join(tup) for tup in bigrams_list]

# Using count vectorizer to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range = (2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

words_dict = dict(words_freq[:100])


wordcloud_2 = WordCloud(background_color = 'white', width = 1800, height = 1400)
plt.figure(4)                     
wordcloud_2.generate_from_frequencies(words_dict)
plt.imshow(wordcloud_2)

# pip install emoji==1.2.0


''' Emotion Mining'''
# pip install text2emotion
import text2emotion as te
import pandas as pd

text = "I was asked to sign a third party contract a week out from stay. If it wasn't an 8 person group that took a lot of wrangling I would have cancelled the booking straight away. Bathrooms - there are no stand alone bathrooms. Please consider this - you have to clear out the main bedroom to use that bathroom. Other option is you walk through a different bedroom to get to its en-suite. Signs all over the apartment - there are signs everywhere - some helpful - some telling you rules. Perhaps some people like this but It negatively affected our enjoyment of the accommodation. Stairs - lots of them - some had slightly bending wood which caused a minor injury."
te.get_emotion(text)

# Capturing the Emotions from Tokens
emotion = te.get_emotion('work')
emotion

emotion = te.get_emotion('worst')
emotion

emotion = te.get_emotion('proper')
emotion


# Capture Emotions for the News article
emotions = []

# Capture the emotions on the tokens
for i in words:
    emotions_r = te.get_emotion(i)
    emotions.append(emotions_r)

## Call to the function
# emotions = te.get_emotion(ip_rev_string)
emotions = pd.DataFrame(emotions)
emotions

tokens_df = pd.DataFrame(tokens, columns=['words'])

emp_emotions = pd.concat([tokens_df, emotions], axis = 1)
emp_emotions.columns

emp_emotions[['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']].sum().plot.bar()


########## End ###########


# Alternately code for wordcloud.
# Create a word cloud
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear') # Display the word cloud
plt.axis("off")
plt.show()



##########
url= 'https://www.axios.com/2023/05/23/elon-musk-ai-regulation-china'
article = Article(url, language="en") # en for English 
article.download() 
article.parse() 
article.nlp() 

file1 =  open("article51.txt", "w+")
file1.write(article.title)
file1.write("\n\n")
file1.write(article.text)
file1.close()






################

# pip install newsapi-python
# from newsapi import NewsApiClient
# import pandas as pd
# import datetime as dt

# newsapi = NewsApiClient(api_key = 'f860364762db4c5a961ca7cc8765f539')

# data = newsapi.get_everything(q = 'illegal drugs', language = 'en', page_size = 100)

# articles = data['articles']

# df = pd.DataFrame(articles)

# df

# df.drop('publishedAt', axis = 1, inplace = True)

# df.to_csv('Article.csv')

# a = list(df['content'])
  
# # converting list into string and then joining it with space
# b = ' '.join(str(e) for e in a)
  
# with open("Article100.txt", "w", encoding = 'utf8') as output:
#     output.write(str(b))
