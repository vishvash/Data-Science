############ deployment #########################

import streamlit as st
# import newspaper
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
# from nltk import ngrams
import re
# Function to extract article content from a given URL
def extract_article_content(url, language_code):
    article = Article(url, language= language_code)
    article.download()
    article.parse()
    article.nlp()
    return {
        'URL': url,
        'Topic': article.title,
        'Text': article.text,
        'Summary': article.summary,
        'Keywords': ', '.join(article.keywords)
    }

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def generate_word_cloud(text, title, filename, mode = "unigram"):
    if mode == "unigram":
        wordcloud = WordCloud(width=800, height=400).generate(text)
    else:
        wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, format='png')  # Save the word cloud as a PNG file
    plt.close()
     

# Streamlit web application
def main():
    st.set_page_config(
        page_title="Article Analysis",
        
        layout = 'wide'
    )
    
    st.markdown("""
    <style>
        [data-testid="block-container"]
        {
            width: 1100px; /* Replace with your desired width */
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("Article Analysis")
    st.markdown('<p style="color:orange;">News Websites: You may extract text from these well-known, free-to-access websites, which include BBC News, CNN, The New York Times, The Guardian, Indian Express, etc.</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:green;">For further information on libary, which is used to extract text from news articles, click on the website given below.</p>', unsafe_allow_html=True)
    st.write(" 👉👉 [link](https://newspaper.readthedocs.io/en/latest/)")
    # Get user input for the URL
    url = st.text_input("Enter the URL of Article (Prefer English Article for accurate analysis):")
    
    if st.button("Extract Data"):
        # Extract article content
        
        article_data = extract_article_content(url, 'en')
        
        # Create a dataframe from the extracted data
        df = pd.DataFrame([article_data])
        
            
        TOInews = re.sub("[^A-Za-z" "]+", " ", article_data['Text']).lower()

        # Tokenize
        TOInews_tokens = TOInews.split(" ")
        
        with open("stop.txt", "r") as sw:
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
        # frequencies = list(reversed([i[1] for i in tokens_frequencies]))
        words = list(reversed([i[0] for i in tokens_frequencies]))
        cleanstrng = " ".join(words)
        
        # Display the dataframe
        st.subheader("Article Data:")
        st.dataframe(df)
        
        # Perform sentiment analysis
        # sentiment_score = perform_sentiment_analysis(article_data['Text'])
        # st.subheader("Sentiment Analysis:")
        # st.write("Sentiment Score:", sentiment_score)
        # st.write(article_data['Text'])
        # Generate word cloud
        # unigrams = article_data['Text'].split()
        # st.write(unigrams)
        # bigrams = [' '.join(grams) for grams in ngrams(article_data['Text'].split(), 2)]
        # st.write(bigrams)
        st.subheader("Word Clouds:")
        st.subheader("Unigram Word Cloud:")
        generate_word_cloud(cleanstrng, "Unigram - " + article_data['Topic'], "unigram_wordcloud.png")
        st.image("unigram_wordcloud.png")
        st.subheader("positive wordcloud:")
        # positive words
        with open("positive-words.txt", "r") as pos:
            poswords = pos.read().split("\n")
        # Positive word cloud
        # Choosing the only words which are present in positive words
        pos_tokens = " ".join ([w for w in words if w in poswords])
        generate_word_cloud(pos_tokens, "Unigram - " + article_data['Topic'], "positive_wordcloud.png")
        st.image("positive_wordcloud.png")
        st.subheader("negative wordcloud:")
        # Negative words
        with open("negative-words.txt", "r") as neg:
            negwords = neg.read().split("\n")
            
        # Negative word cloud
        # Choosing the only words which are present in negwords
        neg_tokens = " ".join ([w for w in words if w in negwords])
        generate_word_cloud(neg_tokens, "Unigram - " + article_data['Topic'], "negative_wordcloud.png")
        st.image("negative_wordcloud.png")
        
        st.subheader("Bigram Word Cloud:")
        bigrams_list = list(nltk.bigrams(words))

        dictionary2 = [' '.join(tup) for tup in bigrams_list]

        # Using count vectorizer to view the frequency of bigrams
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range = (2, 2))
        bag_of_words = vectorizer.fit_transform(dictionary2)
        # vectorizer.vocabulary_

        sum_words = bag_of_words.sum(axis = 0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

        words_dict = dict(words_freq)

        
        generate_word_cloud(words_dict, "Bigram - " + article_data['Topic'], "bigram_wordcloud.png", mode = 'bigram')
        st.image("bigram_wordcloud.png")

# Run the Streamlit application
if __name__ == '__main__':
    main()