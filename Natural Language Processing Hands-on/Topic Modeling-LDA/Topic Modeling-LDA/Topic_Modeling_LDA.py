
"""
from google.colab import files
uploaded = files.upload()
"""


# io.BytesIO(uploaded['undebate1.csv'])
import io
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
undebate = pd.read_csv(r"C:\Users\Lenovo\Downloads\Study material\Data Science\Natural Language Processing Hands-on\Topic Modeling-LDA\Topic Modeling-LDA\undebate.csv")


undebate.info()


print(repr(undebate.iloc[125]["text"][0:200])) #repr a printable representation of an object


print(repr(undebate.iloc[288]["text"][0:200]))


undebate.head(1)


import re
undebate["paragraphs"] = undebate["text"].map(lambda text: re.split('[.?!]\\s*\n', text)) # split at full stops, exclamation points and question marks
undebate["number_of_paragraphs"] = undebate["paragraphs"].map(len)


undebate


# %matplotlib inline
undebate.groupby('year').agg({'number_of_paragraphs': 'mean'}).plot.bar()


from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords


# pip install spacy


stopwords = list(stopwords)


stopwords


tfidf_text = TfidfVectorizer(stop_words = stopwords, min_df = 5, max_df = 0.7)  # min_df means Terms must have DF >= 5 to be considered, max_df # Removes terms with DF higher than the 70% of the documents
vectors_text = tfidf_text.fit_transform(undebate['text'])
vectors_text.shape


vectors_text


# flatten the paragraphs keeping the years
paragraph_df = pd.DataFrame([{ "text": paragraph, "year": year } 
                               for paragraphs, year in \
                               zip(undebate["paragraphs"], undebate["year"]) 
                                    for paragraph in paragraphs if paragraph])


paragraph_df


tfidf_para_vectorizer = TfidfVectorizer(stop_words = stopwords, min_df = 5, max_df = 0.7)
tfidf_para_vectors = tfidf_para_vectorizer.fit_transform(paragraph_df["text"])
tfidf_para_vectors.shape


from sklearn.feature_extraction.text import CountVectorizer

count_para_vectorizer = CountVectorizer(stop_words = stopwords, min_df = 5, max_df = 0.7)
count_para_vectors = count_para_vectorizer.fit_transform(paragraph_df["text"])


from sklearn.decomposition import LatentDirichletAllocation

lda_para_model = LatentDirichletAllocation(n_components = 10, random_state = 42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_vectors) # Documents vs topics
H_lda_para_matrix = lda_para_model.components_   # words vs topics


W_lda_para_matrix


H_lda_para_matrix

def display_topics(model, features, no_top_words = 5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1] # invert sort order
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]],
                  word_vector[largest[i]]*100.0/total))




display_topics(lda_para_model, tfidf_para_vectorizer.get_feature_names_out())


# !pip install pyLDAvis


"""
import pyLDAvis.sklearn

lda_display = pyLDAvis.sklearn.prepare(lda_para_model, count_para_vectors,
                            count_para_vectorizer, sort_topics = False)
pyLDAvis.display(lda_display)
"""

# pip install pyLDAvis


# import pyLDAvis.lda_model

# lda_display = pyLDAvis.lda_model.prepare(lda_para_model, count_para_vectors,
#                             count_para_vectorizer, sort_topics = False)
# pyLDAvis.display(lda_display)



import pyLDAvis.lda_model
from IPython.core.display import display, HTML

lda_display = pyLDAvis.lda_model.prepare(lda_para_model, count_para_vectors,        count_para_vectorizer, sort_topics=False)

pyLDAvis.save_html(lda_display, 'lda_visualization.html')
print("LDA visualization saved as 'lda_visualization.html'")





import matplotlib.pyplot as plt
from wordcloud import WordCloud


def wordcloud_topics(model, features, no_top_words = 40):
    for topic, words in enumerate(model.components_):
        size = {}
        largest = words.argsort()[::-1] # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color = "white", max_words = 100,
                       width = 960, height = 540)
        wc.generate_from_frequencies(size)
        plt.figure(figsize = (12, 12))
        plt.imshow(wc, interpolation = 'bilinear')
        plt.axis("off")
        # if you don't want to save the topic model, comment the next line
        plt.savefig(f'topic{topic}.png')


wordcloud_topics(lda_para_model, count_para_vectorizer.get_feature_names_out())


# create tokenized documents
gensim_paragraphs = [[w for w in re.findall(r'\b\w\w+\b' , paragraph.lower())
                          if w not in stopwords]
                             for paragraph in paragraph_df["text"]]


from gensim.corpora import Dictionary
dict_gensim_para = Dictionary(gensim_paragraphs)


dict_gensim_para.filter_extremes(no_below = 5, no_above = 0.7)

bow_gensim_para = [dict_gensim_para.doc2bow(paragraph) for paragraph in gensim_paragraphs]


from gensim.models import TfidfModel
tfidf_gensim_para = TfidfModel(bow_gensim_para)
vectors_gensim_para = tfidf_gensim_para[bow_gensim_para]


from gensim.models import LdaModel
lda_gensim_para = LdaModel(corpus = bow_gensim_para, id2word = dict_gensim_para,
    chunksize = 2000, alpha = 'auto', eta = 'auto', iterations = 400, num_topics = 10, 
    passes = 20, eval_every = None, random_state = 42)


def display_topics_gensim(model):
    for topic in range(0, model.num_topics):
        print("\nTopic %02d" % topic)
        for (word, prob) in model.show_topic(topic, topn=5):
            print("  %s (%2.2f)" % (word, prob))


display_topics_gensim(lda_gensim_para)


from gensim.models.coherencemodel import CoherenceModel

lda_gensim_para_coherence = CoherenceModel(model = lda_gensim_para, texts = gensim_paragraphs, dictionary = dict_gensim_para, coherence = 'c_v')
lda_gensim_para_coherence_score = lda_gensim_para_coherence.get_coherence()
print(lda_gensim_para_coherence_score)


top_topics = lda_gensim_para.top_topics(vectors_gensim_para, topn = 5)
avg_topic_coherence = sum([t[1] for t in top_topics]) / len(top_topics)
print('Average topic coherence: %.4f.' % avg_topic_coherence)


[(t[1], " ".join([w[1] for w in t[0]])) for t in top_topics]


from tqdm import tqdm


from gensim.models.ldamulticore import LdaMulticore
lda_para_model_n = []
for n in tqdm(range(15, 21)):
    lda_model = LdaMulticore(corpus = bow_gensim_para, id2word = dict_gensim_para,
                             chunksize = 2000, eta = 'auto', iterations = 100,
                             num_topics = n, passes = 20, eval_every = None,
                             random_state = 42)
    lda_coherence = CoherenceModel(model = lda_model, texts = gensim_paragraphs,
                                   dictionary = dict_gensim_para, coherence = 'c_v')
    lda_para_model_n.append((n, lda_model, lda_coherence.get_coherence()))


pd.DataFrame(lda_para_model_n, columns = ["n", "model", \
    "coherence"]).set_index("n")[["coherence"]].plot(figsize = (16,9))


display_topics_gensim(lda_para_model_n[0][1])


