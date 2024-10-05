import pandas as pd
import numpy as np
import nltk
import re                                  
import string
from nltk.corpus import stopwords          
from nltk.stem import WordNetLemmatizer

fake_df = pd.read_csv('E:\projects\Fakenews_Detector\News _dataset\Fake.csv')
real_df = pd.read_csv('E:\projects\Fakenews_Detector\News _dataset\True.csv')

fake_df["class"] = 0
real_df["class"] = 1

fake_df.drop(['date', 'subject'], axis=1, inplace=True)
real_df.drop(['date', 'subject'], axis=1, inplace=True)

news_df = pd.concat([fake_df, real_df], ignore_index=True,
                   sort=False)
news_df.head()

news_df["text"] = news_df["title"] + news_df["text"]
news_df.drop("title", axis=1, inplace=True)
news_df.head()

nltk.download('stopwords')
nltk.download('wordnet')

stopwords_english = stopwords.words('english') 

print('Stop words\n')
print(stopwords_english)

print('\nPunctuation\n')
print(string.punctuation)

def remove_punct(text):
    return ("".join([ch for ch in text if ch not in string.punctuation]))

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

def return_sentences(tokens):
    return " ".join([word for word in tokens])


news_df['text'] = news_df['text'].apply(lambda x: remove_punct(x))

news_df['tokens'] = news_df['text'].apply(lambda msg : tokenize(msg))

news_df['filtered_tokens'] = news_df['tokens'].apply(lambda x : remove_small_words(x))

news_df['clean_tokens'] = news_df['filtered_tokens'].apply(lambda x : remove_stopwords(x))

news_df['lemma_words'] = news_df['clean_tokens'].apply(lambda x : lemmatize(x))

news_df['clean_text'] = news_df['lemma_words'].apply(lambda x : return_sentences(x))

news_df.head()

news_df[['clean_text', 'class']].to_csv('clean.csv', index=False)

def preprocessing_pipeline(df):
    df['text'] = df['text'].apply(lambda x: remove_punct(x))

    df['text'] = df['text'].apply(lambda msg : tokenize(msg))

    df['text'] = df['text'].apply(lambda x : remove_small_words(x))

    df['text'] = df['text'].apply(lambda x : remove_stopwords(x))

    df['text'] = df['text'].apply(lambda x : lemmatize(x))

    df['text'] = df['text'].apply(lambda x : return_sentences(x))

    return df

