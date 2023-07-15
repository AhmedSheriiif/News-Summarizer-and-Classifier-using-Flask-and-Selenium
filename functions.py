# Prepare libraries and data
import pandas as pd
import re
import string
from pyarabic.araby import strip_harakat
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
from string import punctuation
import heapq  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Text categories
# categories = ['Sports', 'Economy', 'Health', 'Economy', 'Health', 'Politics']
categories = ['رياضة', 'اقتصاد', 'صحه', 'اقتصاد', 'صحه', 'سياسة']
categories_supervised = ['اقتصاد', 'أخبار متنوعة', 'سياسة', 'رياضة', 'تكنولوجيا']

##Reading arabic data
# Load Arabic dataset
ar_data = pd.read_csv(r"datasets/arabic_dataset.csv")
ar_data = ar_data.replace("diverse", "diverse news")
ar_data = ar_data.replace("culture", "diverse news")
ar_data = ar_data.replace("politic", "politics")
ar_data = ar_data.replace("technology", "tech")
ar_data = ar_data.replace("economy", "economy & business")
ar_data = ar_data.replace("internationalNews", "politics")
ar_data = ar_data[~ar_data['type'].str.contains('localnews')]
ar_data = ar_data[~ar_data['type'].str.contains('society')]

# Building the summarizer
def nltk_summarizer_text(text, number_of_sentence):
    stopWords = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    word_frequencies = {}  
    for word in nltk.word_tokenize(text):  
        if word not in stopWords:
            if word not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 100:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(number_of_sentence, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)  
    return summary

# Building the summarizer
def nltk_summarizer_dataframe(dataframe, number_of_sentence):
    # stopWords = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    # word_frequencies = {}
    dataframe['summary'] = dataframe.Text.map(lambda s: nltk_summarizer_text(s,number_of_sentence))  
    return dataframe

# Load  dataset
df_train = pd.read_csv("datasets/101121_news_source_HSEP_train.csv" )
df_test = pd.read_csv("datasets/101121_news_source_HSEP_test.csv" )
df = pd.concat([df_train, df_test], axis=0)
df.drop(columns=["ID"], inplace=True)
df = df.reset_index(drop=True)


#Cleaning data
arabic_punctuations = '''«»`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def clean(text):
    output = re.sub(r'\s*[A-Za-z]+\b', ' ' , text) #Remove english letters
    output = strip_harakat(output) #Remove harakat   
    translator = str.maketrans(' ',' ', punctuations_list) #remove arabic and english punctuations
    output = output.translate(translator)
    output = " ".join(output.split()) #remove extra spaces
    output = re.sub('\w*\d\w*', ' ', output)# Remove numbers
    return output


#Cleaning Dataframes
def apply_clean_dataframe(dataframe):
    dataframe.Content = dataframe.Content.map(clean)
    dataframe.Subject = dataframe.Subject.map(clean)

    dataframe["Text"] = dataframe.Subject + " " + dataframe.Content
    return dataframe
    
#Cleaning String
def apply_clean_string(input_text):
    out_text = clean(input_text)
    return out_text

apply_clean_dataframe(df)

ar_data['Processed Text'] = ar_data['text'].apply(apply_clean_string)
# After label encoding we sholud change some labels to another becouse the arabic dataset labels is not the same with english dataset
ar_label_encoder = LabelEncoder()
ar_data['Category Encoded'] = ar_label_encoder.fit_transform(ar_data['type'])

ar_X_train, ar_X_test, ar_y_train, ar_y_test = train_test_split(ar_data['Processed Text'], ar_data['Category Encoded'], test_size=0.2, random_state=0)

##nltk.download("stopwords")
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))

documents = df.Text.copy()

#applying tfidf vectorizer 
def tfidf_features(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words=arb_stopwords,min_df=0.01)
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test


# model = pickle.load(open('models/ar_kmeans.pkl', 'rb'))
# model.fit(X_fit)

# prediction = model.predict(X_fit)
def add_cluster_col(dataframe,xx):
    dataframe["Topic_AR"] = xx
    dataframe.Topic_AR.replace({0:"رياضة", 1:"إقتصاد", 2:"صحة", 3:"إقتصاد", 4:"صحة" ,5:"سياسة"}, inplace=True)

    dataframe["Topic_EN"] = xx
    dataframe.Topic_EN.replace({0: "sport", 1:"Eco", 2:"Health", 3:"Eco", 4:"Health", 5:"politics" }, inplace=True)

# add_cluster_col(df,prediction)

## Summarize and predict for KMeans:
def summerize_category_text_kmeans(input_text, statements, model_name):
    summary_text = nltk_summarizer_text(input_text, statements)
    print("----------------------------------------------------------------------------")
    print("Text summary")
    print("----------------------------------------------------------------------------")
    print(summary_text)
    print("----------------------------------------------------------------------------")
    input_text = str(input_text)
    input_text_arr = [apply_clean_string(input_text)]
  
    X_trans ,y_trans = tfidf_features(documents,input_text_arr)

    text_predection = model_name.predict(y_trans)
    text_category = categories[text_predection[0]]
    print("Text category:", text_category)
    print("----------------------------------------------------------------------------")
    return summary_text,text_category


# Summarize and predict for supervised models:
def tfidf_features_supervised(X_train, X_test, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, ngram_range))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test
def summarize_category_supervised(input_text, statements, model_name):
    statements = int(statements)
    summary_text = nltk_summarizer_text(input_text, statements)
    input_text_arr = [apply_clean_string(input_text)]
    
    features_train, features_test = tfidf_features_supervised(ar_X_train, input_text_arr, 2)
    
    text_prediction = model_name.predict(features_test.toarray())
    text_category = categories_supervised[text_prediction[0]]
    return summary_text, text_category

