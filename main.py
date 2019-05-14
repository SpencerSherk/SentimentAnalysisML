import json
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# load dataset into json object
with open('test2.json', 'r') as file_input:
	json_data=json.load(file_input)

# load json into pandas dataframe
data = pd.DataFrame(json_data)

# stemmer & roots (OPTIMIZE: try other stemmers might work better)
stemmer = SnowballStemmer('english')
words = stopwords.words("english")

# ensure every val in dataframe is type str
data["content"] = data["content"].astype(str) 

# normalize data: strip non-letter chars, split text to list of words, stem, convert list back to strings, set to lowercase, fill "cleaned dataframe"
data['cleaned'] = data["content"].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

# use 80% of dataframe as training set, 20% as test set, "cleaned" dataframe = features, "sentiment" dataframe = target vector
X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.sentiment, test_size=0.2)

# create analysis pipeline (ONLY WORKS ON LINUX MACHINES)							
pipeline = Pipeline([ 
					
					 # create bag of words with TfidfVectorizer, use bigram model, ignore stopwords
					 ('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),

					 # choose best 1000 features (OPTIMIZE: MORE FEATURES IF BIGGER DATASET)
                     ('chi',  SelectKBest(chi2, k=1000)), 

                     # use Linear Support Vector Classifier (OPTIMIZE: TUNE THESE PARAMETERS)
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))]) # might need to feature tune the penalty

# fit model to training data (pipeline applies above steps)
model = pipeline.fit(X_train, y_train)

# 
vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

# debugging output: shows 10 most relevant words per class of classification
target_names = ['empty', 'sadness', 'worry', 'neutral', 'surprise', 'love', 'fun', 'hate', 'happiness']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))

# output accuracy score based on built-in scikit accuracy scorer (will need to describe how this works in paper)
print("accuracy score: " + str(model.score(X_test, y_test)))

# test model for given phrase
print(model.predict(['I love everyone. Life is amazing! yay!']))
