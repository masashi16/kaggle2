import numpy as np
import pandas as pd
import string
import re
import os
import pickle

################################################################################################
## 前処理 ##

################################################################################################
## データ読み込み ##
train_path = './data/train.csv'
df_train = pd.read_csv(train_path, sep=',')
#df_train.head()
#len(df_train)

test_path = './data/test.csv'
df_test = pd.read_csv(test_path, sep=',')



# テキストを単語に分割
def data2words(csvdata):
    words_list = []
    for text in csvdata["comment_text"].__iter__():
        text = re.sub(':', ' ', text)  # 改行を削除
        text = re.sub('_', ' ', text)  # 改行を削除
        text = re.sub('#', ' ', text)  # 改行を削除
        text = re.sub('\n', ' ', text)  # 改行を削除
        text = re.sub('\r', '', text)  # \rを削除
        text = re.sub('\xa0', ' ', text)  # \rを削除
        text = re.sub(r'[0-9]', '', text)  # 数字を削除
        text = re.sub('　', ' ', text)  # \rを削除

        text = text.lower()  # 小文字に
        text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation文字を削除

        words_list.append(re.findall(r"[\w]+", text))  # 単語に分割
        #texts.remove('')

    return words_list


# テキストを単語に分割
def text_cleaning(csvdata):
    texts = []
    for text in csvdata["comment_text"].__iter__():
        text = re.sub('=', ' ', text)
        text = re.sub('/', ' ', text)
        text = re.sub(':', ' ', text)
        text = re.sub('_', ' ', text)
        text = re.sub('#', ' ', text)
        text = re.sub('\n', ' ', text)  # 改行を削除
        text = re.sub('\r', '', text)  # \rを削除
        text = re.sub('\xa0', ' ', text)
        text = re.sub(r'[0-9]', '', text)  # 数字を削除
        text = re.sub('　', ' ', text)

        text = text.lower()  # 小文字に
        #text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation文字を削除

        #texts.append(re.findall(r"[\w]+", text))  # 単語に分割
        texts.append(text)

    return texts


#texts_train = text_cleaning(df_train)
#texts_test = text_cleaning(df_test)
#all_texts = []
#all_texts.extend(texts_train); all_texts.extend(texts_test)
#texts_test[:10]
#all_texts[:10]

#txts_train = text_cleaning(df_train)


###############################################################################################
## Doc2vec ##
"""
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

all_texts = df_train["comment_text"].append(df_test["comment_text"])
# Convert a document into a list of tokens
all_tokens = [gensim.utils.simple_preprocess(txt) for txt in all_texts]
all_corpus = [TaggedDocument(words=data, tags=[i]) for i, data in enumerate(all_tokens)]
#all_corpus[0]


num_features = 300     # Word vector dimensionality
min_word_count = 2   # Minimum word count
num_workers = os.cpu_count()   # Number of threads to run in parallel
context = 10          # Context window size

print ("Training Doc2Vec model...")
#all_texts_taggle = [TaggedDocument(words = data.split(),tags = [i]) for i,data in enumerate(all_texts)]  # TaggedDocument(単語に分解された文章, [文章のID])


model = Doc2Vec(all_corpus, workers=num_workers, dm = 1,
            vector_size=num_features, min_count = min_word_count,
            window = context, epochs=100, seed=1)

print ("Saving Doc2Vec model...")
model.save("./d2v/d2v_baseline_300.model")
"""

###############################################################################################
## d2vを用いた学習 ##
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

d2vmodel = Doc2Vec.load("./d2v/d2v_baseline_300.model")

from sklearn.model_selection import train_test_split

#corpus_train = [data.split() for data in texts_train]
corpus_train = [gensim.utils.simple_preprocess(txt) for txt in df_train["comment_text"]]
corpus_train

d2v_train = np.array([d2vmodel.infer_vector(words) for words in corpus_train])
d2v_train.shape

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(d2v_train,
#                                                    np.array(df_train['identity_hate']),
#                                                    stratify=np.array(df_train['identity_hate']),
#                                                    test_size=0.2,
#                                                    random_state=1)


import gc
del texts_train
del d2v_train
gc.collect()

X_train.shape


##############################################################################################

import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

def build_model():
    model = Sequential()

    model.add(Dense(units=200, input_dim=d2v_train.shape[1], activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))


    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

model = build_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

model.fit(d2v_train, np.array(df_train['identity_hate']), batch_size=1024, epochs=100, validation_split=0.8, callbacks=[early_stopping], verbose=1)
#model.fit(d2d2v_train, np.array(df['toxic']), batch_size=1024, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
#model.save('./model/deepbaseline4_y1.hdf5')
#model.save('./model/deepbaseline4_y2.hdf5')
#model.save('./model/deepbaseline4_y3.hdf5')
#model.save('./model/deepbaseline4_y4.hdf5')
#model.save('./model/deepbaseline4_y5.hdf5')
model.save('./model/deepbaseline4_y6.hdf5')

score = model.evaluate(d2v_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict_proba(X_train, verbose=0)
print("AUC: %f" %(roc_auc_score(y_train, predictions)))
###############################################################################################

d2vmodel = Doc2Vec.load("./d2v/d2v_all_text.model")
#corpus_test = [data.split() for data in texts_test]
corpus_test = [gensim.utils.simple_preprocess(txt) for txt in df_test["comment_text"]]
d2v_test = np.array([d2vmodel.infer_vector(words) for words in corpus_test])


model1 = keras.models.load_model('./model/deepbaseline4_y1.hdf5')
predictions1 = model1.predict_proba(d2v_test, verbose=0)

model2 = keras.models.load_model('./model/deepbaseline4_y2.hdf5')
predictions2 = model1.predict_proba(d2v_test, verbose=0)

model3 = keras.models.load_model('./model/deepbaseline4_y3.hdf5')
predictions3 = model1.predict_proba(d2v_test, verbose=0)

model4 = keras.models.load_model('./model/deepbaseline4_y4.hdf5')
predictions4 = model1.predict_proba(d2v_test, verbose=0)

model5 = keras.models.load_model('./model/deepbaseline4_y5.hdf5')
predictions5 = model1.predict_proba(d2v_test, verbose=0)

model6 = keras.models.load_model('./model/deepbaseline4_y6.hdf5')
predictions6 = model1.predict_proba(d2v_test, verbose=0)

#predictions1

predictions = np.c_[predictions1,predictions2,predictions3,predictions4,predictions5,predictions6]
predictions.shape

columns_names=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
df = pd.DataFrame(predictions, columns=columns_names)
df.index = df_test['id']

df.to_csv('./result/result_deepbaseline4.csv')



##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['toxic']),
                                                    stratify=np.array(df_train['toxic']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y1.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################

##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['severe_toxic']),
                                                    stratify=np.array(df_train['severe_toxic']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y2.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################






##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['obscene']),
                                                    stratify=np.array(df_train['obscene']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y3.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################
##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['threat']),
                                                    stratify=np.array(df_train['threat']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y4.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################
##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['insult']),
                                                    stratify=np.array(df_train['insult']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y5.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################

##############################################################################################

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2v_train,
                                                    np.array(df_train['identity_hate']),
                                                    stratify=np.array(df_train['identity_hate']),
                                                    test_size=0.2,
                                                    random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

parameters = {'C': [10**(i) for i in range(0,4)]}
#parameters = {'kernel': ['rbf'], 'C': [10**i for i in range(4)], 'gamma': [10**(-i) for i in range(2,4)]}
scorer = make_scorer(f1_score)
clf = LogisticRegression()
#clf = SVC()

grid_obj = GridSearchCV(clf, parameters, scorer, cv=3)
grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_
pickle.dump(best_clf, open('./model/logisticreg_model_y6.sav', 'wb'))
#pickle.dump(best_clf, open('./model/svc_model.sav', 'wb'))

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

predictions_prob = (clf.fit(X_train, y_train)).predict_proba(X_test)[:,1]
best_predictions_prob = best_clf.predict_proba(X_test)[:,1]

print("F1score: %f" %f1_score(y_test, predictions))
print("best_F1score: %f" %f1_score(y_test, best_predictions))
print("AUC: %f" %roc_auc_score(y_test, predictions_prob))
print("best_AUC: %f" %roc_auc_score(y_test, best_predictions_prob))
##############################################################################################


d2vmodel = Doc2Vec.load("./d2v/d2v_all_text.model")
corpus_test = [data.split() for data in texts_test]

d2v_test = np.array([d2vmodel.infer_vector(words) for words in corpus_test])

model1 = pickle.load(open('./model/logisticreg_model_y1.sav','rb'))
predictions1 = model1.predict_proba(d2v_test)[:,1]

model2 = pickle.load(open('./model/logisticreg_model_y2.sav', 'rb'))
predictions2 = model2.predict_proba(d2v_test)[:,1]

model3 = pickle.load(open('./model/logisticreg_model_y3.sav', 'rb'))
predictions3 = model3.predict_proba(d2v_test)[:,1]

model4 = pickle.load(open('./model/logisticreg_model_y4.sav', 'rb'))
predictions4 = model4.predict_proba(d2v_test)[:,1]

model5 = pickle.load(open('./model/logisticreg_model_y5.sav', 'rb'))
predictions5 = model5.predict_proba(d2v_test)[:,1]

model6 = pickle.load(open('./model/logisticreg_model_y6.sav', 'rb'))
predictions6 = model6.predict_proba(d2v_test)[:,1]

predictions1.shape
predictions1.shape
predictions2.shape
predictions3.shape
predictions4.shape

predictions = np.c_[predictions1,predictions2,predictions3,predictions4,predictions5,predictions6]
predictions.shape
# np.array([predictions1,predictions2,predictions3,predictions4,predictions5,predictions6])


columns_names=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
df = pd.DataFrame(predictions, columns=columns_names)
df.index = df_test['id']

df.to_csv('./result/result_logisicreg.csv')
