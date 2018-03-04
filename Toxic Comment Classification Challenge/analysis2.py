import pandas as pd
import string
import re
import os

################################################################################################
## 前処理 ##

################################################################################################
## データ読み込み ##
datapath = './data/train.csv'
df_train = pd.read_csv(datapath, sep=',')
df_train.head()
#len(df_train)



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

        #texts.append(re.findall(r"[\w]+", text))  # 単語に分割
        texts.append(text)

    return texts


words_train = data2words(df_train)
#txts_train[:100]

#txts_train = text_cleaning(df_train)

############################################################################################### word2vec ##

from gensim.models import Word2Vec
num_features = 200     # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = os.cpu_count()   # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print ("Training Word2Vec model...")
# Train Word2Vec model.
model = Word2Vec(words_train, workers=num_workers, hs = 0, sg = 1, negative = 10, iter = 25,\
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, seed=1)

print ("Saving Word2Vec model...")
model.save("./w2v/w2v_train.model")


################################################################################################

## create gwbowv
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def cluster_GMM(num_clusters, word_vectors):
    # Initalize a GMM object and use it for clustering.
    clf =  GaussianMixture(n_components=num_clusters,
                    covariance_type="tied", init_params='kmeans', max_iter=50)
    # Get cluster assignments.
    clf.fit(word_vectors)
    idx = clf.predict(word_vectors)
    print ("Clustering Done...")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments.
    pickle.dump(idx, open('./clustering/gmm_clustermodel.pkl',"wb"))
    print ("Cluster Assignments Saved...")

    pickle.dump(idx_proba,open('./clustering/gmm_prob_clustermodel.pkl',"wb"))
    print ("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    pickle.dump(idx, open('./clustering/gmm_clustermodel.pkl',"wb"))
    idx = pickle.load(open('./clustering/gmm_clustermodel.pkl',"rb"))
    idx_proba = pickle.load(open( './clustering/gmm_prob_clustermodel.pkl',"rb"))
    print ("Cluster Model Loaded...")
    return (idx, idx_proba)

def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
    # This function computes probability word-cluster vectors
    prob_wordvecs = {}
    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros( num_clusters * num_features, dtype="float32" )
        for index in range(0, num_clusters):
            try:
                prob_wordvecs[word][index*num_features:(index+1)*num_features] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
            except:
                continue

    # prob_wordvecs_idf_len2alldata = {}
    # i = 0
    # for word in featurenames:
    #     i += 1
    #     if word in word_centroid_map:
    #         prob_wordvecs_idf_len2alldata[word] = {}
    #         for index in range(0, num_clusters):
    #                 prob_wordvecs_idf_len2alldata[word][index] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]



    # for word in prob_wordvecs_idf_len2alldata.keys():
    #     prob_wordvecs[word] = prob_wordvecs_idf_len2alldata[word][0]
    #     for index in prob_wordvecs_idf_len2alldata[word].keys():
    #         if index==0:
    #             continue
    #         prob_wordvecs[word] = np.concatenate((prob_wordvecs[word], prob_wordvecs_idf_len2alldata[word][index]))
    return prob_wordvecs

def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension, word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros( num_centroids * dimension, dtype="float32" )
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = word_centroid_map[word]
        except:
            continue

        bag_of_centroids += prob_wordvecs[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if(norm!=0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


num_features = 200     # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = os.cpu_count   # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context_len2alldata"
# Load the trained Word2Vec model.
model = Word2Vec.load("./w2v/w2v_train.model")
# Get wordvectors for all words in vocabulary.
word_vectors = model.wv.syn0

# Load train data.
train,test = train_test_split(df,test_size=0.3,random_state=40)
all = df

# Set number of clusters.
num_clusters = 60
idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))
# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
# list of probabilities of cluster assignments.
word_centroid_prob_map = dict(zip( model.wv.index2word, idx_proba ))




################################################################################################
## データ分割 ##

#columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(txts_train,
                                                    df_train['toxic'],
                                                    stratify=df_train['toxic'],
                                                    test_size=0.2,
                                                    random_state=1)

#len(X_train)
#len(X_test)

import gc
del df_train
del txts_train
gc.collect()
