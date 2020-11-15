#%%
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import xgboost
from sklearn import svm
from tqdm import tqdm
import pickle
import os
import pprint

from collections import defaultdict
from collections import Counter



class PointWiseLTRModel(object):
    def __init__(self, regressor):
        """
        Arguments:
            classifier: An instance of scikit-learn regressor.
        """
        self.model = regressor

    def train(self, X, y):
        """Trains an LTR model.

        Arguments:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.model is not None
        self.model = self.model.fit(X, y)

    def predict(self, features_list):
        return self.model.predict(np.array(features_list))

    def rank(self, ft, doc_ids):
        """Predicts relevance labels and rank documents for a given query.

        Arguments:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(np.array(ft))
        sort_indices = np.argsort(rel_labels)[::-1]
        results = []
        for i in sort_indices:
            results.append(doc_ids[i])
        return results


def get_reciprocal_rank(doc_rankings, relevant_doc_id):
    for i, doc_id in enumerate(doc_rankings):
        if doc_id == relevant_doc_id:
            return 1 / (i + 1)
    return 0


def get_mean_eval_measure(system_rankings, eval_function):
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, QRELS[query_id])
    return sum_score / len(system_rankings)


def rerank_score(basic_rankings, ltr_model):
    reranked = {}
    for query_id, doc_rankings_features in tqdm(basic_rankings.items(), desc='Reranking'):
        doc_rankings = [x[0] for x in doc_rankings_features]
        features = [x[1] for x in doc_rankings_features]

        doc_reranked = ltr_model.rank(features, doc_rankings)
        reranked[query_id] = doc_reranked

    score = get_mean_eval_measure(reranked, get_reciprocal_rank)
    return score


with open('pickles/basic_rankings.pkl', 'rb') as file:
    basic_rankings = pickle.load(file)


with open('pickles/training_data.pkl', 'rb') as file:
    X_train, y_train = pickle.load(file)

with open('pickles/QUERIES.pkl', 'rb') as file:
    QUERIES = pickle.load(file)

with open('pickles/QRELS.pkl', 'rb') as file:
    QRELS = pickle.load(file)

with open('pickles/advanced_rankings.pkl', 'rb') as file:
    advanced_rankings = pickle.load(file)

with open('pickles\corpus_embeddings.pkl', 'rb') as file:
    corpus_embeddings_dict = pickle.load(file)

with open('pickles\query_embeddings.pkl', 'rb') as file:
    query_embeddings_dict = pickle.load(file)





system_rankings = {}
for query_id, doc_id_features in basic_rankings.items():
    doc_ids = [x[0] for x in doc_id_features]
    system_rankings[query_id] = doc_ids

print('Base score: ', get_mean_eval_measure(system_rankings, get_reciprocal_rank))

clf = RandomForestRegressor(max_depth=7, random_state=0, n_jobs=4, n_estimators=19)
ltr = PointWiseLTRModel(clf)
ltr.train(np.array(X_train), np.array(y_train))
print('Random Forest Score:', rerank_score(basic_rankings, ltr))


clf = xgboost.XGBRegressor(max_depth=3, random_state=0, booster='gbtree',
                           objective='reg:linear', verbosity=0, n_estimators=11)
ltr = PointWiseLTRModel(clf)
ltr.train(np.array(X_train), np.array(y_train))
print('XGBoost Score:', rerank_score(basic_rankings, ltr))

clf = LinearRegression(normalize=True)
ltr = PointWiseLTRModel(clf)
ltr.train(np.array(X_train), np.array(y_train))
print('Linear Score:', rerank_score(basic_rankings, ltr))

#%%
##--- BERRT Functionality implemented
reranked_docs = defaultdict(dict)
with open('pickles/ADV_QRELS.pkl', 'rb') as file:
    QRELS = pickle.load(file)

for query_id, query_embedding in tqdm(query_embeddings_dict.items()):
    doc_ids = list(advanced_rankings[query_id].keys())
    if not QRELS[query_id] in doc_ids:
        continue
    doc_embeddings =  [corpus_embeddings_dict[doc_id] for doc_id in doc_ids ]
    distances= cosine_similarity([query_embedding], doc_embeddings)[0]
    results = zip(doc_ids, distances)
    results = sorted(results, key=lambda x: x[1],reverse=True)
    for doc_id,score in results:
        reranked_docs[query_id][doc_id]=[doc_ids.index(doc_id)+1,score]
adv_score = get_mean_eval_measure(reranked_docs, get_reciprocal_rank)
print('Advanced Model Score:', adv_score )



# %%
