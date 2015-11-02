__author__ = 'husser'


# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int", default=250,
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                             min_df=0.02, stop_words='english', token_pattern=r'\w*[A-Za-z]\w*')
X = vectorizer.fit_transform(dataset.data)
word_dict = vectorizer.get_feature_names()
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

print("Performing dimensionality reduction using LSA")
t0 = time()
svd = TruncatedSVD(opts.n_components)
lsa = make_pipeline(svd, Normalizer(copy=False))
Y = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))
print()


###############################################################################
# Do the actual clustering
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(Y)
print("done in %0.3fs" % (time() - t0))
print()


def get_tokens_per_topic(topic, terms, n_tokens):
    """Return the top features as extracted from the
    simple LSA.
    :param topics: array, [n_tokens]
    """
    top_features = topic.argsort()[::-1]
    return [terms[ind] for ind in top_features[:n_tokens]]


print("Top 5-term topics per cluster:")
km_centroids = km.cluster_centers_
terms = vectorizer.get_feature_names()


def get_features_in_classifier(classfier_data, nodes, topics_data, terms):
    sorted_centroids = classfier_data.argsort()[:, ::-1]
    for i in range(nodes):
        print("Cluster %d:" % i)
        top_topics = sorted_centroids[i, :4]
        for topic_id in top_topics:
            tokens = get_tokens_per_topic(topics_data[topic_id, :], terms, 15)
            print('Topic:%d:\t' % (topic_id) + ','.join(tokens))
        print()


get_features_in_classifier(km_centroids, true_k, Y, terms)
###############################################################################
# Perform SOM clustering
from som.som import SOMMapper

som = SOMMapper(kshape=(5, 5), n_iter=300, learning_rate=0.005)
kohonen = som.fit_transform(Y)
get_features_in_classifier(kohonen, som.n_nodes, Y, terms)
