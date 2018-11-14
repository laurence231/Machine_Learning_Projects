import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv("my_protein_ascii.csv")
#print(ds)

    #here is the input table for analysis

    # This is where the training takes place.
    #      Unigrams, bigrams and trigrams (single, double and triple word combinations) form a TF-IDF matrix
    #      for each individual product description.
    #
    #      stop_words is used to tell the training algorithm to ignore the most common english words such as 'the' etc
    #
    #      SciKit Learn linear_kernel is identical to cosine similarity technique, which effectively determines the cosine of
    #      the angle between the vectors that represent each uni/bi/trigram for each product.
    #
    #      it iterates through and produces each items most similar items. Similarities are stored in a redis database with scores
    #      of similarity.
    #



tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')    #analyse WORDS, with uni,bi,trigrams, ignore 'the' etc
tfidf_matrix = tf.fit_transform(ds['synopsis'])                                           #Learn vocabulary and idf, return term-document matrix

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)                 #this applies to cosine similarities method to the vectors defined above

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]                   #this applies the method for each id in the csv file
    similar_items = [(cosine_similarities[idx][i], ds['title'][i]) for i in similar_indices]          #this produces list of most similar items

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results[row['title']] = similar_items[1:]

print('done!')
#print(results)

def item(id):
    return ds.loc[ds['title'] == id]['synopsis'].tolist()[0].split(' - ')[0]            #this function returns the synopsis for the input ID!




def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item_id + "..." )
    print("-------")
    recs = results[item_id][:num]
    print(recs)
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
        print(" ")
        print(" ")

# Just plug in any item id here (1-500), and the number of recommendations you want (1-99)
# You can get a list of valid item IDs by evaluating the variable 'ds', or a few are listed below

recommend(item_id=ds['title'][100], num=5)












# import pandas as pd  #python data analysis library
# import time
# import redis     #redis is for cache/data storage for the algorithm
# from flask import current_app    #micro web framework
# from sklearn.feature_extraction.text import TfidfVectorizer    #sklearn controls the machine learning/cosine similarity element, and creates vectors from the strings.
# from sklearn.metrics.pairwise import linear_kernel
#

# def info(msg):
#     current_app.logger.info(msg)
#
#
# class ContentEngine(object):
#     SIMKEY = 'p:smlr:%s'   #find out what this is?
#
#
#     def _train(self, ds):
#         """This is where the training takes place.
#         Unigrams, bigrams and trigrams (single, double and triple word combinations) form a TF-IDF matrix
#         for each individual product description.
#
#         stop_words is used to tell the training algorithm to ignore the most common english words such as 'the' etc
#
#         SciKit Learn linear_kernel is identical to cosine similarity technique, which effectively determines the cosine of
#         the angle between the vectors that represent each uni/bi/trigram for each product.
#
#         it iterates through and produces each items most similar items. Similarities are stored in a redis database with scores
#         of similarity.
#
#          :param ds: A pandas dataset containing two fields: description & id
#         """
#         ds = pd.read_csv(data_source)
#
#         tf = TfidfVectorizer(analyzer='word',                      #type of object to analyse
#                              ngram_range=(1, 3),                    #between 1 and 3 word strings, can be varied
#                              min_df=0,                              #When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
#                              stop_words='english')                   #ignore 'the' etc
#         tfidf_matrix = tf.fit_transform(ds['description'])    #Learn vocabulary and idf, return term-document matrix.
#
#         cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
#
#         results = {}
#
#         for idx, row in ds.iterrows():     #this takes the rows and id in the 'ds' input csv file of descriptions
#             similar_indices = cosine_similarities[idx].argsort()[:-100:-1]           #this selects the 100 most similar items to every input item
#             similar_items = [(cosine_similarities[idx][i], ds['id'][i])              #
#                              for i in similar_indices]
#
#             # First item is the item itself, so remove it.
#             # This 'sum' is turns a list of tuples into a single tuple:
#             # [(1,2), (3,4)] -> (1,2,3,4)
#             results[row['id']] = similar_items[1:]
#
#     def predict(self, item_id, num):
#         """
#         Couldn't be simpler! Just retrieves the similar items and
#         their 'score' from redis.
#
#         :param item_id: string
#         :param num: number of similar items to return
#         :return: A list of lists like: [["19", 0.2203],
#         ["494", 0.1693], ...]. The first item in each sub-list is
#         the item ID and the second is the similarity score. Sorted
#         by similarity score, descending.
#         """
#
#         return self._r.zrange(self.SIMKEY % item_id,
#                               0,
#                               num-1,
#                               withscores=True,
#                               desc=True)
#
#     # hacky little function to get a friendly item name from the description field, given an item ID
#     def item(id):
#         return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]
#
#     # Just reads the results out of the dictionary. No real logic here.
#     def recommend(item_id, num):
#         print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
#         print("-------")
#         recs = results[item_id][:num]
#         for rec in recs:
#             print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
#
#     # Just plug in any item id here (1-500), and the number of recommendations you want (1-99)
#     # You can get a list of valid item IDs by evaluating the variable 'ds', or a few are listed below
#
#     recommend(item_id=11, num=5)
#
# content_engine = ContentEngine()
#
#
























