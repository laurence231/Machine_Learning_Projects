import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re


def clean_data(search_term):
    with open('myp_full.csv', 'r', encoding='cp1251') as f, open('ecommerce_no_html.csv', 'w') as g:
        content = re.sub("<.*?>", "", f.read())
        g.write(content)

    with open('ecommerce_no_html.csv', 'a') as g:
        g.write(" \nsearch, %s"%search_term)

def train_TFIDF(ds):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['Description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]                   #this applies the method for each id in the csv file
        similar_items = [(cosine_similarities[idx][i], ds['Title'][i]) for i in similar_indices]          #this produces list of most similar items
        results[row['Title']] = similar_items[1:]
    return results


def item(id):
    return ds.loc[ds['Title'] == id]['Description'].tolist()[0].split(' - ')[0]            #this function returns the synopsis for the input ID!


def recommend(results, search_item,search_item_index, num):
    print("Recommending " + str(num) + " products similar to " + search_item)
    print("-------")
    recs = results[search_item_index][:num]
    print(recs)
    # for rec in recs:
    #     print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")" + " " + ds.loc[ds['Title'] == rec[1]]['Description'])
    #     print(" ")
    #     print(" ")

colnames=['Title','Description']

search_term = input('Enter the term you wish to search for here: ', )
clean_data(search_term)
ds = pd.read_csv("ecommerce_no_html.csv", names=colnames)
pd.options.display.max_colwidth = 10000
trained_algorithm = train_TFIDF(ds)
recommend(trained_algorithm, search_term,'search', num=5)