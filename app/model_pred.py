from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, jsonify, request
# from flask_cors import CORS, cross_origin
import pickle
import pandas as pd


def read_csv(file_path):
    '''
    Read the traits.csv file.
    '''

    traits = pd.read_csv(file_path)

    return traits


def data_processing():
    file_path = '../data/sync_service_full.csv'

    syncdata = read_csv(file_path)
    # drop unmeaningful columns
    unused_columns = ['Style.Code','Style.Theme.NodeName','Style.Node_URL_ID', 'ProductSizes.refvector.URLs', 'Images.Viewable_VaultFullPath', 
                    'Images.Viewable', 'Images.NodeURL','Images.Node Name']
    syncdata.drop(unused_columns, axis=1, inplace=True)
    syncdata = syncdata.astype(str)

    syncdata.set_index('Style.NodeURL', inplace=True)

    return syncdata


def bag_of_words():
    syncdata = data_processing()

    syncdata['bag_of_words'] = ''
    columns = syncdata.columns
    for index, row in syncdata.iterrows():
        words = ''
        for col in columns:
            if col != 'Size':
                words = words + ''.join(row[col])+ ' '
            else:
                words = words + row[col]+ ' '
        row['bag_of_words'] = words

    syncdata.drop(columns=[col for col in syncdata.columns if col != 'bag_of_words'],
                                             inplace=True)

    return syncdata


def find_similarity():
    syncdata = bag_of_words()

    # instantiating and generating the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(syncdata['bag_of_words'])

    # creating a Series for the dog breed so they are associated to an ordered numerical
    # list I will use later to match the indexes
    indices = pd.Series(syncdata.index)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    return indices, cosine_sim


def recommendations(breed):

    indices, cosine_sim = find_similarity()

    recommended_breeds = []

    # getting the index of the breed that matches the breed name
    idx = indices[indices == breed].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar breeds
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the breeds of the best 10 matching breeds
    syncdata = bag_of_words()

    for i in top_10_indexes:
        recommended_breeds.append(list(syncdata.index)[i])

    return recommended_breeds


if __name__ == '__main__':
    from pprint import pprint
    print('Input Product:')
    chat_in = 'C5296872'
    pprint(chat_in)

    products = recommendations(chat_in)
    print(f'Input Values: {chat_in}')
    print('Output Recommended Products')
    pprint(products)