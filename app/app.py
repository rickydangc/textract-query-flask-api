import pickle
import pandas as pd

from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from sklearn.metrics.pairwise import cosine_similarity

from model_build import Vectorizer, ReadCSV
from model_build import insert_row, data_processing, bag_of_words, recommendations, remove_special_characters


model = Vectorizer()
app = Flask(__name__)
api = Api(app)


pd.set_option('mode.chained_assignment', None)
csv = ReadCSV('../data/sync_service_full.csv')
traits_data = csv.read_file()

vec_path = '../models/count_vectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)


# argument parsing
# parser = reqparse.RequestParser()
# parser.add_argument('query')


class PredictBreed(Resource):
    def post(self):
        # use parser and find the user's query
        # args = parser.parse_args()
        # user_query = args['query']
        try:
            json_data = request.get_json(force=True)
            finish_insert_traits_data = traits_data.append(json_data, ignore_index=True)
            new_traits_df = data_processing(finish_insert_traits_data)
            bow_traits_df = bag_of_words(new_traits_df)
            bow_traits_df['bag_of_words'] = bow_traits_df['bag_of_words'].apply(remove_special_characters)
            bow_traits_df['bag_of_words'] = bow_traits_df['bag_of_words'].str.strip()

            # instantiating and generating the count matrix
            count_matrix = model.vectorizer_fit_transform(bow_traits_df['bag_of_words'])
            indices = pd.Series(bow_traits_df.index)
            cosine_sim = cosine_similarity(count_matrix, count_matrix)
            pred_breed = recommendations(bow_traits_df, indices, cosine_sim)

            # create JSON object
            # output = {pred_breed}

            return jsonify({'prediction': str(pred_breed)})

        except Exception as error:
            raise error

        # for i in json_data.keys():
        #    traits_list.append(json_data[i])


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictBreed, '/predict/')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
