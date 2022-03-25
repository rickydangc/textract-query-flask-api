import pandas as pd
import numpy as np

from flask import Flask, jsonify, request
from flask_restful import API, Resource

app = Flask(__name__)
api = API(app)

model_path = '../models/service-2.json'


class TextractQuery(Resource):
    
    def post(self):
        try:
            pass

        except Exception as error:
            raise error


# Setup the API resource routing here
# Route the URL to the resource
api.add_resource(GetAnswer, '/results/')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
