from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import sys
import re
import pickle
import pandas as pd


def insert_row(df, row_number, row_value):
    if row_number > df.index.max() + 1:
        print("Invalid Row Number")
    else:
        # Slice the upper half of the dataframe
        df1 = df[0:row_number]

        # Store the result of lower half of the dataframe
        df2 = df[row_number:]

        # Inset the row in the upper half dataframe
        df1.loc[row_number] = row_value

        # Concat the two dataframes
        df_result = pd.concat([df1, df2])

        # Reassign the index labels
        df_result.index = [*range(df_result.shape[0])]

        return df_result


def data_processing(df):
    df = df.astype(str)
    # drop unmeaningful columns
    unused_columns = ['Style.Code','Style.Theme.NodeName','Style.Node_URL_ID', 'ProductSizes.refvector.URLs', 'Images.Viewable_VaultFullPath', 
                    'Images.Viewable', 'Images.NodeURL','Images.Node Name']
    df.drop(unused_columns, axis=1, inplace=True)
    df = df.astype(str)

    df.set_index('Style.NodeURL', inplace=True)

    return df


def bag_of_words(df):

    df['bag_of_words'] = ''
    columns = df.columns
    for index, row in df.iterrows():
        words = ''
        for col in columns:
            if col != 'size':
                words = words + ''.join(row[col])+ ' '
            else:
                words = words + row[col]+ ' '
        row['bag_of_words'] = words
        
    df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)

    return df


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def calculate_hue(df):
    def hex2hue(hex, node_name):
        if hex != '' and bool(hex):
            color = Color(hex)
            hsv = color.hsv
            return hsv[0]
        else:
            return -1

    df['color.hue'] = df.apply(lambda x: hex2hue(x['ColorSpecification.RGBHex'], x['Colorway.ColorSpecification.NodeName']), axis=1)


def calculate_compositions(df):
    def composition2tuple(composition):
        if bool(composition):
            batRegex = re.compile(r'<blockquote[^>].*>([^<].*)</blockquote')
            values = batRegex.findall(composition)
            if len(values) == 0:
                return []
            batRegex = re.compile(r'(\d+)% (\w+)')
            return batRegex.findall(values[0])
        else:
            return []

    df['composition.values'] = df.apply(lambda x: composition2tuple(x['StyleAttributes.PTO_StyleAttr_Fabric_Composition_String']), axis=1)

    unique_composition_values = set()
    for compositions in df['composition.values']:
        for item in compositions:
            if type(item) is tuple:
                unique_composition_values.add(item[1].lower())

    def composition2bin(unique_comp_values, comps):
        if len(comps) == 0:
            return [10] * len(unique_comp_values)

        bin = [0] * len(unique_comp_values)
        unique_comp_values = list(unique_comp_values)
        try:
            for item in comps:
                if type(item) is tuple:
                    idx = unique_comp_values.index(item[1].lower())
                    bin[idx] = int(item[0]) / 100  # Convert to percent.

        except ValueError:
            print(item)
            raise

        return bin

    df['composition.bin'] = df.apply(lambda x: composition2bin(unique_composition_values, x['composition.values']), axis=1)

def clean_up_string(value):
    if not value:
        return []

    value = value.lower()
    value = value.replace('(', '')
    value = value.replace(')', '')
    value = value.replace('-', '')
    value = value.replace(':', '')
    value = value.replace(',', '')
    value = value.replace('/', '')
    value = value.replace('"', '')
    value = value.replace('"', '')

    result = []
    stop_words = [',', '', None, '&', 'in', 'y', 'with', 'con', 'the', 'a']
    if value and " " in value:
        for word in value.split(" "):
            if word not in stop_words:
                result.append(word)
    else:
        if value not in stop_words:
            result.append(value)

    return result


def calculate_shape_descriptions(df):
    unique_shape_descriptions = df['Shape.Description'].unique()

    unique_description_values = set()
    for description in unique_shape_descriptions:
        for word in clean_up_string(description):
            unique_description_values.add(word)

    def description2bin(unique_values, description):
        if not description or len(description) == 0:
            return [10] * len(unique_values)

        bin = [0] * len(unique_values)
        unique_values = list(unique_values)
        for word in clean_up_string(description):
            idx = unique_values.index(word)
            bin[idx] = 1

        return bin

    df['shape.description.bin'] = df.apply(lambda x: description2bin(unique_description_values, x['Shape.Description']), axis=1)


def recommendations(bow_traits_df, indices, cosine_sim):

    recommended_breeds = []
    res = {}

    # getting the index of the breed that matches the breed_name
    idx = indices[indices == 'nan'].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar breeds
    top_5_indexes = list(score_series.iloc[1:6].index)
    top_5_score = list(score_series.iloc[1:6])

    # populating the list with the breeds of the best 10 matching breeds
    for i in top_5_indexes:
        recommended_breeds.append(bow_traits_df.index[i])
    
    for key in recommended_breeds: 
        for value in top_5_score: 
            res[key] = value 
            top_5_score.remove(value) 
            break  

    return res


class ReadCSV(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.read_file()

    def read_file(self):
        """Read the file concent"""

        try:
            self.df = pd.read_csv(self.file_path)
        except IndexError:
            print("Error: Wrong file name")
            sys.exit(2)
        return self.df

    def display_file(self):
        print(self.df)


class Vectorizer(object):

    def __init__(self):
        """
        Simple NLP Vectorizer
        Attributes:
            vectorizor: CountVectorizer or similar
        """
        self.vectorizer = CountVectorizer(stop_words='english')

    def vectorizer_fit(self, x):
        """
        Fits a CountVectorizer to the text
        """
        self.vectorizer.fit(x)

    def vectorizer_transform(self, x):
        """
        Transform the text data to a sparse Count matrix
        """
        x_transformed = self.vectorizer.transform(x)
        return x_transformed

    def vectorizer_fit_transform(self, x):
        """
        Fits a CountVectorizer to the text
        """
        x_fit_transformed = self.vectorizer.fit_transform(x)
        return x_fit_transformed

    def pickle_vectorizer(self, path='lib/models/count_vectorizer.pkl'):
        """
        Saves the count vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))
