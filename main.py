# importing libraries
from flask import Flask, jsonify, request
import pandas
import requests
import json
from dotenv.main import load_dotenv
import os

load_dotenv()
# authorisation with bert api
API_URL = os.environ['API_URL']
headers = os.environ['headers']


# function to get ner details of given text
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()
    st = []
    for dt in data:
        print(dt['entity_group'], dt['word'], dt['start'], dt['end'])
        st.append({"Entity_Group": dt['entity_group'], "Word in sentence": dt['word']})
    print(st)
    df = pandas.DataFrame(data)
    print(df)
    return st


app = Flask(__name__)
text_data = [

]


# home route
@app.route('/')
def home():
    return "bert ner"


# post a text and respective ner returned
@app.route('/bert/new', methods=['POST'])
def create_ner():
    request_data = request.get_json()
    new_data = {
        'input': request_data['input'],
        'output': query(request_data['input'])
    }

    text_data.append(new_data)
    return jsonify(new_data)


# get all text results
@app.route('/bert')
def get_ner():
    return jsonify({'text_data': text_data})


app.run(port=8000)
