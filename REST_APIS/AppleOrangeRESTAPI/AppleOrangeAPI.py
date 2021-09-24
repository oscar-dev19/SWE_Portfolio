import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle
import traceback
import json

app = Flask(__name__)

def load_model(modelname):
    model = pickle.load(open(modelname,'rb'))
    return model


model = load_model('apple_orangeclassifier.mdl')

@app.route('/')
def hello():
    return "Welcome to my apple/orange classifier model API"


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query_df = pd.DataFrame(json_)
            query = pd.get_dummies(query_df)
            #query = query.reindex(columns=model_columns, fill_value=0)

            prediction = model.predict(query)            

            fruit = None
            if prediction == [1]:
                fruit = 'Orange'
            if prediction == [0]:
                fruit =  'Apple'
            results = {
                'Predicted fruit':fruit
            }
            return jsonify(str(results))
        except:
            #TODO:add error logging.
            return jsonify({'trace': traceback.format_exc()})
    else:
        return('No Model Present to run prediction.')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    try:    
        model = load_model('apple_orangeclassifier.mdl')
        print('Model Loaded')
    except:
        #TODO:add error logging.
        print('ERROR: Model did not load successfully!')

    
    #running the app on debug mode.
    app.run(port=port, debug=True)

