import pandas as pd
from flask import Flask, json, request, Response
import _pickle as cPickle
from google.cloud import storage
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import make_column_transformer
from io import BytesIO
import datetime


#from resources import predictor

app = Flask(__name__)
app.config["DEBUG"] = True

def preprocess(new_cars): 

    #read columns created in the training dataframe that the model expects
    df_cols=pd.read_csv('gs://de-3/data/train_columns.csv')
    df_cols = df_cols['train_df_columns'].tolist()
    #remove the price column which is last
    df_cols.pop()
    

    conditions = {'salvage': 0, 'fair': 1, 'good': 2, 'excellent': 3, 'like new': 4, 'new': 5}
    transmissions = {'manual': 0, 'automatic': 1}

    now = datetime.datetime.now()

    for car in new_cars:
        if car['condition'] not in list(conditions.keys()):
            problematic_car = car.copy()
            problematic_car['message']='error'
            problematic_car['error']= 'car condition must be one of: salvage, fair, good, excellent, like new, new'
            return problematic_car
        if car['transmission'] not in list(transmissions.keys()):
            problematic_car = car.copy()
            problematic_car['message']='error'
            problematic_car['error']= 'car transmission must be one of: automatic, manual'
            return problematic_car
        if car['year'] > now.year:
            problematic_car = car.copy()
            problematic_car['message']='error'
            problematic_car['error']= 'car year cannot be more than '+ str(now.year)
            return problematic_car

    encoded_df = pd.DataFrame(columns=df_cols)

    for car in new_cars:
        new_row_dict = {}
        for col in df_cols:
            if 'manufacturer' in col:
                if car['manufacturer'] in col:
                    new_row_dict[col]=1
                else: 
                    new_row_dict[col]=0
            if 'fuel' in col:
                if car['fuel'] in col:
                    new_row_dict[col]=1
                else: 
                    new_row_dict[col]=0                
            elif 'drive' in col:
                if car['drive'] in col:
                    new_row_dict[col]=1
                else: 
                    new_row_dict[col]=0        
            elif 'type' in col:
                if car['type'] in col:
                    new_row_dict[col]=1
                else: 
                    new_row_dict[col]=0        
            elif col=='condition':
                new_row_dict['condition']= conditions[car['condition']]        
            elif col=='transmission':
                new_row_dict['transmission']= transmissions[car['transmission']]
            elif col=='year': 
                new_row_dict[col]= car[col]
            elif col=='odometer': 
                new_row_dict[col]= car[col]
        #print (new_row_dict)
        encoded_df = encoded_df.append(new_row_dict, ignore_index=True)

        #encoded_df.info()
    encoded_df = encoded_df.astype('float64')
    return {'message': 'success', 'df': encoded_df}


@app.route('/price-predict', methods=['POST'])
def predict_perf():
    content = request.get_json()
    #print(content)

    #test_df = pd.read_json(json.dumps(content), orient='records')

    js_str_ = json.dumps(content)
    dicts = json.loads(js_str_)
    #print(type(dict_[0]['sepal_length']))

    try:
        #read the logs file to add new results
        logs_df = pd.read_csv('gs://de-3/logs/logs.csv')
    except:
        #if it doesnt exist yet, create it
        logs_df = pd.DataFrame(columns=['message', 'error', 'price'].extend(list(dicts[0].keys())))
    
    result = preprocess(dicts)
    
    
    if result['message'] == 'success':
        prep_cars = result['df']

        #find the path of the best model aka smallest mean absolute error
        scores_df = pd.read_csv('gs://de-3/models/scores.csv')
        best_model_path = scores_df['model'][scores_df['score'].idxmin()]

        client = storage.Client()
        bucket = client.get_bucket('de-3')
        blob = bucket.get_blob(best_model_path)
        if blob is None:
            raise AttributeError('No files to download') 
        model_bytestream = BytesIO(blob.download_as_string())
        model = cPickle.load(model_bytestream)

        #result_df= prep_cars.copy()

        pred_result = model.predict(prep_cars)
        print(pred_result)

        for i, car in enumerate(dicts):
            car['price']=str(pred_result[i])
            log = car.copy()
            log['message'] = 'success'
            log['error']= ''
            logs_df = logs_df.append(log, ignore_index=True)

        
        logs_df.to_csv('gs://de-3/logs/logs.csv', index=False)
        js_result=json.dumps(dicts, indent=4, sort_keys=False)


        resp = Response(js_result, status=200, mimetype='application/json')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST'
        resp.headers['Access-Control-Max-Age'] = '1000'
        return resp
    else:
        #logs_df = logs_df.append({'result': 'error', 'message': result['error'] }, ignore_index=True)
        logs_df = logs_df.append(result , ignore_index=True)
        logs_df.to_csv('gs://de-3/logs/logs.csv', index=False)
        
        js_result=json.dumps(result, indent=4, sort_keys=False)

        resp = Response(js_result, status=200, mimetype='application/json')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST'
        resp.headers['Access-Control-Max-Age'] = '1000'
        return resp
    


app.run(host='0.0.0.0', port=5000)
