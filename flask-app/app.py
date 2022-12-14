# from crypt import methods
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
 
app = Flask(__name__)
html_name = 'index_init_style.html'

@app.route('/')
def index():
    return render_template(html_name)

# @app.route('/predict')
# def predict():
#     return render_template('result.html')

# def valuePredictor(to_predict_list, model_ensemble, model_qda, model_rf):
#     size = len(to_predict_list)
#     to_predict = np.array(to_predict_list).reshape(1,size)
#     preds = model_ensemble.predict(to_predict)
#     pred = preds[0]
#     # 0 : operating, 1 : acquired, 2 : closed, 3 : ipo
#     if pred == 0:
#         result = 'operating'
#     elif pred == 1:
#         result = 'acquired'
#     elif pred == 2:
#         result = 'closed'
#     elif pred == 3:
#         result = 'ipo'
    # return result


def valuePredictor(to_predict_list, model_ensemble, model_qda, model_rf):
    size = len(to_predict_list)
    to_predict = np.array(to_predict_list).reshape(1,size)

    preds = model_qda.predict(to_predict)
    # 0 : operating, 1 : closed
    pred = preds[0]
    if pred == 0:
        result = 'operating'
    elif pred == 1:
        preds = model_rf.predict(to_predict)
    # 0 : operating, 1 : acquired, 2 : closed, 3 : ipo
        pred = preds[0]
        if pred == 0:
            result = 'operating'
        elif pred == 1:
            result = 'acquired'
        elif pred == 2:
            result = 'closed'
        elif pred == 3:
            result = 'ipo'

    return result


# 1800 25 30 1803 2012 1000000 1805 2019 50 12 300 212
# 'founded_at', 'investment_rounds', 'first_funding_at', 'last_funding_at', 'funding_rounds', 'funding_total_usd', 'first_milestone_at', 'last_milestone_at', 'milestones', 'relationships'

@app.route('/', methods=['POST'])
# @app.route('/result', methods=['POST'])
def result():
    model_ensemble = pickle.load(open('./models/ensemble.pkl', 'rb'))
    model_qda = pickle.load(open('./models/qda.pkl', 'rb'))
    model_rf = pickle.load(open('./models/rf.pkl', 'rb'))

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = valuePredictor(to_predict_list, model_ensemble, model_qda, model_rf)
        pred = str(result)
        return render_template(html_name, prediction=pred)




if __name__ == '__main__':
    app.run(debug=True)