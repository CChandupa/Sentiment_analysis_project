from flask import Flask, render_template, request, redirect
from PredictionPipeline import preprocessing, vectorizer, get_prediction
from logger import logging

app = Flask(__name__)

logging.info('Flask server started')

data = dict()
reviews = []
possitive = 0
negetive = 0

@app.route('/')
def index():
    data['reviews'] = reviews
    data['possitive'] = possitive
    data['negetive'] = negetive
    
    logging.info('************** Open home page **************')
    
    return render_template('index.html', data=data)
   

@app.route("/", methods=['POST'])
def my_post():
    global possitive, negetive 
    text = request.form['text']
    
    logging.info(f'Text : {text}')
    
    preprocessed_text = preprocessing(text)
    logging.info(f'Vectorized_Text : {preprocessed_text}')
    
    vectorized_text = vectorizer(preprocessed_text)
    logging.info(f'Vectorized_Text : {vectorized_text}')
    
    prediction = get_prediction(vectorized_text)
    logging.info(f'Prediction : {prediction}')
    
    if prediction == 'negetive':
        negetive += 1
    else:
        possitive += 1
        
    reviews.insert(0, text)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)