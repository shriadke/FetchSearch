# -*- coding: utf-8 -*-
"""
@purpose: This file generates an API service using Flask that can be used to compare texts for similarity.
          The API can be accessed with the web interface given in 'index.html'. Currently the html supports 3 texts.
          However, original 'textSimilarity.py' is capable of taking more than 3 texts as inputs.
          
@usage:   To run this file use "python3 app.py" This will run the text comparison UI on localhost. 
          Select appropriate values in the form to get the Similarity matrix.
          
Created on Wed Feb  3 19:15:37 2021

@author: shri
"""

from flask import Flask, render_template, request

from textSimilarity import *

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/getSimilarity", methods=['POST'])
def getSimilarity():
    
    if request.method == 'POST':
        texts = []
        
        texts.append(request.form['text1'])
        texts.append(request.form['text2'])
        texts.append(request.form['text3'])
        
        method = request.form['similarity_method']
        
        stopwords = False
        if 'stopwords' in  request.form:
            if request.form['stopwords'] == "yes":
                stopwords = True
        
        punct = False
        if 'punctuation' in  request.form:
            if request.form['punctuation'] == "yes":
                punct = True
            
        sim_matrix = get_text_similarity(texts, method, exclude_stopwords=stopwords, exclude_punctuation=punct)
        
        
        if sim_matrix is None:
            return render_template('index.html',prediction_texts="Sorry, Please enter valid input!")
        else:
            return render_template('index.html', text_1_2="{}".format(sim_matrix[0][1]), text_1_3="{}".format(sim_matrix[0][2]), text_2_3="{}".format(sim_matrix[1][2]))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
