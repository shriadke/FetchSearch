# Coding Exercise - Intelligent Text Search based on NLP

## Task

This challenge is focused on the similarity between search texts. The objective is to build a tool that allows users to intelligently search for offers via text input from the user. Here is a program that takes as inputs a text sting and returns the best matched offers one can get on Fetch App.


## File Structure
This repo contains python implementation of text similarity app. 

1. [fetchsearch.ipynb](https://github.com/shriadke/FetchSearch/blob/main/fetchsearch.ipynb)
	The notebook  contains my initial approach towards this project and includes line by line implementation of my approach along with explanation.
	
2. [fetchsearch.py](https://github.com/shriadke/FetchSearch/blob/main/fetchsearch.py)
	This is the Python module that encapsulates all the differen methods performed in this approach along with appropriate comments. This module is further used by the API service in [app.py](https://github.com/shriadke/FetchSearch/blob/main/app.py)

3. [app.py](https://github.com/shriadke/FetchSearch/blob/main/app.py)
	This is the API POST service implementation of fetchsearch app that uses `get_text_similarity(texts)` method to compute the similarity matrix between 3 given texts.

4. [templates/index.html](https://github.com/shriadke/FetchSearch/blob/main/templates/index.html)
	This is the web page which will be loaded to use the above service through a web browser.
	
5. [requirements.txt](https://github.com/shriadke/FetchSearch/blob/main/requirements.txt)
	This file contains the necessary packages required for this project.

6. [Dockerfile](https://github.com/shriadke/FetchSearch/blob/main/Dockerfile)
	This file contains build instructions for API deployment using Docker.
	
7. [Procfile](https://github.com/shriadke/FetchSearch/blob/main/Procfile)
	This file is used to deploy the app using Heroku platform.

## Environment

This project is implementes using Python 3.10, Flask 1.1.12, Jupyter Notebook and Spyder IDE along with DockerHub and Heroku for deployment.

## [Approach](https://github.com/shriadke/FetchSearch/blob/main/docs/APPROACH.md)

Here I have considered 3 different basic text similarity approaches that are easy to implement without the use of external libraries such as Scikit-Learn, NLTK, Gensim, Spacy, etc. These approaches consider individual texts as a list of tokenized words and performs mathematical similarity operations. The details of all of these can be found in [Approach](https://github.com/shriadke/FetchSearch/blob/main/docs/APPROACH.md). Follwing are the steps involved in this project:

1. Data Cleaning
	
	Tokenization and filtering words to create text vocabulary
	
2. Data Processing

	Vectorization to compute similarity based on word vectors.

3. Text Similarity Metrics

	Three metrics used to compute text similarity: Jaccard Index, Cosine Similarity, Euclidian Distance.

4. App deployment

	The app is deployed on web using flask. This is not tested for all the edge cases as this is a basic implementation.

### Do you count punctuation or only words?

	It depends on the texts to be compared. In the case of short texts, it was observed that counting punctuations and stop words helps in comparison output.
	
### Which words should matter in the similarity comparison?

	Mainly the words that occurs frequently in both texts matters most to the similarity. But the words unique to the text also impacts the score as they define dissimilarity between texts.
	
### Do you care about the ordering of words?

	It should be considered as a whole, but to get somewhat idea about the relevance of two texts, I have not considered this here.
	
	This may cause problems in statements such as "I am Shrinidhi" and "Am I Shrinidhi" as they are totally different but based on the word counts, my approach will yield in 1.
	
### What metric do you use to assign a numerical value to the similarity?

	Jaccard Index, Cosine Similarity, Euclidian Distance.
	
### What type of data structures should be used?

	I used Lists and Dictionaries to store the data and prepare the vocabulary. The result is in the form of Similarity matrix which is a list of lists.


### If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.


### If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.


### If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.


### The tool should also return the score that was used to measure the similarity of the text input with each offer

### Detailed responses to each problem, with a focus on the production pipeline surrounding the model.

### Identifies several useful techniques to approach eReceipt classification and entity extraction.
	
## Usage

Clone this repo using `git clone  https://github.com/shriadke/FetchSearch`.

As described in [File Structure](https://github.com/shriadke/FetchSearch#file-structure), you can verify the files.

This repo can be used in multiple ways such as command line, Jupyter Notebooks, python IDEs, flask app on local, Run app on Docker, hosting on web site. The simplest of which to validate is the use of online web page.

The Text Similarity App is deployed on Heroku which can be tested directly by visiting https://fetchsearchapp.herokuapp.com/. *Have some fun with it, because I did :wink:!!*

More details of other excecution methods can be found in [Usage.md](https://github.com/shriadke/FetchSearch/blob/main/docs/Usage.md)

## Results and Discussion

This project allows us to use a simple text similarity app and we can compare multiple texts/ queries with each other in a single use.

### Results on Sample texts

Following table shows the results on sample texts using the 3 approaches with allowing stop words and punctuations with considering all 3 texts at a time in the corpus.

| Comparison | Cosine Similarity | Euclidian Distance | Jaccard Index |
| :-: | :-: | :-: | :-: |
|Sample1 <-> Sample2|0.717|0.829|0.412|
|Sample1 <-> Sample3|0.283|0.675|0.135|
|Sample2 <-> Sample3|0.254|0.664|0.125|

Though we see euclidian distance gave a better score for comparison between Sample1 and Sample2, the similarity between Sample1 and Sample3 is also high, which is not the case. Therefore, It can be seen that the Cosine similarity gives overall better results on these sentences.

### Failure Results

Here I have selected three sentences which represents the disadvantage of using the simple word count based approaches.

'''
Sample1 = "Protect yourself from coronavirus wear the mask."
Sample2 = "Protect yourself from the mask wear coronavirus."
Sample3 = "Coronavirus protect yourself from wear the mask."
'''

| Comparison | Cosine Similarity | Euclidian Distance | Jaccard Index |
| :-: | :-: | :-: | :-: |
|Sample1 <-> Sample2|0.997|1|1|
|Sample1 <-> Sample3|0.997|1|1|
|Sample2 <-> Sample3|0.997|1|1|

It was obvious that all these methods consider all three texts as the same because of the foundation i.e. word frequency.

### Real-world Use case

Suppose we have to identify the products available at different super-markets based on their item description. Consider a product "Coke" which has following descriptions at respective stores:

'''
Walmart = "A best-selling , refreshing , sweetened , carbonated soft drink."
Kroger = "World's best-selling Coke , is a carbonated soft drink manufactured by The Coca-Cola Company."
Target = "It is a carbonated , sweet soft drink and is the world's best-selling drink."
'''

| Comparison | Cosine Similarity | Euclidian Distance | Jaccard Index |
| :-: | :-: | :-: | :-: |
|Sample1 <-> Sample2|0.194|0.347|0.35|
|Sample1 <-> Sample3|0.222|0.367|0.35|
|Sample2 <-> Sample3|0.409|0.458|0.455|

It can be seen that euclidian distance and Jaccard Index works well on this set of sample texts. Based on improvements, these methods can be used in such real-world applications.

### Future Work

Due to the restrictions of using only standard libraries, I achieved this basic app. In future, given time and additional libraries, one can build a stable app that does the task as above. Following are some areas of improvements in my approach:

1. Better Tokenization

	- NLTK has better tokenization methods that does a precise work in identifying every little token in the given text.
	- Punctuations, parantheses, symbols can be easily tokenized using `nltk.word_tokenize()` functions.

2. Stop words and Punctuations
	
	- Though I have addressed this in a custom way, we can use built-in lists and functions to get rid of these.
	- Also, libraries such as Scikit-Learn has provision of performing all the text cleaning and processing tasks in single built-in function.
	- Example of such is `TfidfVectorizer` method that gives feature vectors along with tokenization, stop_word removal, n-grams, etc.
	
3. Feature Improvement

	- TF-IDF is a strong method though it lacks in basic approaches.
	- Implementing proper N-grams can improve TF-IDF performace significantly.
	- Other representations such as Word2Vec embeddings, Smooth Inverse Frequency also can be used in conjugation with this.
	- Library such as Gensim makes it easier to compare such texts based on multiple feature representations.
	
4. Similarity Methods/Metrics

	- This project implements basic vector/list based similarity methods that does not perform well in every scenario.
	- Instead one can change the approach and use language modelling.
	- The use of lemmatization and synonyms checker will definitely improve any method's performace.
	- Different supervised learning methods including neural nets such as RNN, Bi-LSTMs can be used to perform this task with the help of adequate training data.
	
	
