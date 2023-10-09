# Using the Text Similarity module

1. Jupyter Notebook

	- [textSimilarity.ipynb](https://github.com/shriadke/Text-Similarity/blob/main/textSimilarity.ipynb) can be used to develop and test the project line-by-line.
	
2. Run [textSimilarity.py](https://github.com/shriadke/Text-Similarity/blob/main/textSimilarity.py)

	- This file gives public methods for computing text Similarity as well as it can be run as a stand-alone module.
	- To verify the results on sample texts with standard configurations, run this file as `python3 textSimilarity.py`.
	- The main method can be modified to change any parameters.
	- This module can be imported in any other python file using `from textSimilarity import *`.
	- Method `get_text_similarity(texts)` can be called with following custom arguments:
		* texts : list of input texts to compare.
		* method: specifies method to use. default="cosine"
				  options=["cosine","jaccard","euclidian"]
		* exclude_stopwords: A boolean flag indicating whether to exclude stop words from comparison or not.
		* exclude_punctuation: A boolean flag indicating whether to exclude punctuation from comparison or not.
		
3. Run [app.py](https://github.com/shriadke/Text-Similarity/blob/main/app.py)

	- Run this file using `python3 app.py` in a command prompt to start the service on localhost.
	- Access this service along with the web page on http://127.0.0.1:5000/ .
	
4. Docker App
	
	- If needed, this service/app can be deployed with the help of docker.
	- Follow the following instructions to build and run the container from the main folder:
		* `docker build -t text_similarity_app .`
		* `docker run -p <port>:<port> text_similarity_app`
		* The service can be accessed via the localhost URL http://localhost:5000/ .
		
5. Heroku App

	- This app is deployed on Heroku platform by connecting this github repo.
	- Visit https://textsimilarityapp.herokuapp.com/ to have fun with this app implementation.