import os
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation, util, CrossEncoder
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import pandas as pd
import re
import pickle

from fetchSearch.config.configuration import ConfigurationManager
from fetchSearch.logging import logger



class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('[-]+',' ', text)
        text = re.sub('[^A-Za-z0-9\[\]\s]+', '', text)   
        #text = text.strip()
        return text

    
    def predict(self,text):
        
        import pickle
        with open(self.config.synthetic_df, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_offers = stored_data['offers']
            stored_sentences = stored_data['offer_processed']
            stored_embeddings = stored_data['embeddings']
        
        # Bi-encoder saved pretrained path # currently pre-trained msmacro, but can be updatd to our trained model
        trained_sbert_model = SentenceTransformer(self.config.model_path)  
        
        query = self.clean_text(text)
        top_k = 10  # Can adjust to see more results
        q_embeddings =  trained_sbert_model.encode([query], show_progress_bar=True)
        
        logger.info(f"Searched for:{text}")
        logger.info("Getting all available offers!!")

        hits = util.semantic_search(q_embeddings, stored_embeddings, top_k=top_k)
        hits = hits[0] 
        logger.info("Found all available offers!!")

        offers = []
        scores = []
        for hit in hits:
            offer = stored_offers[hit['corpus_id']]
            score = hit['score']
            offers.append(offer)
            scores.append(score)
        logger.info("Filtering duplicate offers!!")
        # Use cross-encoder on offers to filter similar offers and re-ranking the remaining.
        cross_model = CrossEncoder(self.config.tokenizer_path) # Using cross-encoder/ms-marco-MiniLM-L-6-v2, but similar can be trained

        remove_idx = set()
        for i in range(len(offers)):
            for j in range(i+1, len(offers)):
                pair = [offers[i],offers[j]]
                cross_score = np.round(torch.sigmoid(torch.from_numpy(np.array(cross_model.predict(pair, show_progress_bar=False)))),3)
                if cross_score == 1:
                    remove_idx.add(j) # Remove same offers
        logger.info("Removed duplicate offers!!")
        
        logger.info("Re-Ranking available offers!!")
        cross_list = []
        filtered_hits = []
        for idx in range(len(hits)):
            if idx not in remove_idx:
                cross_list.append([stored_sentences[hits[idx]['corpus_id']], query])
                filtered_hits.append(hits[idx])

        # Use Cross Encoder to re-rank
        ce_scores = cross_model.predict(cross_list)
        for idx in range(len(filtered_hits)):
            filtered_hits[idx]['cross-encoder_score'] = np.round(torch.sigmoid(torch.from_numpy(np.array(ce_scores[idx]))).numpy().item(),4) * 100

        #Sort list by CrossEncoder scores
        filtered_hits = sorted(filtered_hits, key=lambda x: x['cross-encoder_score'], reverse=True)
        logger.info("Re-Ranking Done!!")

        offers = []
        scores = []
        for hit in filtered_hits:
            logger.info(f"\t{hit['cross-encoder_score']}\t{stored_offers[hit['corpus_id']]}\t")
            offers.append(stored_offers[hit['corpus_id']])
            scores.append(np.round(hit['cross-encoder_score'],2))

        output_df = pd.DataFrame({
            "Offer" : offers,
            "Relevance" : scores
        })

        return output_df