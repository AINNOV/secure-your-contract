## This code is designed to be a module in evaluate_rag.py. ##

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf

class ContractRetriever:
    def __init__(self, index_path = None, model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        ## load faiss index and contract - 'response' pairs ##
        if index_path:
            self.index = faiss.read_index(index_path)
            self.prompts, self.responses = self.load_prompts_and_responses(index_path)

        else:
            self.index = None
            self.prompts = []
            self.responses = []

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    ## create faiss index from your json only if needed ##
    def create_faiss_index(self, json_data):
        prompts = [item["prompt"] for item in json_data]
        embeddings = self.model.encode(prompts, convert_to_numpy = True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) # L2 distance
        index.add(embeddings)
        return index, prompts

    ## save only if needed ##
    def save_faiss_index(self, index, file_path="./faiss_index.bin"):
        faiss.write_index(index, file_path)

    ## load "prompt"-"response" json and save to instance variables ##
    def load_prompts_and_responses(self, index_path):
        json_file_path = "./data/raw/SYC_train_with_testPDF.json"
        json_data = self.load_json(json_file_path)
        prompts = [item["prompt"] for item in json_data]
        responses = [item["response"] for item in json_data]
        return prompts, responses

    ## retrieve top k similar docs(=contracts) and corresponding responses(=analyses) ##
    def search_similar_documents(self, query, top_k=  3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"prompt": self.prompts[idx], "response": self.responses[idx], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]
        return results