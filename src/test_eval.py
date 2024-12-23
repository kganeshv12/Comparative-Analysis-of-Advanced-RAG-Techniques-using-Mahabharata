from eval import evaluate_2, evaluate_1
from RAG_Methods import Chunk_high, Chunk_low, Decomposition, Hyde, MultiQueryRetrievers, RagFusion, StepBackPrompt, SentenceWindowRetriever, Rerank, SemanticRouting
import json
import time

LLMs =  ["llama-3.2-1b-preview","llama-3.2-3b-preview","mixtral-8x7b-32768","gemma-7b-it", "gemma2-9b-it","llama-3.1-8b-instant","llama-3.1-70b-versatile" ]
Methods = [Chunk_high, Chunk_low, Decomposition, Hyde, MultiQueryRetrievers, RagFusion, StepBackPrompt, SentenceWindowRetriever, Rerank, SemanticRouting]

for method in Methods:
    print("Using ",method.__name__)
    for llm in LLMs:
        
        evaluate_2(method, llm)