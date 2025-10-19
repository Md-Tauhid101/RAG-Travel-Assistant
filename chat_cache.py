import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import hashlib
import json
import time

from config import REDIS_CACHE_TTL

class SemanticCache():
    def __init__(self, redis_client, embed_fuc, ttl = REDIS_CACHE_TTL, similarity_threshold = 0.7):
        """Initialize the enhanced semantic cache with redis client, embedidng model, and TTL.

        Args:
            redis_client: Redis client instance
            enbedding nodel: Model to use for generating enbeddings
            ttl: Time to Live in seconds (default: 1 hour)
            sinilarity threshold: Threshold for considering queries sinitar (0-1)
        """
        self.redis_client = redis_client
        self.ttl= ttl
        self.similarity_threshold = similarity_threshold
        self.embed_func = embed_fuc

        # Create a namespace for storing query metadata
        self.query_meta_prefix = "query_meta:"

    def get_cache_key(self, query, context):
        """It will generate hash key for stoing in Redis
        Args:
            query: User query string
            context: Retrieved context string
            The cached result or None If not found

        """
        # Combine query and context in a determinitic way
        combined = f"{query}::{context}"

        # Create a hash to use as the cache key
        return hashlib.md5(combined.encode()).hexdigest()

    def add_to_cache(self, query, context, result, query_embedding = None):
        """Add a result to the cache with query metadata

        Args:
            query: User query string
            context: Retrienved context string
            result: Result to came 
        """

        cache_key = self.get_cache_key(query, context)
        self.redis_client.setex(cache_key, self.ttl, json.dumps(result))

        # If no embedding provided, only then compute it
        if query_embedding is None:
            query_embedding = self.embed_func(query)

        meta_key = f"{self.query_meta_prefix}{cache_key}"

        meta_data = {
            "query": query,
            "embedding": query_embedding
        }

        self.redis_client.setex(meta_key, self.ttl, json.dumps(meta_data))
        print(f"DEBUG: Added to metadata with key: {cache_key}")
    
    def get_from_cache_semantic(self, query, query_embedding = None):
        """Try to get a response fron cache using senantic similarity.
        Args:
            query: User query string
            Returns:
            The cached result or None If not found
        """
        # If no embedding provided, only then compute it
        if query_embedding is None:
            query_embedding = self.embed_func(query)

        meta_keys = self.redis_client.scan_iter(f"{self.query_meta_prefix}*")

        best_similarity = 0
        best_key = None

        for meta_key in meta_keys:
            meta_data = self.redis_client.get(meta_key)
            if not meta_data:
                continue

            try:
                meta_data = json.loads(meta_data)
                cached_embedding = meta_data.get("embedding")

                if cached_embedding:
                    similarity = self.calculate_similarity(query_embedding, cached_embedding)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        # remove prefix only — no decode needed
                        best_key = meta_key.replace(self.query_meta_prefix, "")
            except Exception as e:
                print(f"Error processing meta key: {e}")
                continue

        if best_key:
            cached_result = self.redis_client.get(best_key)
            if cached_result:
                print(f"Semantic cache hit (similarity={best_similarity:.3f})")
                return json.loads(cached_result)

        return None
    
    def get_from_cache(self, query):
        """Try to get a response fron cache using senantic matching.
        Args:
            query: User query string
            context: Retrieved context string
            Returns:
            The cached result or None If not found
        """
        
        semantic_result = self.get_from_cache_semantic(query)
        if semantic_result:
            print("Semantic cache hit!")
            return semantic_result
        
        print("DEBUS: CACHE MISS")
        return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """
            Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
        Returns:
            float: Cosine similarity (0-1)
        """
        # Convert to numpy array and reshape for sklearn
        vec1 = np.array(embedding1).reshape(1,-1)
        vec2 = np.array(embedding2).reshape(1,-1)

        # Calculate cosine similarity
        return cosine_similarity(vec1, vec2)[0][0]