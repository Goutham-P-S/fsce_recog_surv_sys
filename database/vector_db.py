import numpy as np
import logging
import json
import os
from typing import List, Dict, Union

# Try to use standard types if pymilvus is not available
# We are removing pymilvus dependency here completely.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorDB")

class VectorDB:
    def __init__(self, uri: str = "./milvus_demo_local.json", collection_name: str = "criminals", dim: int = 512):
        """
        Initialize a simple local VectorDB using a JSON file.
        
        Args:
            uri: Path to local JSON file.
            collection_name: Key in the JSON (simulated collection).
            dim: Dimension of the face embeddings.
        """
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim
        self.data = []
        
        self._load()

    def _load(self):
        """Load data from JSON if exists."""
        if os.path.exists(self.uri):
            try:
                with open(self.uri, 'r') as f:
                    all_data = json.load(f)
                    self.data = all_data.get(self.collection_name, [])
                logger.info(f"Loaded {len(self.data)} records from {self.uri}")
            except Exception as e:
                logger.error(f"Failed to load DB: {e}")
        else:
            logger.info("No existing database found. Starting fresh.")

    def _save(self):
        """Save data to JSON."""
        try:
            # Load existing if any (to preserve other collections if we were supporting that)
            if os.path.exists(self.uri):
                with open(self.uri, 'r') as f:
                    try:
                        all_data = json.load(f)
                    except json.JSONDecodeError:
                        all_data = {}
            else:
                all_data = {}
            
            all_data[self.collection_name] = self.data
            
            with open(self.uri, 'w') as f:
                json.dump(all_data, f)
            logger.info("Database saved.")
        except Exception as e:
            logger.error(f"Failed to save DB: {e}")

    def insert_embeddings(self, identities: List[Dict]):
        """
        Insert embeddings into the database.
        
        Args:
            identities: List of dicts, each containing 'vector' (List[float]) and other metadata.
        """
        # Convert all to list for JSON serialization
        for item in identities:
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item[k] = v.tolist()
        
        self.data.extend(identities)
        self._save()
        logger.info(f"Inserted {len(identities)} records.")

    def search_embedding(self, query_vector: Union[List[float], np.ndarray], limit: int = 1, threshold: float = 0.30):
        """
        Search for the closest embedding using Cosine Similarity with k-NN Voting.
        """
        if not self.data:
            return []

        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
            
        # Normalize query
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q
        else:
            return []
            
        # Vectorized approach:
        vectors = np.array([item['vector'] for item in self.data])
        
        # Normalize DB vectors (in case they weren't on insert)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-10
        vectors_norm = vectors / norms[:, np.newaxis]
        
        # Dot product (Cosine Similarity)
        scores = np.dot(vectors_norm, query_vector)
        
        # k-NN Logic: Get top K candidates
        K = 5
        top_indices = np.argsort(scores)[::-1][:K]
        
        # Vote/Aggregate
        candidates = {} # name -> {max_score: float, count: int, sum_score: float, id: str}
        
        for idx in top_indices:
            score = float(scores[idx])
            # Only consider "votes" that cross a minimal sanity threshold
            # somewhat lower than the final strict threshold to allow partial matches to contribute
            if score < 0.25: 
                continue
                
            record = self.data[idx]
            name = record.get('name', 'Unknown')
            
            if name not in candidates:
                candidates[name] = {'max_score': score, 'count': 0, 'sum_score': 0.0, 'id': record.get('id')}
            
            candidates[name]['count'] += 1
            candidates[name]['sum_score'] += score
            if score > candidates[name]['max_score']:
                candidates[name]['max_score'] = score

        if not candidates:
             return []

        # Decide winner:
        # Strategy: Highest Max Score (closest single match) is usually best for FaceID,
        # but frequent consistent matches (high sum score) is good for stability.
        # Let's use a hybrid: Sort by Max Score.
        best_name = max(candidates, key=lambda n: candidates[n]['max_score'])
        winner = candidates[best_name]
        
        final_score = winner['max_score']
        
        logger.info(f"k-NN Search: Winner '{best_name}' (Score: {final_score:.4f}, Votes: {winner['count']}/{K})")
        
        matches = []
        if final_score >= threshold:
            matches.append({
                "id": winner['id'],
                "name": best_name,
                "score": final_score,
                "landmark_3d_68": winner.get('landmark_3d_68') # Return stored 3D mesh for comparison
            })
            
        return matches
