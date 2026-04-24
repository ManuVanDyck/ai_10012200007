#!/usr/bin/env python3
"""
Migration script to convert ChromaDB embeddings to pickle format
Preserves all existing data while fixing compatibility issues
"""

import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

def migrate_chroma_to_pickle():
    """Migrate existing ChromaDB data to pickle format"""
    print("Starting migration from ChromaDB to pickle format...")
    
    try:
        # Load existing ChromaDB data
        client = chromadb.PersistentClient(path='./chroma_db')
        collection = client.get_or_create_collection(name='budget_chunks')
        
        # Get all data from ChromaDB
        result = collection.get(include=['documents', 'embeddings', 'metadatas'])
        
        documents = result['documents']
        embeddings = result['embeddings']
        
        print(f"Found {len(documents)} documents and {len(embeddings)} embeddings")
        
        if not documents:
            print("No data found in ChromaDB - nothing to migrate")
            return False
            
        # Create data directory
        os.makedirs('./data', exist_ok=True)
        
        # Store in pickle format
        data = {
            'chunks': documents,
            'embeddings': np.array(embeddings),
            'collection_name': 'budget_chunks'
        }
        
        with open('./data/budget_chunks.pkl', 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Successfully migrated {len(documents)} chunks to ./data/budget_chunks.pkl")
        print("Your embeddings are now preserved in the new format!")
        
        # Verify the migration
        with open('./data/budget_chunks.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
            
        print(f"Verification: Loaded {len(loaded_data['chunks'])} chunks")
        print("Migration completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_chroma_to_pickle()
