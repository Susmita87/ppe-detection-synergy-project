import sqlite3
import json
import numpy as np
import os
from datetime import datetime

DB_PATH = "database/embeddings.db"

class PersonDatabase:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        self.conn.commit()

    def save_person(self, embedding, metadata=None):
        """
        Saves a new person embedding.
        embedding: numpy array or list
        """
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            embedding_bytes = embedding.tobytes()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO persons (embedding, metadata)
                VALUES (?, ?)
            ''', (embedding_bytes, metadata_json))
            self.conn.commit()
            last_id = cursor.lastrowid
            print(f"DATABASE: Successfully saved person with Global ID {last_id}")
            return last_id
        except Exception as e:
            print(f"DATABASE ERROR in save_person: {e}")
            self.conn.rollback()
            return None

    def update_last_seen(self, person_id):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE persons SET last_seen = CURRENT_TIMESTAMP WHERE id = ?
        ''', (person_id,))
        self.conn.commit()

    def get_all_persons(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, embedding, metadata FROM persons')
        rows = cursor.fetchall()
        
        persons = []
        for row in rows:
            persons.append({
                "id": row[0],
                "embedding": np.frombuffer(row[1], dtype=np.float32),
                "metadata": json.loads(row[2]) if row[2] else {}
            })
        return persons

    def find_match(self, embedding, threshold=0.75):
        """
        Finds the closest person in the DB.
        Returns (person_id, similarity) or (None, 0)
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        all_persons = self.get_all_persons()
        if not all_persons:
            return None, 0
        
        best_match_id = None
        best_similarity = -1
        
        for person in all_persons:
            # Skip if shapes don't match (e.g. model change)
            if embedding.shape != person["embedding"].shape:
                continue
                
            sim = float(np.dot(embedding, person["embedding"]))
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = person["id"]
        
        if best_similarity >= threshold:
            return best_match_id, best_similarity
        return None, best_similarity

# Global instance
db = PersonDatabase()
