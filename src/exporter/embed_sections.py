import os
import numpy as np
from neo4j import GraphDatabase
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password123")

def get_embedding(text: str) -> list[float]:
    """Generate OpenAI embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]  # OpenAI limit
    )
    return response.data[0].embedding

def embed_all_sections():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    with driver.session() as session:
        # Get all sections
        result = session.run("MATCH (s:Section) RETURN s.id AS id, s.text AS text")
        sections = [r.data() for r in result]
        
        print(f"Generating embeddings for {len(sections)} sections...")
        
        for sec in sections:
            embedding = get_embedding(sec['text'])
            # Store as JSON string (Neo4j doesn't have native float arrays easily)
            session.run("""
                MATCH (s:Section {id: $id})
                SET s.embedding = $embedding
            """, id=sec['id'], embedding=embedding)  # ← pass list directly, not str()
            print(f"  ✓ {sec['id']}")
    
    driver.close()
    print("Done! All sections have embeddings.")

if __name__ == "__main__":
    embed_all_sections()
