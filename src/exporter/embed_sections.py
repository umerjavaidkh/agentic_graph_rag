from ..shared.config.settings import EMBEDDING_MODEL
from ..shared.neo4j.driver import get_neo4j_driver
from ..shared.model_providers.factory import get_embedding_provider
from ..shared.storage.vector.factory import get_vector_store


def get_embedding(text: str) -> list[float]:
    """Generate an embedding for text — always OpenAI, independent of MODEL_PROVIDER."""
    provider = get_embedding_provider()
    response = provider.embeddings(model=EMBEDDING_MODEL, input=text[:8000])
    return response.data[0].embedding


def embed_all_sections():
    driver = get_neo4j_driver()
    vector_store = get_vector_store()

    with driver.session() as session:
        # Get all sections
        result = session.run("MATCH (s:Section) RETURN s.id AS id, s.search_text AS text")
        sections = [r.data() for r in result]

        print(f"Generating embeddings for {len(sections)} sections...")

        for sec in sections:
            embedding = get_embedding(sec["text"])
            vector_store.upsert(sec["id"], embedding)
            print(f"  ✓ {sec['id']}")

    print("Done! All sections have embeddings.")


if __name__ == "__main__":
    embed_all_sections()
