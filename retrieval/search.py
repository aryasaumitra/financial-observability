from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder


# COLLECTION_NAME = "financial_documents"

# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "financial_documents_bge"

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME
    )
    reranker = CrossEncoder(
        "BAAI/bge-reranker-base"
    )

    # ----------------------------------------------
    # Connect Qdrant
    # ----------------------------------------------

    client = QdrantClient(
        url="http://localhost:6333"
    )

    # ----------------------------------------------
    # Question
    # ----------------------------------------------

    query = input(
        "\nAsk a question about IRFC: "
    )

    # ----------------------------------------------
    # Embed question
    # ----------------------------------------------

    query_vector = model.encode(
        query
    ).tolist()

    # ----------------------------------------------
    # Search
    # ----------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=20,

        with_payload=True
    )

    # ----------------------------------------------
    # Rerank results
    # ----------------------------------------------

    candidates = results.points

    pairs = [
        (query, result.payload["text"])
        for result in candidates
    ]

    reranker_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidates, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # ----------------------------------------------
    # Display top 5
    # ----------------------------------------------

    print("\n" + "=" * 80)
    print("RERANKED SEARCH RESULTS")
    print("=" * 80)

    for i, (result, reranker_score) in enumerate(
        ranked_results[:5],
        start=1
    ):

        payload = result.payload

        print()
        print(f"RESULT {i}")
        print(f"Vector Score    : {result.score:.4f}")
        print(f"Reranker Score  : {reranker_score:.4f}")
        print(f"Page            : {payload['page_start']}")
        print(f"Chunk           : {payload['chunk_id']}")

        print()
        print(payload["text"])

        print()
        print("-" * 80)

if __name__ == "__main__":
    main()