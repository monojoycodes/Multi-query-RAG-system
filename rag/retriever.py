def retrieve_documents(
    collection,
    original_query,
    expanded_queries
):

    all_queries = (
        [original_query]
        + expanded_queries
    )

    results = collection.query(
        query_texts=all_queries,
        n_results=5
    )

    unique_docs = set()

    for docs in results["documents"]:
        unique_docs.update(docs)

    return list(unique_docs)