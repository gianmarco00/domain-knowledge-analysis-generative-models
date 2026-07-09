
def compute_tsne(embeddings, seed):
    import cupy as cp
    from cuml.manifold import TSNE

    tsne_embeddings = TSNE(
        n_components=2,  
        random_state=seed,
    ).fit_transform(embeddings)

    if isinstance(tsne_embeddings, cp.ndarray):
        tsne_embeddings = cp.asnumpy(tsne_embeddings)

    return tsne_embeddings
