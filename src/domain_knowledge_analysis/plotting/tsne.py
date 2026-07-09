from sklearn.manifold import TSNE


def compute_tsne(embeddings, seed):
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)