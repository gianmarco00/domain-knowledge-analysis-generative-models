def compute_pca(embeddings):
    from cuml.decomposition import PCA
    
    pca = PCA(
        n_components=2,
        output_type="numpy",
    )

    pca_embeddings = pca.fit_transform(embeddings)

    return (
        pca_embeddings,
        pca.explained_variance_ratio_,
    )