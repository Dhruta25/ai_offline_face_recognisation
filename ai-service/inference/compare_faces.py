from scipy.spatial.distance import cosine

def compare_faces(
        emb1,
        emb2):

    similarity = 1 - cosine(
        emb1,
        emb2
    )

    return similarity