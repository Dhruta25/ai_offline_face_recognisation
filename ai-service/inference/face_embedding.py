import cv2
import numpy as np

def generate_embedding(face_path):

    image = cv2.imread(face_path)

    image = cv2.resize(
        image,
        (112,112)
    )

    embedding = image.flatten()

    embedding = embedding / 255.0

    return embedding


if __name__ == "__main__":

    emb = generate_embedding(
        "cropped_face.jpg"
    )

    print(emb.shape)