import os
import face_recognition
import pickle

def encode_faces(dataset_path="dataset"):
    known_encodings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder):
            continue
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)

    data = {"encodings": known_encodings, "names": known_names}
    with open("embeddings.pickle", "wb") as f:
        pickle.dump(data, f)
