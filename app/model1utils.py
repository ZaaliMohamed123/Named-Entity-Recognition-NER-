import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json



# Load the NER_model_1
model_path = os.path.join('..','NER_model_1','ner_model.keras')
model_1 = load_model(model_path)


# Load the word2id dictionary from the JSON file
with open(os.path.join('..','NER_model_1','word2id.json'), 'r') as f:
    word2id = json.load(f)

def predict_ner_tags(sentence, model, word2id):
    # Diviser la phrase en tokens
    tokens = sentence.split()

    # Convertir les tokens en leurs indices correspondants
    token_indices = [word2id.get(word, word2id["The"]) for word in tokens]

    # Rembourrer les indices de tokens à la longueur maximale
    token_indices_padded = pad_sequences([token_indices], maxlen=104, padding="post", value=word2id["The"])

    # Faire des prédictions
    predictions = model.predict(token_indices_padded)

    # Obtenir les indices prédits
    predicted_indices = np.argmax(predictions, axis=-1)[0]


    return list(zip(tokens, predicted_indices))


tag_mapping= {
    0: 'Geographical Entity',
    1: 'Event',
    2: 'Organization' ,
    3: 'Time' ,
    4: 'Person' ,
    5: 'Nationality' ,
    6: 'Organization',
    7: 'Time' ,
    8: 'Nationality' ,
    9: 'Artifact' ,
    10: 'Geopolitical Entity' ,
    11: 'Person' ,
    12: 'Artifact',
    13: 'Geopolitical Entity',
    14: 'Event',
    15: 'Geographical Entity',
    16: 'Outside' 
}

def display_ner_predictions(predictions, tag_mapping):

    entities = []
    current_entity = []
    for word, tag_id in predictions:
        tag = tag_mapping.get(tag_id, "O")  # Get full tag name from mapping

        if tag != "Outside":  # Beginning of a new entity or inside an existing one
            if not current_entity or current_entity[0][1] == tag:
                current_entity.append((word, tag))
            else: # Different entity type
                entities.append((" ".join([w for w, _ in current_entity]), current_entity[0][1]))
                current_entity = [(word, tag)]
        else:  # Outside any entity (O tag)
            if current_entity:
                entities.append((" ".join([w for w, _ in current_entity]), current_entity[0][1]))
            current_entity = []

    if current_entity:
        entities.append((" ".join([w for w, _ in current_entity]), current_entity[0][1]))

    result = {}
    if entities:
        print("Entities:")
        for entity_text, entity_type in entities:
            result[entity_text] = entity_type
            print(f"- {entity_text} ({entity_type})")
    else:
        print("No entities found.")

    return result


def predict_model_1(sentence):
    predictions = predict_ner_tags(sentence, model_1, word2id)
    return display_ner_predictions(predictions, tag_mapping)

