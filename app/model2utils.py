import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json


# Load the NER_model_1
model_path = os.path.join('..','NER_model_2','ner_model_2.keras')
model_2 = load_model(model_path)

tag_names = {
      0: 'Miscellaneous' ,
      1: 'Person',
      2: 'Location' ,
      3: 'Other' ,
      4: 'Location' ,
      5: 'Miscellaneous' ,
      6: 'Organization',
      7: 'Person',
      8: 'Organization'
    }

# Load the word2id dictionary from the JSON file
with open(os.path.join('..','NER_model_2','word_to_idx.json'), 'r') as f:
    word_to_idx = json.load(f)

# Définir une fonction pour tester le modèle sur une phrase personnalisée
def test_sentence(sentence, model, word_to_idx, tag_names, max_len=50):
    
    # Tokeniser la phrase
    tokens = sentence.split()

    # Convertir les tokens en indices
    token_indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in tokens]

    # Appliquer le padding à la séquence
    token_indices_padded = pad_sequences([token_indices], maxlen=max_len, padding="post", value=word_to_idx["<PAD>"])

    # Faire des prédictions
    predictions = model.predict(token_indices_padded)
    predicted_indices = np.argmax(predictions, axis=-1)[0]  # Obtenir les indices des étiquettes les plus probables

    # Décoder les prédictions
    #predicted_tags = [idx_to_tag[idx] for idx in predicted_indices[:len(tokens)]]

    predicted_tags = [tag_names[idx] for idx in predicted_indices[:len(tokens)]]


    # Associer les mots à leurs étiquettes
    return list(zip(tokens, predicted_tags))

def display_ner_predictions(predictions):

    entities = []
    current_entity = []
    for word, tag in predictions:

        if tag != "Other":  # Beginning of a new entity or inside an existing one
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

def predict_model_2(sentence):
    predictions = test_sentence(sentence, model_2, word_to_idx, tag_names)
    return display_ner_predictions(predictions)

