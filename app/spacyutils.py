import spacy

nlp = spacy.load("en_core_web_sm")

def predict_spacy(sentence):
    
    doc = nlp(sentence)
    entities = {}

    for ent in doc.ents:
        entities[ent.text] = ent.label_    
    

    return entities
