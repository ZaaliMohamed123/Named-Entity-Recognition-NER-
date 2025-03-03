import stanza 

nlp = stanza.Pipeline('en',download_method=None)  # Charger le pipeline NLP anglais

def predict_stanza(sentence):
    
    doc = nlp(sentence)
    entities = {entity.text:entity.type for entity in doc.entities}


    return entities
