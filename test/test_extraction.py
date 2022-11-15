import spacy

nlp_model = spacy.load("./output/model-best")
text = 'where is the bedroom'
doc = nlp_model(text)
 
for ent in doc.ents:
    if ent.label_ == "DEST":
        print(str(ent.text))