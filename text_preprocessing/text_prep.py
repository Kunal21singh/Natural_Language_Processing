import spacy
import re

text = "The movie was very entertaining"

# Removing punctuation and normalizing the text
def clean_and_normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

text = clean_and_normalize_text(text)

# Loading the English language model and creating a Doc object
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for token in doc:
    print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}, Tag: {token.tag_}, Dep: {token.dep_}")

