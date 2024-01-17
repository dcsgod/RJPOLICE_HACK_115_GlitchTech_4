import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def apply_nlp_on_fir(fir_text):
    # Tokenize the FIR text into sentences and words
    sentences = sent_tokenize(fir_text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        [word for word in sentence if word.lower() not in stop_words]
        for sentence in words
    ]

    # Part-of-speech tagging
    pos_tags = [pos_tag(sentence) for sentence in filtered_words]

    # Named Entity Recognition (NER)
    ner_tags = [ne_chunk(pos_tag) for pos_tag in pos_tags]

    # Extract entities from NER results
    entities = []
    for ner_tree in ner_tags:
        entities.extend([
            " ".join([token for token, pos in entity])
            for entity in ner_tree
            if isinstance(entity, nltk.Tree)
        ])

    return entities

# Example FIR text
fir_text = "On January 1, a theft occurred at 123 Main Street. The complainant reported the incident to the police."

# Apply NLP on FIR
result_entities = apply_nlp_on_fir(fir_text)

# Print the extracted entities
print("Extracted Entities:", result_entities)