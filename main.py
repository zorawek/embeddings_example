import nltk
import numpy
import pdfplumber
import psycopg2
import torch
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel


def connect_database():
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="admin",
        host="localhost",
        port="5434"
    )
    register_vector(conn)
    return conn


# Function to tokenize text into sentences
def tokenize_into_sentences(text):
    return nltk.sent_tokenize(text)


def insert_record(cur, model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Extract the output embeddings
    last_hidden_states = outputs.last_hidden_state
    # Extract feature vectors for each sentence
    feature_vectors = last_hidden_states[:, 0, :]  # Using the [CLS] token representation
    cur.execute('INSERT INTO items (sentence, embedding) VALUES (%s, %s)',
                (sentence[:250], feature_vectors[0].tolist(),))


def query_database(cur, model, sentence, operator):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    feature_vectors = last_hidden_states[:, 0, :]
    embedding = numpy.array(feature_vectors[0].tolist())
    cur.execute('SELECT sentence, embedding ' + operator + ' %s as distance FROM items ORDER BY distance LIMIT 5', (embedding,))
    # cur.execute('SELECT * FROM items WHERE items.embedding <-> (%s) < 5', (feature_vectors[0].tolist()))
    rows = cur.fetchall()
    return rows


def upload_pdf(cur, model):
    # Open the PDF file
    with pdfplumber.open(r"D:\docs\rpg\Warhammer\Warhammer_Fantasy_Roleplay_Rulebook_PDF_180824.pdf") as pdf:
        text = ""
        # Extract text from each page
        for page in pdf.pages:
            text += page.extract_text()

    print(len(text))
    sentences = tokenize_into_sentences(text)
    # Print the list of sentences
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")
        insert_record(cur, model, sentence)

    # Tokenize the extracted text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # Extract feature vectors for each sentence
    feature_vectors = last_hidden_states[:, 0, :]  # Using the [CLS] token representation
    print(len(feature_vectors))
    print(len(feature_vectors[0]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Connecting to a database")
    conn = connect_database()
    # Create a cursor object to execute SQL queries
    cur = conn.cursor()

    # nltk.download('punkt')

    # Load pre-trained model tokenizer (vocabulary)
    print("Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Define the sentences
    sentences = ["This person is happy", "I need to go to work"]
    # Tokenize the sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    # Load pre-trained model (weights)
    print("Automodel")
    model = AutoModel.from_pretrained("avsolatorio/GIST-small-Embedding-v0")
    # model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    # Forward pass, get model output
    print("Model")
    outputs = model(**inputs)
    print("State")
    # Distance
    print("### L2 Distance ###")
    for n in query_database(cur, model, "critical hit", "<->"):
        print(n)
    print("### Dot Product ###")
    for n in query_database(cur, model, "critical hit", "<#>"):
        print(n)
    print("### Cosine Distance ###")
    for n in query_database(cur, model, "critical hit", "<=>"):
        print(n)


    # Extract the output embeddings
    last_hidden_states = outputs.last_hidden_state
    # Extract feature vectors for each sentence
    feature_vectors = last_hidden_states[:, 0, :]  # Using the [CLS] token representation
    # Calculate the cosine similarity between the feature vectors
    print("Similarity")
    # cur.execute('INSERT INTO items (sentence, embedding) VALUES (%s, %s)', ("Hello", feature_vectors[0].tolist(),))
    cosine_similarity = torch.nn.functional.cosine_similarity(feature_vectors[0], feature_vectors[1], dim=0)
    # Print the feature vectors and cosine similarity
    # print("Feature vector for 'This person is happy':", feature_vectors[0])
    # print("Feature vector for 'This person is sad':", feature_vectors[1])
    print("Cosine similarity between the feature vectors:", cosine_similarity.item())
    # upload_pdf(cur, model)
    # # Close cursor and connection
    cur.close()
    conn.commit()
    conn.close()
