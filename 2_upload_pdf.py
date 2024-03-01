import nltk
import pdfplumber
import psycopg2
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


def upload_pdf(cur, model):
    # Open the PDF file
    with pdfplumber.open(r"D:\docs\rpg\Warhammer\Warhammer_Fantasy_Roleplay_Rulebook_PDF_180824.pdf") as pdf:
        text = ""
        # Extract text from each page
        for page in pdf.pages:
            text += page.extract_text()
    sentences = tokenize_into_sentences(text)
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")
        insert_record(cur, model, sentence)

    # Tokenize the extracted text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # Extract feature vectors for each sentence
    feature_vectors = last_hidden_states[:, 0, :]  # Using the [CLS] token representation


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Connect to a database ...")
    conn = connect_database()
    # Create a cursor object to execute SQL queries
    cur = conn.cursor()
    # nltk.download('punkt')
    # Load pre-trained model tokenizer (vocabulary)
    print("Load tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Load pre-trained model (weights)
    print("Load model ...")
    model = AutoModel.from_pretrained("avsolatorio/GIST-small-Embedding-v0")
    print("Upload pdf ...")
    upload_pdf(cur, model)
    # Close cursor and connection
    cur.close()
    conn.commit()
    conn.close()
