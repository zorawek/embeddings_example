import nltk
import numpy
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


def query_database(cur, model, sentence, operator):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    feature_vectors = last_hidden_states[:, 0, :]
    embedding = numpy.array(feature_vectors[0].tolist())
    cur.execute('SELECT sentence, embedding ' + operator + ' %s as distance FROM items ORDER BY distance LIMIT 5',
                (embedding,))
    rows = cur.fetchall()
    return rows


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Connect to a database")
    conn = connect_database()
    # Create a cursor object to execute SQL queries
    cur = conn.cursor()
    # Load pre-trained model tokenizer (vocabulary)
    print("Load tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Load model ...")
    model = AutoModel.from_pretrained("avsolatorio/GIST-small-Embedding-v0")

    # query_text = "critical hit"
    query_text = "horse riding"
    # Distance
    print("\n### L2 Distance ###")
    for index, row in enumerate(query_database(cur, model, query_text, "<->")):
        print(f"{index + 1}: {row}")
    print("\n### Dot Product ###")
    for index, row in enumerate(query_database(cur, model, query_text, "<#>")):
        print(f"{index + 1}: {row}")
    print("\n### Cosine Distance ###")
    for index, row in enumerate(query_database(cur, model, query_text, "<=>")):
        print(f"{index + 1}: {row}")
    cur.close()
    conn.close()
