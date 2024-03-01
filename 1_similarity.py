import nltk
import torch
from transformers import AutoTokenizer, AutoModel


# Function to tokenize text into sentences
def tokenize_into_sentences(text):
    return nltk.sent_tokenize(text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    print("Load tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Define the sentences
    sentence_1 = "This person is happy"
    sentence_2 = "This person is content"
    sentences = [sentence_1, sentence_2]
    # Tokenize the sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    # Load pre-trained model (weights)
    print("Load model ...")
    model = AutoModel.from_pretrained("avsolatorio/GIST-small-Embedding-v0")
    # model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    # Forward pass, get model output
    print("Create outputs ...")
    outputs = model(**inputs)
    print("Fetch states ...")
    # Extract the output embeddings
    last_hidden_states = outputs.last_hidden_state
    # Extract feature vectors for each sentence
    feature_vectors = last_hidden_states[:, 0, :]  # Using the [CLS] token representation
    # Calculate the cosine similarity between the feature vectors
    print("Similarity:")
    cosine_similarity = torch.nn.functional.cosine_similarity(feature_vectors[0], feature_vectors[1], dim=0)
    # Print the feature vectors and cosine similarity
    print(f"Feature vector for '{sentence_1}':", feature_vectors[0])
    print(f"Feature vector for '{sentence_2}':", feature_vectors[1])
    print("Cosine similarity between the feature vectors:", cosine_similarity.item())
