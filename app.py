from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# TODO: Fetch dataset, initialize vectorizer and LSA here

newsgroups = fetch_20newsgroups(subset = 'all')

stop_words = stopwords.words('english')

vectorizer = TfidfVectorizer(stop_words = stop_words, max_features = 10000)

X = vectorizer.fit_transform(newsgroups.data)

svd = TruncatedSVD(n_components=100)
X_lsa = svd.fit_transform(X)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    # Transform the query into the same space as the documents
    query_tfidf = vectorizer.transform([query])
    query_lsa = svd.transform(query_tfidf)
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_lsa, X_lsa).flatten()
    
    # Get top 5 documents
    top_5_indices = similarities.argsort()[-5:][::-1]
    top_5_documents = [newsgroups.data[i] for i in top_5_indices]
    top_5_similarities = [similarities[i] for i in top_5_indices]
    
    return top_5_documents, top_5_similarities, top_5_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)

    print(f"Documents type: {type(documents)}")
    print(f"Similarities type: {type(similarities)}")
    print(f"Indices type: {type(indices)}")

    indices = indices.tolist()

    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True, port = 3000)
