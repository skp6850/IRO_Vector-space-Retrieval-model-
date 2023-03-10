.. code:: ipython3

    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import jaccard_score
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Define the set of documents
    docs = ["Every year Maha Shivratri is celebrated with a lot of pomp and grandeur. It is considered to be a very special time of the year since millions of people celebrate this momentous occasion with a lot of fervour and glee.",
            "Lord Shiva devotees celebrate this occasion with a lot of grandness. It is accompanied by folk dances, songs, prayers, chants, mantras etc. This year, the beautiful occasion of Maha Shivratri will be celebrated on February 18.",
            "People keep a fast on this Maha shivratri, stay awake at night and pray to the lord for blessings, happiness, hope and prosperity. This festival holds a lot of significance and is considered to be one of the most important festivals in India.",
            "The festival of Maha Shivratri will be celebrated on February 18 and is a very auspicious festival. This Hindu festival celebrates the power of Lord Shiva. Lord Shiva protects his devotees from negative and evil spirits. He is the epitome of powerful and auspicious energy."]
    
    # Define the query
    query = "Maha Shivratri will be celebrated on February 18."
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and remove stop words from each document and the query
    docs_tokens = []
    for doc in docs:
        tokens = word_tokenize(doc.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        docs_tokens.append(tokens)
    
    query_tokens = word_tokenize(query.lower())
    query_tokens = [token for token in query_tokens if token.isalpha()]
    query_tokens = [token for token in query_tokens if token not in stop_words]
    
    # Convert the documents and the query into vectors of TF-IDF values
    vectorizer = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
    vectorizer.fit(docs_tokens)
    doc_vectors = vectorizer.transform(docs_tokens)
    query_vector = vectorizer.transform([query_tokens])
    
    # Compute cosine similarity score between the query and all documents
    cosine_sim = cosine_similarity(query_vector, doc_vectors)
    
    # Compute Jaccardian similarity score between the query and all documents
    query_tf = {}
    for term in query_tokens:
        if term in query_tf:
            query_tf[term] += 1
        else:
            query_tf[term] = 1
            
    query_tfidf = {}
    for term in query_tf:
        if term in vectorizer.vocabulary_:
            query_tfidf[term] = query_tf[term] * vectorizer.idf_[vectorizer.vocabulary_[term]]
            
    jaccardian_scores = []
    for doc_tf in doc_vectors.toarray():
        doc_tfidf = {}
        for term in vectorizer.vocabulary_:
            if doc_tf[vectorizer.vocabulary_[term]] > 0:
                doc_tfidf[term] = doc_tf[vectorizer.vocabulary_[term]] * vectorizer.idf_[vectorizer.vocabulary_[term]]
            
        intersection = set(doc_tfidf.keys()) & set(query_tfidf.keys())
        union = set(doc_tfidf.keys()) | set(query_tfidf.keys())
        jaccardian_scores.append(len(intersection)/ len(union))
    
    # Print the results
    print("Cosine similarity score:")
    for i in range(len(docs)):
        print(f"Similarity between query and doc {i}: {cosine_sim[0][i]:.2f}")
            
    print("\nJaccardian similarity score:")
    for i in range(len(docs)):
        print(f"Similarity between query and doc {i}: {jaccardian_scores[i]:.2f}")
    


.. parsed-literal::

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\skp68\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\skp68\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    C:\Users\skp68\anaconda31\lib\site-packages\sklearn\feature_extraction\text.py:516: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
      warnings.warn(
    

.. parsed-literal::

    Cosine similarity score:
    Similarity between query and doc 0: 0.18
    Similarity between query and doc 1: 0.30
    Similarity between query and doc 2: 0.10
    Similarity between query and doc 3: 0.25
    
    Jaccardian similarity score:
    Similarity between query and doc 0: 0.15
    Similarity between query and doc 1: 0.19
    Similarity between query and doc 2: 0.08
    Similarity between query and doc 3: 0.21
    

