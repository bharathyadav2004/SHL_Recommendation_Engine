import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from metrics.metrics import recall_at_k, mean_average_precision_at_k

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

# Load the SHL_CATALOG
def load_catalog():
    try:
        with open("backend/SHL_CATALOG.json") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError("Catalog file not found")
    except json.JSONDecodeError:
        raise ValueError("Error decoding the catalog file")

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Recommendation function
def get_recommendations(jd_text, job_role, relevant_items_map, k=5):
    SHL_CATALOG = load_catalog()
    jd_text = preprocess_text(jd_text)
    job_role = job_role.lower()

    if job_role not in relevant_items_map:
        raise ValueError("Invalid job role")

    # Get only relevant item indices for the job_role
    relevant_indices = relevant_items_map[job_role]
    filtered_catalog = [SHL_CATALOG[i] for i in relevant_indices]
    catalog_descriptions = [item["tags"] for item in filtered_catalog]

    # TF-IDF over only relevant items
    vectorizer = TfidfVectorizer()
    corpus = [jd_text] + catalog_descriptions
    vectors = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Assign relevance scores and prepare recommendations
    recommendations = []
    for i, score in enumerate(similarities):
        assessment = filtered_catalog[i]
        assessment["relevance_score"] = score
        recommendations.append(assessment)

    # Sort by relevance
    recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Convert back to original indices for evaluation
    recommended_indices = [relevant_indices[i] for i, _ in enumerate(recommendations[:k])]

    recall = recall_at_k(recommended_indices, relevant_indices, k)
    mapk = mean_average_precision_at_k(recommended_indices, relevant_indices, k)

    return recommendations[:k], recall, mapk

