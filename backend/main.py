import os
import json
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.metrics.metrics import recall_at_k, mean_average_precision_at_k
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Load SHL catalog data
with open("backend/SHL_CATALOG.json") as file:
    SHL_CATALOG = json.load(file)

# Relevant items map for various job roles
RELEVANT_ITEMS_MAP = {
    "java_developer": {0, 6, 5},
    "sales": {2, 4, 7},
    "coo": {3, 6},
    "content_writer": {4, 3},
    "data_associate": {8, 9},
    "graduate_sales": {2, 7, 10},
}

# Request model for Job Description
class JobDescriptionRequest(BaseModel):
    jd_text: str

# Preprocessing function to lemmatize and remove stop words and punctuation
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function to infer job role from the job description text
def infer_job_role(original_text):
    text = original_text.lower()

    job_roles_keywords = {
        "java_developer": ["java", "developer", "backend", "spring", "hibernate"],
        "sales": ["sales", "marketing", "client", "business development", "lead"],
        "graduate_sales": ["graduate", "entry level", "fresher", "trainee", "sales"],
        "coo": ["chief operating officer", "operations", "management", "executive", "coo"],
        "content_writer": ["content", "writing", "editor", "copywriter", "creative"],
        "data_associate": ["data", "analyst", "python", "statistics", "data science"],
    }

    for role, keywords in job_roles_keywords.items():
        if any(keyword in text for keyword in keywords):
            return role
    return "unknown"

# Route to handle job description and return recommendations
@app.post("/recommendations")
def get_recommendations(request: JobDescriptionRequest):
    job_role = infer_job_role(request.jd_text)

    if job_role not in RELEVANT_ITEMS_MAP:
        raise HTTPException(status_code=400, detail="Unable to infer job role from the description")

    # Preprocess the job description text
    jd_text_processed = preprocess_text(request.jd_text)

    # Retrieve relevant items for the inferred job role
    relevant_items = RELEVANT_ITEMS_MAP[job_role]
    recommendations = []

    # Prepare catalog descriptions
    catalog_descriptions = [assessment.get('tags', '') for assessment in SHL_CATALOG]

    # Perform TF-IDF vectorization and cosine similarity calculation
    vectorizer = TfidfVectorizer()
    corpus = [jd_text_processed] + catalog_descriptions
    vectors = vectorizer.fit_transform(corpus)

    # Calculate cosine similarities between job description and catalog descriptions
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Attach relevance scores to catalog items
    for i, score in enumerate(cosine_similarities):
        SHL_CATALOG[i]['relevance_score'] = float(score)
        recommendations.append(SHL_CATALOG[i])

    # Sort recommendations based on relevance score
    recommendations = sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)

    # Set the top-k recommendations
    k = 5
    recommended_indices = list(range(len(recommendations)))

    # Calculate evaluation metrics
    recall = recall_at_k(recommended_indices, relevant_items, k)
    mapk = mean_average_precision_at_k(recommended_indices, relevant_items, k)

    return {
        "recommendations": recommendations[:k],
        "metrics": {
            "Recall@K": recall,
            "MAP@K": mapk
        }
    }

# Entry point for running the app with dynamic port binding
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
