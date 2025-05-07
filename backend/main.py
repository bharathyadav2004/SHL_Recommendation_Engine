import os
import json
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.metrics.metrics import recall_at_k, mean_average_precision_at_k
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://celebrated-malabi-83bb5f.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load SHL catalog data
with open("backend/SHL_CATALOG.json") as file:
    SHL_CATALOG = json.load(file)

# Mapping job roles to relevant SHL catalog indices
RELEVANT_ITEMS_MAP = {
    "java_developer": {0, 6, 5},
    "sales": {2, 4, 7},
    "coo": {3, 6},
    "content_writer": {4, 3},
    "data_associate": {8, 9},
    "graduate_sales": {2, 7, 10},
}

# Pydantic model for API input
class JobDescriptionRequest(BaseModel):
    jd_text: str

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Infer job role from JD
def infer_job_role(text):
    text = text.lower()
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

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>🚀 SHL Assessment Recommendation Engine</h2>
    <p>Use <a href="/docs">/docs</a> to explore the API.</p>
    """

@app.post("/recommendations")
def get_recommendations(request: JobDescriptionRequest):
    jd_original = request.jd_text.strip()
    print("Received request:", jd_original) 
    if not jd_original:
        raise HTTPException(status_code=400, detail="Job description is empty.")

    job_role = infer_job_role(jd_original)

    if job_role not in RELEVANT_ITEMS_MAP:
        raise HTTPException(status_code=400, detail="Unable to infer job role from the description.")

    # Preprocess JD
    jd_text_processed = preprocess_text(jd_original)
    if not jd_text_processed:
        raise HTTPException(status_code=400, detail="Job description has no meaningful content after preprocessing.")

    relevant_items = RELEVANT_ITEMS_MAP[job_role]
    recommendations = []

    # Prepare catalog tags (fallback if empty)
    catalog_descriptions = [
        preprocess_text(item.get('tags', '') or "placeholder") for item in SHL_CATALOG
    ]

    # Combine JD with catalog
    corpus = [jd_text_processed] + catalog_descriptions

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(corpus)
    except ValueError:
        raise HTTPException(status_code=500, detail="Failed to generate vector due to empty vocabulary.")

    # Cosine similarity
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(cosine_similarities):
        SHL_CATALOG[i]['relevance_score'] = float(score)
        recommendations.append(SHL_CATALOG[i])

    recommendations = sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)

    k = 5
    recommended_indices = [SHL_CATALOG.index(rec) for rec in recommendations[:k]]

    # Evaluation metrics
    recall = recall_at_k(recommended_indices, relevant_items, k)
    mapk = mean_average_precision_at_k(recommended_indices, relevant_items, k)

    return {
        "recommendations": recommendations[:k],
        "metrics": {
            "Recall@K": recall,
            "MAP@K": mapk
        }
    }

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
