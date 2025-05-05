from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.metrics.metrics import recall_at_k, mean_average_precision_at_k
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")

with open("backend/SHL_CATALOG.json") as file:
    SHL_CATALOG = json.load(file)

RELEVANT_ITEMS_MAP = {
    "java_developer": {0, 6, 5},
    "sales": {2, 4, 7},
    "coo": {3, 6},
    "content_writer": {4, 3},
    "data_associate": {8, 9},
    "graduate_sales": {2, 7, 10},
}

class JobDescriptionRequest(BaseModel):
    jd_text: str

def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

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

@app.post("/recommendations")
def get_recommendations(request: JobDescriptionRequest):
    job_role = infer_job_role(request.jd_text)

    if job_role not in RELEVANT_ITEMS_MAP:
        raise HTTPException(status_code=400, detail="Unable to infer job role from the description")

    jd_text_processed = preprocess_text(request.jd_text)

    relevant_items = RELEVANT_ITEMS_MAP[job_role]
    recommendations = []

    catalog_descriptions = [assessment.get('tags', '') for assessment in SHL_CATALOG]

    vectorizer = TfidfVectorizer()
    corpus = [jd_text_processed] + catalog_descriptions
    vectors = vectorizer.fit_transform(corpus)

    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(cosine_similarities):
        SHL_CATALOG[i]['relevance_score'] = float(score)
        recommendations.append(SHL_CATALOG[i])

    recommendations = sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)

    k = 5
    recommended_indices = list(range(len(recommendations)))

    recall = recall_at_k(recommended_indices, relevant_items, k)
    mapk = mean_average_precision_at_k(recommended_indices, relevant_items, k)

    return {
        "recommendations": recommendations[:k],
        "metrics": {
            "Recall@K": recall,
            "MAP@K": mapk
        }
    }
