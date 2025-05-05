# SHL Assessment Recommendation Engine

This project implements a modular and extensible system to recommend SHL assessments based on job descriptions. The system utilizes intelligent Natural Language Processing (NLP) techniques like TF-IDF, BERT embeddings, and a FastAPI backend for delivering recommendations. It also includes a frontend in HTML/JS and evaluation metrics (Recall@K and MAP@K).

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Backend (FastAPI)](#backend-fastapi)
- [Frontend (HTML/JS)](#frontend-htmljs)
- [Evaluation Metrics](#evaluation-metrics)
- [Sample Queries](#sample-queries)
- [SHL Catalog](#shl-catalog)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The SHL Assessment Recommendation Engine suggests the most relevant SHL assessments based on job descriptions for various roles. The project is split into three main parts:

1. **Part 1**: Intelligent NLP-based Assessment Recommendation System  
2. **Part 2**: API for fetching recommendations with frontend  
3. **Part 3**: Evaluation Module with metrics like Recall@K and MAP@K  

## Features

- **Intelligent Recommendations**: TF-IDF, BERT embeddings, and fuzzy matching.  
- **Modular System**: Easily extendable with new assessments or models.  
- **Backend API**: FastAPI serving JSON responses.  
- **Frontend**: Simple HTML/JS interface.  
- **Evaluation**: Recall@K and MAP@K metrics.  
- **Comprehensive Catalog**: Coverage for Java Dev, Data Scientist, COO, Content Writer, etc.

## Installation

### Prerequisites

- Python 3.x  
- FastAPI  
- SpaCy  
- HuggingFace’s `transformers` (for BERT)  
- Uvicorn  
- Browser (for frontend)

### Steps to Install

1. **Clone the repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/shl-assessment-recommendation.git
   cd shl-assessment-recommendation

## Create a Virtual Environment

To create a virtual environment, run the following command:

```bash
python -m venv venv
```

### Activate the Virtual Environment

**For macOS/Linux:**

```bash
source venv/bin/activate
```

**For Windows:**

```bash
venv\Scripts\activate
```

## Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download NLP Models

**SpaCy model:**

```bash
python -m spacy download en_core_web_md
```

**(Optional) Transformers for BERT:**

```bash
pip install transformers
```

## Run the FastAPI Backend

Start the backend server:

```bash
uvicorn app.main:app --reload
```

## Open the Frontend

Open `frontend/index.html` in your browser to interact with the frontend.

## Usage

### Backend API

**Endpoint:** `POST /recommendations`

**Request body (JSON):**

```json
{
  "job_description": "Java Developer with 3 years backend experience."
}
```

**Response:**

```json
{
  "recommended_assessments": [
    "Java Coding Test",
    "Backend Development Skills Assessment"
  ]
}
```

### Frontend

- Enter a job description.
- Click "Get Recommendations".
- View results in the table.

## Folder Structure

```bash
shl-assessment-recommendation/
│
├── app/
│   ├── main.py           # FastAPI entrypoint
│   ├── models.py         # Pydantic models
│   └── utils.py          # NLP helpers
│
├── frontend/
│   ├── index.html        # UI
│   ├── script.js         # AJAX calls
│   └── style.css         # Styles
│
├── data/
│   ├── sample_queries.json  # For evaluation
│   └── shl_catalog.json     # Assessment catalog
│
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Backend (FastAPI)

- Endpoint: `/recommendations`
- Vectorizes the input via TF-IDF and/or BERT embeddings.
- Returns top-K matching assessments in JSON.

## Frontend (HTML/JS)

- Simple form to input job description.
- AJAX call to backend.
- Displays results in a table.

## Evaluation Metrics

- **Recall@K**: Fraction of relevant assessments in top-K.
- **MAP@K**: Mean average precision of top-K.
- Implemented in `evaluation.py`, using entries from `sample_queries.json`.

## Sample Queries

Stored in `data/sample_queries.json`:

```json
[
  {
    "job_description": "Software engineer with 5 years of Python backend experience.",
    "relevant_assessments": [
      "Python Coding Test",
      "Backend Development Skills Assessment"
    ]
  }
]
```

## SHL Catalog

Stored in `data/shl_catalog.json`:

```json
[
  {
    "assessment_id": 1,
    "assessment_name": "Java Coding Test",
    "relevant_skills": ["Java", "Backend Development"]
  },
  {
    "assessment_id": 2,
    "assessment_name": "Python Coding Test",
    "relevant_skills": ["Python", "Software Development"]
  }
]
```

## Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature-name`
3. Commit: `git commit -am "Add feature"`
4. Push: `git push origin feature-name`
5. Open a Pull Request

## License

Licensed under the MIT License. See `LICENSE` for details.
