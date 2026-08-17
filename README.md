# Resume Ranking Prototype

A small Flask prototype that ranks resume text against a job description using
transparent heuristics. It is an educational decision-support demo, not a
production applicant-tracking system, and must not be used to make automated
hiring decisions.

## What is implemented

- Flask routes for a health check, one-resume scoring, and batch ranking.
- Text cleaning and a small vocabulary-based skill extractor.
- TF-IDF cosine similarity, skill overlap, and regex-based years-of-experience
  matching combined as a weighted score.
- A response explaining skill-match, experience-match, and overall-fit values.

The score uses 60% text similarity, 30% skill overlap, and 10% experience
matching. It has not been validated against recruiters or labelled candidate
data, so it must not be interpreted as a measure of candidate quality.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

The local service listens on `http://localhost:5000`.

```bash
curl http://localhost:5000/health
```

Run the test suite with:

```bash
pytest -q
```

## API

`POST /api/rank`

```json
{
  "job_description": "Python engineer with ML experience",
  "resume": "Python developer with 3 years of ML work"
}
```

`POST /api/rank/batch` accepts the same job description and an array of named
resume-text objects.

## Not implemented

This repository does not contain Sentence-BERT, FAISS, React, Docker, AWS,
MLflow, DVC, a real resume dataset, human-recruiter evaluation, latency
benchmarks, or measured business outcomes.
