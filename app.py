"""Flask app for AI resume ranking."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from src.preprocessing.entity_extractor import extract_skills
from src.ranker import ResumeRanker

app = Flask(__name__)
ranker = ResumeRanker()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/api/rank")
def rank_single_endpoint():
    data = request.get_json(silent=True) or {}
    job_description = data.get("job_description")
    resume = data.get("resume")
    if not job_description or not resume:
        return jsonify({"error": "job_description and resume are required"}), 400

    score_result = ranker.rank_single(job_description, resume)
    if isinstance(score_result, dict):
        return jsonify(score_result)
    explanation = ranker.generate_explanation(job_description, resume, extract_skills(resume))
    return jsonify({"score": score_result, "rank": 1, "explanation": explanation})


@app.post("/api/rank/batch")
def rank_batch_endpoint():
    data = request.get_json(silent=True) or {}
    job_description = data.get("job_description")
    resumes = data.get("resumes")
    if not job_description or not resumes:
        return jsonify({"error": "job_description and resumes are required"}), 400

    resume_texts = [item.get("text", "") for item in resumes if isinstance(item, dict)]
    ranked = ranker.rank_batch(job_description, resume_texts)

    response = []
    for idx, item in enumerate(ranked, 1):
        if "id" in item:
            original = resumes[item["id"] - 1]
            name = original.get("name", f"Candidate {idx}")
        else:
            name = item.get("name", f"Candidate {idx}")
        response.append({"name": name, "score": item.get("score", 0.0)})
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
