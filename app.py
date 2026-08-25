"""Flask app for AI resume ranking."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.preprocessing.entity_extractor import extract_skills
from src.ranker import ResumeRanker

app = Flask(__name__)
ranker = ResumeRanker()

MAX_BATCH_SIZE = 100
MAX_TEXT_LENGTH = 50000


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

    if not isinstance(job_description, str) or not isinstance(resume, str):
        return jsonify({"error": "job_description and resume must be strings"}), 400

    if len(job_description) > MAX_TEXT_LENGTH or len(resume) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text payload exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters"}), 400

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

    if not isinstance(job_description, str):
        return jsonify({"error": "job_description must be a string"}), 400

    if len(job_description) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"job_description exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters"}), 400

    if not isinstance(resumes, list):
        return jsonify({"error": "resumes must be a list"}), 400

    if len(resumes) > MAX_BATCH_SIZE:
        return jsonify({"error": f"resumes count exceeds maximum allowed batch limit of {MAX_BATCH_SIZE}"}), 400

    for item in resumes:
        if isinstance(item, dict):
            text = item.get("text", "")
            if isinstance(text, str) and len(text) > MAX_TEXT_LENGTH:
                return jsonify({"error": f"Resume text exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters"}), 400

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
    app.run(host="0.0.0.0", port=5000, debug=False)
