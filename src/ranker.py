"""Resume ranking module with explainable weighted scoring."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List

from .feature_engineering.tfidf import TFIDFExtractor
from .feature_engineering.similarity import cosine_similarity
from .preprocessing.entity_extractor import extract_skills
from .preprocessing.text_cleaner import clean_text


@dataclass
class ScoreWeights:
    semantic: float = 0.6
    skills: float = 0.3
    experience: float = 0.1


class ResumeRanker:
    def __init__(self) -> None:
        self.weights = ScoreWeights()
        self.load_models()

    def load_models(self) -> None:
        self.tfidf = TFIDFExtractor()

    def normalize_score(self, raw_score: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(raw_score, 20), -20)))

    def _extract_years(self, text: str) -> int:
        match = re.search(r"(\d+)\+?\s*years?", text.lower())
        return int(match.group(1)) if match else 0

    def compute_score(self, job_description: str, resume_text: str) -> float:
        jd = clean_text(job_description)
        resume = clean_text(resume_text)

        self.tfidf.fit([jd, resume])
        jd_vec = self.tfidf.transform(jd)
        resume_vec = self.tfidf.transform(resume)
        semantic_score = max(0.0, cosine_similarity(jd_vec, resume_vec))

        jd_skills = set(extract_skills(jd))
        resume_skills = set(extract_skills(resume))
        skill_score = len(jd_skills & resume_skills) / max(1, len(jd_skills))

        jd_years = self._extract_years(jd)
        resume_years = self._extract_years(resume)
        if jd_years <= 0:
            experience_score = 1.0 if resume_years > 0 else 0.5
        else:
            experience_score = min(1.0, resume_years / jd_years)

        final_score = (
            self.weights.semantic * semantic_score
            + self.weights.skills * skill_score
            + self.weights.experience * experience_score
        )
        return float(max(0.0, min(1.0, final_score)))

    def rank_single(self, job_description: str, resume_text: str) -> float:
        return self.compute_score(job_description, resume_text)

    def rank_batch(self, job_description: str, resumes: List[str]) -> List[Dict]:
        results = []
        for idx, resume in enumerate(resumes, 1):
            score = self.compute_score(job_description, resume)
            results.append({"id": idx, "score": score, "text": resume})
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def generate_explanation(self, job_description: str, resume_text: str, resume_skills: List[str]) -> Dict:
        jd_skills = set(extract_skills(job_description))
        provided_skills = set(resume_skills) if resume_skills else set(extract_skills(resume_text))
        skill_match = len(jd_skills & provided_skills) / max(1, len(jd_skills))
        experience_match = min(1.0, self._extract_years(resume_text) / max(1, self._extract_years(job_description) or 1))
        overall_fit_raw = self.rank_single(job_description, resume_text)
        if isinstance(overall_fit_raw, dict):
            overall_fit = float(overall_fit_raw.get("score", 0.0))
        else:
            overall_fit = float(overall_fit_raw)
        return {
            "skill_match": round(skill_match, 3),
            "experience_match": round(experience_match, 3),
            "overall_fit": round(overall_fit, 3),
        }
