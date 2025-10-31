"""Unit tests for Resume Ranking functionality."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd


class TestResumePreprocessing(unittest.TestCase):
    """Test cases for resume preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_resume = """
        John Doe
        Email: john@example.com
        Phone: 123-456-7890
        
        Experience:
        - 5 years as Python Developer at Tech Corp
        - Worked on ML projects using TensorFlow and PyTorch
        
        Education:
        - BS in Computer Science, MIT
        
        Skills: Python, Machine Learning, NLP, Docker, AWS
        """
        
        self.sample_job_desc = """
        We are looking for a Senior Python Developer with:
        - 3+ years of Python experience
        - Machine Learning and NLP expertise
        - Experience with cloud platforms (AWS/GCP)
        - Strong problem-solving skills
        """
    
    def test_text_cleaning(self):
        """Test text cleaning and normalization."""
        from src.preprocessing.text_cleaner import clean_text
        
        dirty_text = "  This\t\thas   extra\n\nspaces  "
        cleaned = clean_text(dirty_text)
        
        self.assertNotIn('\t', cleaned)
        self.assertNotIn('  ', cleaned)
        self.assertEqual(cleaned.strip(), cleaned)
    
    def test_email_extraction(self):
        """Test extraction of email addresses."""
        from src.preprocessing.entity_extractor import extract_email
        
        email = extract_email(self.sample_resume)
        self.assertEqual(email, "john@example.com")
    
    def test_phone_extraction(self):
        """Test extraction of phone numbers."""
        from src.preprocessing.entity_extractor import extract_phone
        
        phone = extract_phone(self.sample_resume)
        self.assertIn("123-456-7890", phone or "")
    
    def test_skill_extraction(self):
        """Test skill extraction from resume."""
        from src.preprocessing.entity_extractor import extract_skills
        
        skills = extract_skills(self.sample_resume)
        
        self.assertIsInstance(skills, list)
        self.assertIn("Python", skills)
        self.assertIn("Machine Learning", skills)
        self.assertTrue(len(skills) > 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering."""
    
    def test_embedding_generation(self):
        """Test generation of text embeddings."""
        from src.feature_engineering.embeddings import generate_embedding
        
        text = "This is a sample text for embedding"
        embedding = generate_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)  # 1D array
        self.assertTrue(embedding.shape[0] > 0)  # Non-empty
    
    def test_similarity_score(self):
        """Test cosine similarity calculation."""
        from src.feature_engineering.similarity import cosine_similarity
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        sim_identical = cosine_similarity(vec1, vec2)
        sim_different = cosine_similarity(vec1, vec3)
        
        self.assertAlmostEqual(sim_identical, 1.0, places=5)
        self.assertAlmostEqual(sim_different, 0.0, places=5)
    
    def test_tfidf_features(self):
        """Test TF-IDF feature extraction."""
        from src.feature_engineering.tfidf import TFIDFExtractor
        
        documents = [
            "python machine learning developer",
            "java backend engineer",
            "python data scientist"
        ]
        
        extractor = TFIDFExtractor()
        extractor.fit(documents)
        features = extractor.transform(documents[0])
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(features.shape[0] > 0)


class TestResumeRanker(unittest.TestCase):
    """Test cases for resume ranking logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_resumes = [
            {
                'name': 'John Doe',
                'text': 'Python developer with 5 years ML experience',
                'skills': ['Python', 'ML', 'TensorFlow']
            },
            {
                'name': 'Jane Smith',
                'text': 'Java developer with 3 years backend experience',
                'skills': ['Java', 'Spring', 'SQL']
            },
            {
                'name': 'Bob Johnson',
                'text': 'Python ML engineer with NLP expertise',
                'skills': ['Python', 'NLP', 'PyTorch', 'ML']
            }
        ]
        
        self.job_description = "Looking for Python ML engineer with NLP experience"
    
    @patch('src.ranker.ResumeRanker.load_models')
    def test_ranker_initialization(self, mock_load):
        """Test ranker initialization."""
        from src.ranker import ResumeRanker
        
        ranker = ResumeRanker()
        self.assertIsNotNone(ranker)
        mock_load.assert_called_once()
    
    @patch('src.ranker.ResumeRanker.compute_score')
    def test_single_resume_ranking(self, mock_score):
        """Test ranking of a single resume."""
        from src.ranker import ResumeRanker
        
        mock_score.return_value = 0.85
        ranker = ResumeRanker()
        
        score = ranker.rank_single(
            self.job_description,
            self.sample_resumes[0]['text']
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    @patch('src.ranker.ResumeRanker.compute_score')
    def test_batch_ranking(self, mock_score):
        """Test batch ranking of multiple resumes."""
        from src.ranker import ResumeRanker
        
        # Mock different scores for different resumes
        mock_score.side_effect = [0.85, 0.45, 0.92]
        ranker = ResumeRanker()
        
        results = ranker.rank_batch(
            self.job_description,
            [r['text'] for r in self.sample_resumes]
        )
        
        self.assertEqual(len(results), len(self.sample_resumes))
        # Results should be sorted by score (descending)
        self.assertTrue(results[0]['score'] >= results[1]['score'])
        self.assertTrue(results[1]['score'] >= results[2]['score'])
    
    def test_score_normalization(self):
        """Test that scores are properly normalized to [0, 1]."""
        from src.ranker import ResumeRanker
        
        ranker = ResumeRanker()
        raw_score = 15.7
        normalized = ranker.normalize_score(raw_score)
        
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
    
    def test_explanation_generation(self):
        """Test generation of ranking explanations."""
        from src.ranker import ResumeRanker
        
        ranker = ResumeRanker()
        resume = self.sample_resumes[0]
        
        explanation = ranker.generate_explanation(
            self.job_description,
            resume['text'],
            resume['skills']
        )
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('skill_match', explanation)
        self.assertIn('experience_match', explanation)
        self.assertIn('overall_fit', explanation)


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        from app import app
        self.app = app
        self.client = self.app.test_client()
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
    
    @patch('src.ranker.ResumeRanker.rank_single')
    def test_rank_single_endpoint(self, mock_rank):
        """Test single resume ranking endpoint."""
        mock_rank.return_value = {
            'score': 0.85,
            'rank': 1,
            'explanation': {'skill_match': 0.9}
        }
        
        payload = {
            'job_description': 'Looking for Python developer',
            'resume': 'Python developer with 5 years experience'
        }
        
        response = self.client.post('/api/rank', json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('score', data)
        self.assertGreaterEqual(data['score'], 0.0)
    
    def test_rank_missing_parameters(self):
        """Test error handling for missing parameters."""
        payload = {'job_description': 'Looking for developer'}
        # Missing 'resume' field
        
        response = self.client.post('/api/rank', json=payload)
        
        self.assertEqual(response.status_code, 400)
    
    @patch('src.ranker.ResumeRanker.rank_batch')
    def test_batch_ranking_endpoint(self, mock_batch):
        """Test batch ranking endpoint."""
        mock_batch.return_value = [
            {'name': 'John', 'score': 0.85},
            {'name': 'Jane', 'score': 0.72}
        ]
        
        payload = {
            'job_description': 'Looking for Python developer',
            'resumes': [
                {'name': 'John', 'text': 'Python expert'},
                {'name': 'Jane', 'text': 'Java developer'}
            ]
        }
        
        response = self.client.post('/api/rank/batch', json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance and accuracy."""
    
    def test_ranking_consistency(self):
        """Test that ranking is consistent across multiple runs."""
        from src.ranker import ResumeRanker
        
        ranker = ResumeRanker()
        job_desc = "Python ML engineer needed"
        resume = "Experienced Python ML engineer with 5 years"
        
        # Rank the same resume multiple times
        scores = [ranker.rank_single(job_desc, resume) for _ in range(5)]
        
        # All scores should be identical (deterministic)
        self.assertTrue(all(s == scores[0] for s in scores))
    
    def test_ranking_order(self):
        """Test that better matches get higher scores."""
        from src.ranker import ResumeRanker
        
        ranker = ResumeRanker()
        job_desc = "Looking for Python ML engineer with NLP experience"
        
        # Perfect match
        resume1 = "Python ML engineer with 5 years NLP experience"
        # Partial match
        resume2 = "Python developer with some ML knowledge"
        # Poor match
        resume3 = "Java backend developer"
        
        score1 = ranker.rank_single(job_desc, resume1)
        score2 = ranker.rank_single(job_desc, resume2)
        score3 = ranker.rank_single(job_desc, resume3)
        
        self.assertGreater(score1, score2)
        self.assertGreater(score2, score3)


if __name__ == '__main__':
    unittest.main()
