import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from components.schemas import CVSchema, JobSchema


class SimilarityCalculator:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between both vectors.
        """
        # calculate the norms of the vectors
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            raise ValueError("One or both of the vectors are zero vectors, cannot compute cosine similarity. This may be due to invalid or empty input text.");
        
        dot_product = np.dot(vec1, vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return float(cosine_similarity)
    
    def embed_text(self, text: str):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def vectorizeSkills(self, cv: CVSchema, job: JobSchema):
        # Get skills and remove duplicates
        cv_skill_list = list(set(cv.skills))
        job_skill_list = list(set(job.skills))

        # Return the embedded skills and their vectors
        embedded_cv_skills = {skill: self.embed_text(skill) for skill in cv_skill_list}
        embedded_job_skills = {skill: self.embed_text(skill) for skill in job_skill_list}
        return embedded_cv_skills, embedded_job_skills

    def computeSkillSimilarity(self, cv_skills: dict, job_skills: dict, threshold: float = 0.75) -> float:
        matched = []
        unmatched = []

        # naive n^2 comparison - suffices for a realistically long list of skills. 
        for cv_skill, cv_vec in cv_skills.items():
            for job_skill, job_vec in job_skills.items():
                similarity = self.cosine_similarity(np.array(cv_vec), np.array(job_vec))
                if similarity >= threshold:
                    matched.append((cv_skill, job_skill, similarity))
                else:
                    unmatched.append((job_skill, similarity))

        # Generate similarity report
        job_set = set(job_skills.keys())
        matched_set = set([m[1] for m in matched])
        unmatched_skills = job_set - matched_set
        total_skills = len(matched_set) + len(unmatched_skills)
        matched_skills = len(matched_set)
        similarity_score = matched_skills / total_skills if total_skills > 0 else 0.0

        similarity_report = {
            "matched_skills": matched,
            "unmatched_skills": list(unmatched_skills),
            "similarity_score": similarity_score
        }
        return similarity_report




    

