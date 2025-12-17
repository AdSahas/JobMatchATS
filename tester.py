from components.extractor import CVProcessor, DescriptionProcessor
from components.similarity import SimilarityCalculator

def main():
    sample_path = "sample_cv.pdf"
    sample_job_description = """We are looking for a software engineer with experience in Python, machine learning, and data analysis.
    The ideal candidate will have strong problem-solving skills and the ability to work in a team environment."""

    # Process CV
    cv_processor = CVProcessor()
    cv_schema = cv_processor.getCVSchema(sample_path)
    print("Extracted CV Schema:", cv_schema)

    # Process Job Description
    description_processor = DescriptionProcessor()
    job_schema = description_processor.GetJobSchema(sample_job_description)
    print("Extracted Job Schema:", job_schema)

    # Calculate Similarity
    similarity_calculator = SimilarityCalculator()
    embedded_cv_skills, embedded_job_skills = similarity_calculator.vectorizeSkills(cv_schema, job_schema)
    similarity_report = similarity_calculator.computeSkillSimilarity(embedded_cv_skills, embedded_job_skills)
    print("Similarity Report:", similarity_report)

if __name__ == "__main__":
    main()