import re
import os
import json
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from components.schemas import CVSchema, JobSchema

load_dotenv()


class CVProcessor:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def extractCV(self, pdf_path) -> str:
        """
        Extract the text from CV in a PDF format.
        """
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip().lower()

    def cleanText(self, raw_text: str) -> str:
        """
        Clean the extracted text by removing extra spaces and non-ASCII characters.
        """
        text = re.sub(
            r'\s+', ' ', raw_text)  # Replace multiple spaces with a single space
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def processCV(self, processedText: str) -> CVSchema:
        prompt = f"""
        Extract structured resume information from the following CV text.

        Return JSON with exactly these fields:
        - name (string)
        - email (string or null)
        - phone (string or null)
        - skills (list of strings)
        - full_text (string)

        Only return JSON. No explanations.

        CV Text:
        {processedText}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        return CVSchema(**data)

    def getCVSchema(self, pdf_path: str) -> CVSchema:
        raw_text = self.extractCV(pdf_path)
        cleaned_text = self.cleanText(raw_text)
        cv_data = self.processCV(cleaned_text)
        return cv_data


class DescriptionProcessor:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def cleanText(self, raw_text: str) -> str:
        """
        Clean the extracted text by removing extra spaces and non-ASCII characters.
        """
        text = re.sub(
            r'\s+', ' ', raw_text)  # Replace multiple spaces with a single space
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def processDescription(self, jobDescription: str) -> dict:
        prompt = f"""
        Extract structured job description information from the following text.

        Return JSON with exactly these fields:
        - title (string or null)
        - company (string or null)
        - skills (list of strings)
        - full_text (string)

        Only return JSON. No explanations.

        Job Description Text:
        {jobDescription}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        return JobSchema(**data)
    
    def GetJobSchema(self, jobDescription: str) -> JobSchema:
        cleaned_text = self.cleanText(jobDescription)
        job_data = self.processDescription(cleaned_text)
        return job_data
    

