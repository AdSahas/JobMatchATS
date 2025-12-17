from pydantic import BaseModel, Field
from typing import List, Optional
import re

# Schema for CV data
class CVSchema(BaseModel):
    name: Optional[str] = Field(None, description="Candidate name if extracted.")
    email: Optional[str] = Field(None, description="Candidate email if extracted.")
    phone: Optional[str] = Field(None, description="Candidate phone if extracted.")
    skills: List[str] = Field(default_factory=list, description="Extracted skills.")
    full_text: str = Field(..., description="Cleaned full text of the CV.")

# Schema for job description data
class JobSchema(BaseModel):
    title: Optional[str] = Field(None, description="Job title extracted from the job description.")
    company: Optional[str] = Field(None, description="Company name if present.")
    skills: List[str] = Field(default_factory=list, description="Extracted skills required for the job.")
    full_text: str = Field(..., description="The full cleaned job description text.")

