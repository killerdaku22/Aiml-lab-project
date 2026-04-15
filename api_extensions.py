"""
api_extensions.py
-----------------
Handles web scraping of job descriptions and uses the Gemini API
to generate a dynamic resume review against that job description.
"""

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Configure Gemini with the provided API key
GEMINI_API_KEY = "AIzaSyDTbvzpt63V69g8uyyOQYObX6P0tMOMTUQ"
genai.configure(api_key=GEMINI_API_KEY)


def scrape_job_description(url: str) -> str:
    """
    Scrapes the text content from a given job URL.
    Returns the cleaned text showing the job requirements.
    
    Parameters:
    -----------
    url : str
        The web address to scrape.
        
    Returns:
    --------
    str
        The extracted page text.
    """
    try:
        # A simple header to bypass basic bot blockers
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"[Error] Failed to read URL (Status Code: {response.status_code}). Some websites actively block automated scripts."
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        # Get raw text
        text = soup.get_text(separator=' ')
        
        # Clean up excessive newlines/spaces
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:3000]  # Cap at 3000 characters to prevent massive overload
    
    except Exception as e:
        return f"[Error] Could not scrape {url}: {e}"


def gemini_resume_review(resume_text: str, job_text: str = None) -> str:
    """
    Uses Google's Gemini API to generate a professional review
    of the resume, optionally comparing it against a job description.
    
    Parameters:
    -----------
    resume_text : str
        The candidate's resume.
    job_text : str
        The scraped job description (optional).
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        
        if job_text and "[Error]" not in job_text:
            prompt = f"""
            You are an expert HR Recruiter and Career Coach. 
            I am going to give you a candidate's resume, and the Job Description they are applying for.
            
            1. Rate how well their resume matches the job (Percentage).
            2. List 3 strong points where they perfectly match.
            3. List 2 missing skills or weak points they should add.
            
            JOB DESCRIPTION:
            {job_text}
            
            CANDIDATE RESUME:
            {resume_text}
            """
        else:
            prompt = f"""
            You are an expert HR Recruiter and Career Coach. 
            Review the following resume and provide:
            1. An overall professional impression.
            2. Top 3 strengths.
            3. 2 actionable pieces of advice to improve it.
            
            CANDIDATE RESUME:
            {resume_text}
            """
            
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"🚨 API Error: Make sure your Gemini API key is valid. Detail: {e}"
