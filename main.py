
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from components.extractor import CVProcessor, DescriptionProcessor
from components.similarity import SimilarityCalculator
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()
# HTML files will be stored in the "templates" folder
templates = Jinja2Templates(directory="templates")
# CSS files in the "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")
    
CVProcessor = CVProcessor()
DescriptionProcessor = DescriptionProcessor()
SimilarityCalculator = SimilarityCalculator()


@app.get("/")
def form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/match-html")
async def match_html(
    request: Request,
    cv_file: UploadFile = File(...),
    job_text: str = Form(...)
):
    # 1. Save the uploaded PDF to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await cv_file.read())
        cv_path = tmp.name

    # 2. Process data
    CV_schema = CVProcessor.getCVSchema(cv_path)
    Job_schema = DescriptionProcessor.GetJobSchema(job_text)

    embedded_cv, embedded_job = SimilarityCalculator.vectorizeSkills(
        CV_schema, Job_schema)
    report = SimilarityCalculator.computeSkillSimilarity(
        embedded_cv, embedded_job, threshold=0.65)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "matched_skills": report["matched_skills"],
            "unmatched_skills": report["unmatched_skills"],
            "final_score": "%.2f" % report["similarity_score"],
        }
    )
