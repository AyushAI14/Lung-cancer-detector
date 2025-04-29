from fastapi import FastAPI, Request, Form
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import uvicorn

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

cat_boost = joblib.load('models/cat_boost_model.pkl')


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "predicted_price": None}
    )


@app.post("/predict")
async def cancer_predict(
    request: Request,
    age: int = Form(...),
    gender: int = Form(...),
    cancer_stage: int = Form(...),
    family_history: int = Form(...),
    bmi: float = Form(...),
    cholesterol_level: int = Form(...),
    hypertension: int = Form(...),
    asthma: int = Form(...),
    cirrhosis: int = Form(...),
    other_cancer: int = Form(...),
    treatment_duration_days: int = Form(...),
    diagnosis_year: int = Form(...),
    diagnosis_month: int = Form(...),
    bmi_category: float = Form(...),
    cholesterol_cat: float = Form(...),
    comorbidity_score: int = Form(...),
    treatment_type_Combined: int = Form(...),
    treatment_type_Radiation: int = Form(...),
    treatment_type_Surgery: int = Form(...),
    smoking_status_Former_Smoker: int = Form(...),
    smoking_status_Never_Smoked: int = Form(...),
    smoking_status_Passive_Smoker: int = Form(...)
):
    if not cat_boost:
        return "Model not loaded."

    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'cancer_stage': [cancer_stage],
        'family_history': [family_history],
        'bmi': [bmi],
        'cholesterol_level': [cholesterol_level],
        'hypertension': [hypertension],
        'asthma': [asthma],
        'cirrhosis': [cirrhosis],
        'other_cancer': [other_cancer],
        'treatment_duration_days': [treatment_duration_days],
        'diagnosis_year': [diagnosis_year],
        'diagnosis_month': [diagnosis_month],
        'bmi_category': [bmi_category],
        'cholesterol_cat': [cholesterol_cat],
        'comorbidity_score': [comorbidity_score],
        'treatment_type_Combined': [treatment_type_Combined],
        'treatment_type_Radiation': [treatment_type_Radiation],
        'treatment_type_Surgery': [treatment_type_Surgery],
        'smoking_status_Former Smoker': [smoking_status_Former_Smoker],
        'smoking_status_Never Smoked': [smoking_status_Never_Smoked],
        'smoking_status_Passive Smoker': [smoking_status_Passive_Smoker]
    })

    prediction = cat_boost.predict(input_data)[0]
    status = "Positive (Cancer Detected)" if prediction == 1 else "Negative (No Cancer Detected)"

    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction": status}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
