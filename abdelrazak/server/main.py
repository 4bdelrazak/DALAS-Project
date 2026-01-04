from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import json
import joblib

# Paths
SERVER_DIR = Path(__file__).parent
MODEL_DIR = SERVER_DIR / "model"  # Models folder inside server

# Disease info
DISEASES = ['asthma', 'copd', 'lri', 'lung_cancer']
DISEASE_NAMES = {
    'asthma': 'Asthma',
    'copd': 'COPD (Chronic Obstructive Pulmonary Disease)',
    'lri': 'Lower Respiratory Infections',
    'lung_cancer': 'Lung Cancer'
}

# Age risk multipliers (epidemiological data)
AGE_RISK_FACTORS = {
    (0, 18): 0.3,      # Children/teens
    (18, 30): 0.5,     # Young adults
    (30, 45): 0.8,     # Adults
    (45, 60): 1.2,     # Middle-aged
    (60, 75): 1.8,     # Seniors
    (75, 120): 2.5     # Elderly
}

# Cache
models_cache = {}
country_data_cache = None


def load_model(model_type: str = "shortterm"):
    """Load model from predict_disease_risk folder."""
    cache_key = f"model_{model_type}"
    if cache_key in models_cache:
        return models_cache[cache_key]
    
    try:
        if model_type == "longterm":
            model_path = MODEL_DIR / "model_longterm.joblib"
        else:
            model_path = MODEL_DIR / "model_shortterm.joblib"
        
        model_data = joblib.load(model_path)
        models_cache[cache_key] = model_data
        print(f"✓ Loaded {model_type} model from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        return None


def load_country_data():
    """Load environmental data for all countries from pre-generated JSON."""
    global country_data_cache
    if country_data_cache is not None:
        return country_data_cache
    
    try:
        json_path = SERVER_DIR / "country_data.json"
        with open(json_path, 'r') as f:
            country_data_cache = json.load(f)
        print(f"✓ Loaded data for {len(country_data_cache)} countries")
        return country_data_cache
    except Exception as e:
        print(f"✗ Error loading country data: {e}")
        return {}


def get_age_factor(age: int) -> float:
    """Get risk multiplier based on age."""
    for (min_age, max_age), factor in AGE_RISK_FACTORS.items():
        if min_age <= age < max_age:
            return factor
    return 1.0


def get_risk_level(mortality_rate: float) -> str:
    """Categorize mortality rate into risk level."""
    if mortality_rate < 5:
        return "Very Low"
    elif mortality_rate < 15:
        return "Low"
    elif mortality_rate < 35:
        return "Moderate"
    elif mortality_rate < 70:
        return "High"
    return "Very High"


def get_risk_description(disease: str, risk_level: str) -> str:
    """Get description for risk level."""
    descriptions = {
        "Very Low": f"Your risk of {DISEASE_NAMES[disease]} is minimal based on your location and demographics.",
        "Low": f"You have a relatively low risk of {DISEASE_NAMES[disease]}. Maintain healthy habits.",
        "Moderate": f"Moderate risk detected. Consider regular health checkups for {DISEASE_NAMES[disease]}.",
        "High": f"Higher than average risk for {DISEASE_NAMES[disease]}. Consult a healthcare provider.",
        "Very High": f"Significant risk for {DISEASE_NAMES[disease]}. Please seek medical advice."
    }
    return descriptions.get(risk_level, "")


# ============== Pydantic Models ==============

class UserInput(BaseModel):
    country: str = Field(..., description="Country name")
    gender: Literal["male", "female"] = Field(..., description="Gender")
    age: int = Field(..., ge=0, le=120, description="Age in years")


class DiseaseRisk(BaseModel):
    disease: str
    disease_name: str
    base_mortality_rate: float
    adjusted_mortality_rate: float
    risk_level: str
    description: str


class PredictionResponse(BaseModel):
    country: str
    gender: str
    age: int
    age_risk_factor: float
    model_used: str
    model_year_range: str
    predictions: list[DiseaseRisk]
    environmental_summary: dict


class CountryListResponse(BaseModel):
    countries: list[str]
    total: int


class ModelInfoResponse(BaseModel):
    name: str
    description: str
    year_range: str
    n_features: int
    overall_r2: float
    features_type: str


# ============== FastAPI App ==============

app = FastAPI(
    title="Respiratory Disease Risk Prediction API",
    description="""
    Predicts respiratory disease risk based on location, gender, and age.
    
    **Two models available:**
    - `shortterm`: 2003-2023, includes air pollution data (default, recommended)
    - `longterm`: 1980-2023, meteorological data only
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Pre-load models and data on startup."""
    load_model("shortterm")
    load_model("longterm")
    load_country_data()


@app.get("/")
async def health_check():
    """Health check endpoint."""
    shortterm = load_model("shortterm")
    longterm = load_model("longterm")
    countries = load_country_data()
    
    return {
        "status": "healthy",
        "models": {
            "shortterm": shortterm is not None,
            "longterm": longterm is not None
        },
        "countries_available": len(countries)
    }


@app.get("/models")
async def list_models():
    """Get information about available models."""
    models_info = []
    
    shortterm = load_model("shortterm")
    if shortterm:
        models_info.append(ModelInfoResponse(
            name="shortterm",
            description=shortterm.get('description', 'Short-term model with pollution data'),
            year_range=f"{shortterm['year_range'][0]}-{shortterm['year_range'][1]}",
            n_features=shortterm['n_features'],
            overall_r2=round(shortterm['overall_r2'], 3),
            features_type="Meteorological + Pollutants"
        ))
    
    longterm = load_model("longterm")
    if longterm:
        models_info.append(ModelInfoResponse(
            name="longterm",
            description=longterm.get('description', 'Long-term model with meteorological data'),
            year_range=f"{longterm['year_range'][0]}-{longterm['year_range'][1]}",
            n_features=longterm['n_features'],
            overall_r2=round(longterm['overall_r2'], 3),
            features_type="Meteorological Only"
        ))
    
    return {"models": models_info}


@app.get("/countries", response_model=CountryListResponse)
async def get_countries():
    """Get list of available countries."""
    countries = load_country_data()
    country_list = sorted(countries.keys())
    return CountryListResponse(countries=country_list, total=len(country_list))


@app.get("/diseases")
async def get_diseases():
    """Get list of diseases the model can predict."""
    return {
        "diseases": DISEASE_NAMES,
        "note": "Predictions are gender-specific (Male/Female mortality rates)"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    user_input: UserInput,
    model_type: Literal["shortterm", "longterm"] = Query(
        default="shortterm",
        description="Model to use: 'shortterm' (2003-2023, with pollution) or 'longterm' (1980-2023, meteorological only)"
    )
):
    """
    Predict respiratory disease risk based on location, gender, and age.
    
    - **country**: Your country of residence
    - **gender**: male or female
    - **age**: Your age in years
    - **model_type**: Which model to use (shortterm recommended)
    """
    # Load model
    model_data = load_model(model_type)
    if model_data is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_type}' not loaded. Please train the model first."
        )
    
    # Load country data
    country_data = load_country_data()
    if user_input.country not in country_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Country '{user_input.country}' not found. Use /countries endpoint to see available countries."
        )
    
    # Get environmental data
    env_data = country_data[user_input.country]
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    targets = model_data['targets']
    
    # Build feature vector
    X = np.array([[env_data.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)
    
    # Predict
    all_predictions = model.predict(X_scaled)[0]
    
    # Get age factor
    age_factor = get_age_factor(user_input.age)
    
    # Select gender-specific predictions
    gender_suffix = '_M' if user_input.gender == 'male' else '_F'
    
    predictions = []
    for disease in DISEASES:
        target_key = f"{disease}{gender_suffix}"
        if target_key in targets:
            idx = targets.index(target_key)
            base_rate = max(0, float(all_predictions[idx]))
            adjusted_rate = base_rate * age_factor
            risk_level = get_risk_level(adjusted_rate)
            
            predictions.append(DiseaseRisk(
                disease=disease,
                disease_name=DISEASE_NAMES[disease],
                base_mortality_rate=round(base_rate, 2),
                adjusted_mortality_rate=round(adjusted_rate, 2),
                risk_level=risk_level,
                description=get_risk_description(disease, risk_level)
            ))
    
    # Environmental summary
    env_summary = {
        "pm25": round(env_data.get('pm25_mean', 0), 1),
        "pm10": round(env_data.get('pm10_mean', 0), 1),
        "temperature_avg": round(env_data.get('temp_C_mean', 0), 1),
        "humidity_avg": round(env_data.get('relative_humidity_mean', 0), 1),
        "data_year": int(env_data.get('year', 0))
    }
    
    return PredictionResponse(
        country=user_input.country,
        gender=user_input.gender,
        age=user_input.age,
        age_risk_factor=age_factor,
        model_used=model_type,
        model_year_range=f"{model_data['year_range'][0]}-{model_data['year_range'][1]}",
        predictions=predictions,
        environmental_summary=env_summary
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
