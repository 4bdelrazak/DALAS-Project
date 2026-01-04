# Respiratory Disease Risk Prediction API

FastAPI server that predicts respiratory disease mortality risk based on location, gender, and age.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# OR
uvicorn main:app --reload
```

Server runs at: http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/models` | GET | List available models |
| `/countries` | GET | List available countries |
| `/diseases` | GET | List diseases we predict |
| `/predict` | POST | Get risk predictions |

## Usage Example

```bash
curl -X POST "http://localhost:8000/predict?model_type=shortterm" \
  -H "Content-Type: application/json" \
  -d '{"country": "Germany", "gender": "male", "age": 45}'
```

## Two Models Available

### 1. Short-term Model (default, recommended)
- **Years**: 2003-2023
- **Features**: Meteorological + Air Pollution data
- **R² Score**: 0.890

### 2. Long-term Model
- **Years**: 1980-2023  
- **Features**: Meteorological data only
- **R² Score**: 0.886

Use `?model_type=longterm` to switch models.

## Response Example

```json
{
  "country": "Germany",
  "gender": "male",
  "age": 45,
  "age_risk_factor": 1.2,
  "model_used": "shortterm",
  "predictions": [
    {
      "disease": "copd",
      "disease_name": "COPD",
      "base_mortality_rate": 25.4,
      "adjusted_mortality_rate": 30.5,
      "risk_level": "Moderate",
      "description": "..."
    }
  ]
}
```

## Interactive Docs

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

