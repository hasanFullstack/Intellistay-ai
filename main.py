from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="IntelliStay AI Service",
    description="Machine Learning Microservice for Dynamic Pricing and Recommendations",
    version="1.0.0"
)

# Enable CORS so the Node.js backend can call it easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained ML model on startup
# DB-DRIVEN MODEL (trained from MongoDB data):
# MODEL_PATH = "pricing_model_db.pkl"

# DATASET-DRIVEN MODEL (trained from synthetic dataset - ACTIVE):
MODEL_PATH = "pricing_model_dataset.pkl"
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded ML model from {MODEL_PATH}")
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found. Ensure train.py has been run.")

# Define the expected input schema
class PricingInput(BaseModel):
    # Required old fields (Sent by the existing React Frontend)
    occupancy_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of occupied beds (0.0 to 1.0)")
    total_beds: int = Field(..., ge=1, description="Total beds in the room")
    base_price: float = Field(..., ge=0, description="The base monthly price set by owner")
    month: int = Field(..., ge=1, le=12, description="Month of the year (1-12)")
    city_tier: int = Field(..., ge=1, le=3, description="1=Major, 2=Medium, 3=Small city")
    
    # Optional New Dataset Features (So the frontend doesn't break if it doesn't send them)
    # The amenity_score is kept for backward compatibility with the frontend payload, but we don't use it in the new model directly
    amenity_score: int = Field(default=3, ge=0, le=10, description="Legacy amenity count")
    
    distance_to_university: float = Field(default=3.0, description="km to nearest university")
    student_rating: float = Field(default=4.0, description="Average student rating 1.0-5.0")
    has_ac: int = Field(default=0, description="1 if AC exists, 0 otherwise")
    has_wifi: int = Field(default=1, description="1 if WiFi exists, 0 otherwise")
    is_weekend: int = Field(default=0, description="1 if weekend, 0 otherwise")
    competitor_avg_price: float = Field(default=-1.0, description="Local market average")

    def __init__(self, **data):
        super().__init__(**data)
        # If the frontend didn't send a competitor price, just assume it's roughly the base price
        if getattr(self, "competitor_avg_price", -1.0) == -1.0:
            self.competitor_avg_price = self.base_price

@app.post("/predict-price", summary="Predict optimal dynamic price for a room")
async def predict_price(data: PricingInput):
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded into memory")
        
    try:
        # ---------------------------------------------------------
        # OLD LOGIC (DB MODEL):
        # ---------------------------------------------------------
        # features = np.array([[
        #     data.occupancy_rate,
        #     data.total_beds,
        #     data.base_price,
        #     data.month,
        #     data.amenity_score,
        #     data.city_tier
        # ]])
        
        # ---------------------------------------------------------
        # NEW LOGIC (DATASET MODEL):
        # Order MUST match train_from_dataset.py features:
        # ['base_price', 'total_beds', 'occupancy_rate', 'month', 'city_tier', 
        #  'distance_to_university', 'student_rating', 'has_ac', 'has_wifi', 
        #  'is_weekend', 'competitor_avg_price']
        # ---------------------------------------------------------
        features = np.array([[
            data.base_price,
            data.total_beds,
            data.occupancy_rate,
            data.month,
            data.city_tier,
            data.distance_to_university,
            data.student_rating,
            data.has_ac,
            data.has_wifi,
            data.is_weekend,
            data.competitor_avg_price
        ]])
        
        # Make Prediction
        predicted_price = model.predict(features)[0]
        
        # Round to nearest 100 Rs for cleaner pricing
        suggested_price = round(float(predicted_price) / 100) * 100
        
        # Calculate percentage difference
        diff_percent = ((suggested_price - data.base_price) / max(data.base_price, 1)) * 100
        
        # Generate human-readable reasoning based on the new premium dataset variables
        reason = "Base price based on standard market rates."
        if data.distance_to_university < 2.0 and data.has_ac == 1:
            reason = "Significant premium suggested due to AC availability and prime proximity (<2km) to university."
        elif diff_percent > 10:
            reason = "High demand surcharge (High occupancy / Peak season / Favorable market position)."
        elif diff_percent < -10:
            reason = "Discount suggested to attract bookings (Low occupancy / Market competition)."
        elif diff_percent > 0:
            reason = "Slight premium recommended based on local area averages."
        elif diff_percent < 0:
            reason = "Slight discount recommended to stay competitive."
            
        return {
            "suggested_price": suggested_price,
            "base_price": data.base_price,
            "price_change_percent": round(diff_percent, 1),
            "reasoning": reason
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health Check")
def health():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "service": "IntelliStay AI API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
