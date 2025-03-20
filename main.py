from fastapi import FastAPI, HTTPException
import tensorflow.lite as tflite
import numpy as np
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#Add all authorized and trustable domain here
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


tflite_model_path = "diabetes_model.tflite"
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class LiverDiseaseInput(BaseModel):
    age: int = Field(..., ge=20, le=80, description="Age in years (20-80)")
    gender: int = Field(..., ge=0, le=1, description="Gender: Male (0) or Female (1)")
    bmi: float = Field(..., ge=15, le=40, description="Body Mass Index (BMI) (15-40)")
    alcoholConsumption: float = Field(..., ge=0, le=20, description="Alcohol Consumption (units per week, 0-20)")
    smokingHistory: int = Field(..., ge=0, le=1, description="Smoking: No (0) or Yes (1)")
    geneticRisk: int = Field(..., ge=0, le=2, description="Genetic Risk: Low (0), Medium (1), High (2)")
    physicalActivity: float = Field(..., ge=0, le=10, description="Physical Activity (hours per week, 0-10)")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes: No (0) or Yes (1)")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension: No (0) or Yes (1)")
    liverFunctionTest: float = Field(..., ge=20, le=100, description="Liver Function Test (20-100)")


@app.get("/")
def home():
    return {"message": "Welcome to the Liver Disease Prediction API!"}

@app.post("/predict")
async def predict(data: DiabetesInput):
    try:
        input_data = np.array([[data.age, data.gender, data.bmi, data.alcoholConsumption, data.smokingHistory, data.geneticRisk, data.physicalActivity, data.diabetes, data.hypertension, 
         data.liverFunctionTest
    ]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index']).tolist()

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
