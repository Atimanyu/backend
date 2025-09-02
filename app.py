# from models import engine, Patients
from fastapi import Body
from fastapi import FastAPI
# from sqlalchemy.orm import sessionmaker
from fastapi.responses import Response
from pydantic import BaseModel
from datetime import date
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io
import boto3
import requests
SAGEMAKER_URL = "https://runtime.sagemaker.ap-south-1.amazonaws.com/endpoints/pipeline-endpoint1/invocations"  # Replace with your actual SageMaker endpoint URL
sagemaker_client = boto3.client("sagemaker-runtime", region_name="ap-south-1",aws_access_key_id="",aws_secret_access_key="")
class Patients(BaseModel):
    name: str
    age: int
    gender: str
    dob: str
    image_b64: str
app = FastAPI()

@app.post("/predict")
def predict(patients: Patients):
    try:
        # Prepare payload as JSON (image in base64)
        payload = {
            "patient_name": patients.name,
            "age": patients.age,
            "dob": patients.dob,
            "gender": patients.gender,
            "image": patients.image_b64
        }

        # Invoke SageMaker endpoint with JSON input
        response = sagemaker_client.invoke_endpoint(
            EndpointName="pipeline-endpoint1",      # replace with your endpoint
            ContentType="application/json",
            Body=bytes(str(payload), "utf-8")       # send JSON as bytes
        )

        # Read the PDF bytes returned by the model
        pdf_bytes = response["Body"].read()
        pdf_stream = io.BytesIO(pdf_bytes)

        # Return as streaming response
        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename=report_{patients.name}.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
