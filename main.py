import base64
import logging
import json
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import openai

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Make sure to set it in your .env file.")

openai.api_key = OPENAI_API_KEY

# --- Basic Setup ---
app = FastAPI()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("backend.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Request/Response Models ---
class OCRRequest(BaseModel):
    base64Image: str

class OCRResponse(BaseModel):
    company: str = Field(default="")
    name: str = Field(default="")
    title: str = Field(default="")
    phone: str = Field(default="")
    email: str = Field(default="")
    address: str = Field(default="")

# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Request to {request.url} completed with status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "OCR Backend is running"}

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(request_data: OCRRequest):
    try:
        base64_image = request_data.base64Image
        if "," in base64_image:
            base64_image = base64_image.split(',')[1]

        logger.info("Sending image to OpenAI for analysis...")

        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze the image of the business card and extract the key information.
                            Return the data as a clean JSON object with the following keys: 'company', 'name', 'title', 'phone', 'email', 'address'.
                            If a piece of information is not found, return an empty string for that key."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        json_response_str = response.choices[0].message.content
        logger.info(f"Received from OpenAI: {json_response_str}")

        if "```json" in json_response_str:
            json_response_str = json_response_str.split("```json")[1].split("```")[0].strip()

        # Parse the JSON response
        parsed_data = json.loads(json_response_str)

        # --- FIX: Prepend apostrophe to phone numbers starting with '+' ---
        phone_number = parsed_data.get("phone", "")
        if phone_number and phone_number.startswith('+'):
            parsed_data["phone"] = "'" + phone_number
        # --- END OF FIX ---

        return OCRResponse(**parsed_data)

    except Exception as e:
        logger.error(f"An error occurred during OpenAI processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server. Run with: uvicorn main:app --host 0.0.0.0 --port 8000")
