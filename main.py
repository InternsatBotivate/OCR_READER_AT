import base64
import logging
import json
import os
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import openai

# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found.")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    print("WARNING: Google API Key or CSE ID not found. Validation search will be skipped.")
if not APPS_SCRIPT_URL:
    raise ValueError("No APPS_SCRIPT_URL found in .env file. Cannot submit data.")

openai.api_key = OPENAI_API_KEY

# --- Basic Setup ---
app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Request/Response Models ---
class OCRRequest(BaseModel):
    base64Image1: str # From Card Front
    base64Image2: str | None = None # From Card Back

class OCRResponse(BaseModel):
    company: str = Field(default="")
    name: str = Field(default="")
    title: str = Field(default="")
    phone: str = Field(default="")
    email: str = Field(default="")
    address: str = Field(default="")
    website: str = Field(default="")
    validation_source: str = Field(default="")
    is_validated: bool = Field(default=False)
    about_the_company: str = Field(default="")
    location: str = Field(default="")

# --- Google Search Function (Unchanged) ---
def search_google(query, num_results=3):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.info("Skipping Google search as API keys are not configured.")
        return None
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CSE_ID, 'q': query, 'num': num_results}
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        if "items" in results and len(results["items"]) > 0:
            logger.info(f"Google validation found {len(results['items'])} results.")
            return results["items"]
        else:
            logger.info("Google validation found no matching results.")
            return None
    except Exception as e:
        logger.error(f"Error during Google search: {e}", exc_info=True)
        return None

# --- Helper to parse JSON (Unchanged) ---
def parse_openai_json(response_content):
    if "```json" in response_content:
        response_content = response_content.split("```json")[1].split("```")[0].strip()
    return json.loads(response_content)

# --- Middleware (Unchanged) ---
@app.middleware("http")
async def log_requests(request, call_next):
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
        base64_image1 = request_data.base64Image1
        base64_image2 = request_data.base64Image2

        if "," in base64_image1:
            base64_image1 = base64_image1.split(',')[1]
        
        base64_image2_cleaned = ""
        if base64_image2:
            if "," in base64_image2:
                base64_image2_cleaned = base64_image2.split(',')[1]
            else:
                base64_image2_cleaned = base64_image2

        # === AI STEP 1: Extract data from image ===
        logger.info("Step 1: Sending image(s) to OpenAI for extraction...")
        extract_prompt = """Analyze the image(s) of the business card and extract all key information.
        There may be a front and a back image. Look at BOTH images to find all details.
        **Pay extremely close attention to stylized logos, gradients, shadows, or complex animations.** Use context (like slogans or other text) to determine the most likely correct spelling.
        Return the data as a clean JSON object with the following keys:
        'company', 'name', 'title', 'phone', 'email', 'address', 'slogan', 'location'
        If a piece of information is not found, return an empty string for that key."""
        
        image_list_content = [{"type": "text", "text": extract_prompt}]
        image_list_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"}})
        if base64_image2_cleaned:
            logger.info("Processing front and back images.")
            image_list_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image2_cleaned}"}})
        else:
             logger.info("Processing front image only.")

        extract_response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": image_list_content }],
            max_tokens=500
        )
        ocr_data_str = extract_response.choices[0].message.content
        ocr_data = parse_openai_json(ocr_data_str)
        logger.info(f"Step 1: Received from OpenAI: {ocr_data}")

        # === Step 2: Google Search for VALIDATION (Unchanged) ===
        search_query = ""
        company = ocr_data.get("company", "")
        name = ocr_data.get("name", "")
        slogan = ocr_data.get("slogan", "")
        if company and slogan:
            search_query = f"{company} {slogan}"
        elif company and name:
            search_query = f"{name} {company}"
        elif company:
            search_query = company
        elif slogan:
            search_query = slogan
        elif name:
            search_query = name
        google_results_list = None
        if search_query:
            logger.info(f"Step 2: Performing validation search for: {search_query}")
            google_results_list = search_google(search_query, num_results=3)

        # === AI STEP 3: Validate and FIRST Enrichment (Unchanged) ===
        logger.info("Step 3: Sending data to OpenAI for validation and first enrichment...")
        validation_prompt = f"""
        You are a data validation expert. I have OCR data and a list of Google results.
        Your job is to find the **single best match** from the results and use it to **correct** the OCR data.
        Here is the data from OCR: {json.dumps(ocr_data)}
        Here is the list of top 3 Google search results: {json.dumps(google_results_list) if google_results_list else "No results found."}
        **Instructions:**
        1.  **Find Best Match:** Find the one result (e.g., a company website or LinkedIn) that is the most likely match.
        2.  **If a Genuine Match is Found:**
            - Set 'is_validated' to true.
            - **Correct Data:** Correct any misspellings in the OCR 'company' or 'name' using the Google result.
            - **Enrich Website:** Fill in the 'website' field using the Google 'link'.
            - **Set 'validation_source' to the Google 'link'.**
            - **Return all other OCR fields as-is** (we will search for more in the next step).
        3.  **If NO Genuine Match is Found:**
            - Set 'is_validated' to false.
            - Return *only* the original OCR data.
        Return a single, final JSON object with all keys from the OCR data, plus 'website', 'validation_source', 'is_validated'.
        """
        validate_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": validation_prompt}],
            response_format={"type": "json_object"}
        )
        validated_data_str = validate_response.choices[0].message.content
        validated_data = json.loads(validated_data_str)
        logger.info(f"Step 3: Received validated data: {validated_data}")

        # === STEP 4: Targeted Search for ENRICHMENT ===
        enrichment_search_results = None
        if validated_data.get("is_validated"):
            validated_company = validated_data.get("company")
            # --- [SLIGHTLY UPDATED QUERY] ---
            contact_query = f'"{validated_company}" about description location contact info email phone address'
            # --- [END OF UPDATE] ---
            logger.info(f"Step 4: Performing targeted enrichment search for: {contact_query}")
            enrichment_search_results = search_google(contact_query, num_results=3)
        else:
            logger.info("Step 4: Skipping enrichment search (validation failed).")

        # === STEP 5: Final Merge and Enrichment ===
        if enrichment_search_results:
            logger.info("Step 5: Sending data to OpenAI for final merge...")
            # --- [UPDATED PROMPT FOR RELIABILITY] ---
            final_merge_prompt = f"""
            You are a data enrichment assistant. I have partially validated data from a business card.
            I have performed a *second* Google search to find missing company details like 'about', 'location', and contact info.
            Your job is to merge this new information reliably.

            Here is the data we have so far (after Step 3 validation):
            {json.dumps(validated_data)}

            Here is the list of results from the *new* enrichment search:
            {json.dumps(enrichment_search_results)}

            **Instructions:**
            1.  **Review Snippets Carefully:** Look through the 'snippet' and 'title' of **all** search results.
            2.  **Extract Key Details:**
                * `about_the_company`: Find a concise (1-2 sentence) description of what the company does.
                * `location`: Find the city, state, or general region mentioned. If an address exists, extract the city/state from it.
                * `phone`, `email`, `address`: Find the first credible contact details.
            3.  **Merge Data:** Return a final JSON object. Use the data from "data we have so far" as the base.
            4.  **Fill Empty Fields:** Use the details extracted in step 2 to fill ONLY the fields that are currently empty or contain just placeholder text.
            5.  **Prioritize:** If multiple sources provide information for the same empty field, prioritize the most official-looking source (e.g., the company's own website snippet over a directory listing).
            6.  **Do not overwrite** data that already exists unless the existing data is clearly wrong or a placeholder. Keep `validation_source` as it is.

            Return a single, final JSON object including 'about_the_company' and 'location'.
            """
            # --- [END OF UPDATE] ---
            
            final_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_merge_prompt}],
                response_format={"type": "json_object"}
            )
            final_data_str = final_response.choices[0].message.content
            final_data = json.loads(final_data_str)
            logger.info(f"Step 5: Received final merged data: {final_data}")
        else:
            logger.info("Step 5: No final merge needed.")
            final_data = validated_data
        
        phone_number = final_data.get("phone", "")
        if phone_number and phone_number.startswith('+'):
            final_data["phone"] = "'" + phone_number
        if 'slogan' in final_data:
            del final_data['slogan']

        # === STEP 6: Submit to Google Apps Script (Unchanged) ===
        logger.info(f"Step 6: Submitting final data to Google Apps Script...")
        
        apps_script_payload = {
            "action": "save",
            "photo1Base64": base64_image1,
            "photo2Base64": base64_image2_cleaned or "", 
            "extractedData": final_data
        }

        try:
            headers = {'Content-Type': 'text/plain;charset=utf-8'}
            response = requests.post(
                APPS_SCRIPT_URL,
                data=json.dumps(apps_script_payload),
                headers=headers
            )
            response.raise_for_status()
            save_result = response.json()
            if not save_result.get("success"):
                logger.error(f"Google Apps Script failed to save: {save_result.get('message')}")
                raise HTTPException(status_code=500, detail=f"Google Script Error: {save_result.get('message')}")
            logger.info("Step 6: Successfully submitted to Google Apps Script.")
        
        except Exception as e:
            logger.error(f"Failed to submit to Google Apps Script: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Apps Script Submission Failed: {e}")

        return OCRResponse(**final_data)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.head("/")
def status_check():
    return Response(status_code=200)

if __name__ == "__main__":
    print("Starting FastAPI server. Run with: uvicorn main:app --host 0.0.0.0 --port 8000")
