import os
import time
import json
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
# from groq import Groq  # Commented out as per request

# Import Pydantic models
from pydantic import BaseModel, Field, ValidationError
from models import ChecklistItem, ChecklistEvaluation

# Load environment variables
load_dotenv()

# Initialize Groq client - Commented out
# groq_api_key = os.getenv("GROQ_API_KEY")
# groq_client = Groq(api_key=groq_api_key)

# Initialize Gemini client through OpenAI compatible client - Commented out
from openai import OpenAI

# gemini_api_key = os.getenv("GEMINI_API_KEY")
# gemini_client = OpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Model configurations - Commented out unused models
# GROQ_MODEL = os.getenv("LLM_GROQ_LARGE", "llama-3.3-70b-versatile")
# GEMINI_MODEL = os.getenv("LLM_GEMINI_LARGE", "gemini-2.0-flash")

# Rate limits (per minute) - Commented out unused models
# GROQ_RPM = 30
# GEMINI_RPM = 15

# Token limits - Commented out unused models
# GROQ_TPM = 12000
# GROQ_TPD = 100000
# GEMINI_TPM = 1000000

# OpenAI limits
OPENAI_RPM = 100 # no limit
OPENAI_TPM = 100000000 # no limit

# Tracking usage - Commented out unused models
# last_groq_call = 0
# last_gemini_call = 0
# groq_calls_this_minute = 0
# gemini_calls_this_minute = 0
last_openai_call = 0
openai_calls_this_minute = 0


# Define Pydantic models for structured extraction
class ChecklistItemExtract(BaseModel):
    """Schema for extracting a single checklist item"""
    checklist_title: str = Field(..., description="Short title of the checklist item")
    checklist_description: str = Field(..., description="Detailed description of what needs to be disclosed")
    rationale: str = Field(..., description="Rationale from SEBI circular or AMFI email")
    page_numbers: str = Field(..., description="Page numbers in the regulatory document")


class ChecklistItemsExtract(BaseModel):
    """Schema for extracting a list of checklist items"""
    items: List[ChecklistItemExtract] = Field(default_factory=list, description="List of checklist items")


class DocumentComplianceExtract(BaseModel):
    """Schema for extracting compliance assessment for a single checklist item"""
    findings_summary: str = Field(..., description="Brief assessment of compliance (1-2 sentences)")
    citations: str = Field(..., description="Direct quotes from the document supporting the conclusion")


class RAGEvaluationResponse(BaseModel):
    """Schema for RAG-based evaluation response"""
    findings_summary: str = Field(..., description="Brief assessment of compliance (1-2 sentences)")
    citations: str = Field(..., description="Direct quotes from the document supporting the conclusion")
    compliance_status: str = Field(..., description="Either 'Compliant', 'Partially Compliant', 'Non-Compliant', or 'Unknown'")
    matched_pages: List[int] = Field(default_factory=list, description="List of page numbers where relevant information was found")


def reset_rate_limits_if_needed():
    """Reset call counters if a minute has passed."""
    global last_openai_call, openai_calls_this_minute
    
    current_time = time.time()
    
    # Reset OpenAI counters if a minute has passed
    if current_time - last_openai_call >= 60:
        openai_calls_this_minute = 0


def preprocess_json_response(json_data: Dict, schema_model=None) -> Union[Dict, Any]:
    """Preprocess JSON data to ensure compatibility with Pydantic models"""
    if not schema_model:
        return json_data
    
    # Handle special case for DocumentComplianceExtract
    if schema_model == DocumentComplianceExtract or schema_model == RAGEvaluationResponse:
        # Convert citations from list to string if needed
        if 'citations' in json_data and isinstance(json_data['citations'], list):
            json_data['citations'] = '\n\n'.join(json_data['citations'])
            
        # Make sure the required fields exist
        if 'findings_summary' not in json_data:
            json_data['findings_summary'] = "No findings summary available"
        if 'citations' not in json_data:
            json_data['citations'] = "No citations found"
            
        # For RAGEvaluationResponse, ensure matched_pages is a list of integers
        if schema_model == RAGEvaluationResponse:
            if 'matched_pages' not in json_data:
                json_data['matched_pages'] = []
            elif isinstance(json_data['matched_pages'], str):
                # Convert string to list of integers
                try:
                    # Handle comma-separated lists
                    if ',' in json_data['matched_pages']:
                        json_data['matched_pages'] = [int(p.strip()) for p in json_data['matched_pages'].split(',')]
                    # Handle ranges
                    elif '-' in json_data['matched_pages']:
                        start, end = map(int, json_data['matched_pages'].split('-'))
                        json_data['matched_pages'] = list(range(start, end + 1))
                    else:
                        # Single number
                        json_data['matched_pages'] = [int(json_data['matched_pages'])]
                except:
                    json_data['matched_pages'] = []
            
            # Ensure compliance_status is valid
            if 'compliance_status' not in json_data:
                json_data['compliance_status'] = "Unknown"
            elif json_data['compliance_status'] not in ["Compliant", "Partially Compliant", "Non-Compliant", "Unknown"]:
                json_data['compliance_status'] = "Unknown"
            
    # Convert any other list fields to strings when the schema expects strings
    try:
        result = schema_model(**json_data)
        return result
    except ValidationError as e:
        print(f"Validation error before preprocessing: {e}")
        # Try to fix common issues
        for field_name, field in schema_model.__annotations__.items():
            if field_name in json_data and isinstance(json_data[field_name], list) and field == str:
                json_data[field_name] = '\n\n'.join(str(item) for item in json_data[field_name])
        
        # Try again with fixed data
        try:
            return schema_model(**json_data)
        except ValidationError as e2:
            print(f"Validation error after preprocessing: {e2}")
            # Create an empty model if we can
            if hasattr(schema_model, '__pydantic_generic_metadata__') or hasattr(schema_model, '__pydantic_fields_set__'):
                return schema_model()
            return json_data
    
    return json_data


# Commented out Groq function
# def call_groq_with_structure(prompt: str, system_prompt: Optional[str] = None, schema_model=None) -> Any:
#     """Call Groq API with rate limiting and structured output using Pydantic."""
#     # Implementation commented out


# Commented out Gemini function
# def call_gemini_with_structure(prompt: str, system_prompt: Optional[str] = None, schema_model=None) -> Any:
#     """Call Gemini API with rate limiting and structured output using Pydantic."""
#     # Implementation commented out


def call_openai_with_structure(prompt: str, system_prompt: Optional[str] = None, schema_model=None) -> Any:
    """Call OpenAI API with rate limiting and structured output using Pydantic."""
    global last_openai_call, openai_calls_this_minute
    
    reset_rate_limits_if_needed()
    
    # Check if we've hit the rate limit
    if openai_calls_this_minute >= OPENAI_RPM:
        wait_time = 60 - (time.time() - last_openai_call)
        if wait_time > 0:
            print(f"Waiting {wait_time:.1f}s for OpenAI rate limit...")
            time.sleep(wait_time)
            openai_calls_this_minute = 0
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # For OpenAI client, try to use the parse method for structured output
        if schema_model:
            try:
                # Use parse method for structured output if available
                completion = openai_client.beta.chat.completions.parse(
                    model=OPENAI_MODEL,
                    messages=messages,
                    response_format=schema_model,
                )
                
                # Update tracking
                last_openai_call = time.time()
                openai_calls_this_minute += 1
                
                return completion.choices[0].message.parsed
            except (AttributeError, NotImplementedError):
                # Fallback to regular completion with JSON format if parse is not available
                completion = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                # Update tracking
                last_openai_call = time.time()
                openai_calls_this_minute += 1
                
                # Parse JSON response
                response_content = completion.choices[0].message.content
                json_data = json.loads(response_content)
                
                # Preprocess before returning
                result = preprocess_json_response(json_data, schema_model)
                return result
        else:
            # Regular chat completion without structure
            completion = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Update tracking
            last_openai_call = time.time()
            openai_calls_this_minute += 1
            
            # Parse JSON response
            response_content = completion.choices[0].message.content
            json_data = json.loads(response_content)
            
            return json_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from OpenAI: {e}")
        if schema_model:
            # Return empty instance of schema model
            return schema_model()
        return {}
    except ValidationError as e:
        print(f"Error validating OpenAI response against schema: {e}")
        if schema_model:
            return schema_model()
        return {}
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        if schema_model:
            return schema_model()
        return {}


def extract_requirements_from_text(text: str, page_range: str) -> Dict:
    """
    Extract disclosure requirements from text using OpenAI.
    """
    reset_rate_limits_if_needed()
    
    system_prompt = """
    You are an expert in mutual fund regulations and compliance. Your task is to extract disclosure requirements from SEBI and AMFI regulatory circulars.
    
    Extract specific disclosure requirements that mutual funds must follow in their documents.
    
    Structure your response as a valid JSON object with these fields:
    - items: A list of extracted requirements, each with:
      - checklist_title: A short title for the requirement
      - checklist_description: Detailed description of what needs to be disclosed
      - rationale: The rationale from the regulatory document
      - page_numbers: The page numbers in the regulatory document where this requirement is found
    
    Format your response ONLY as a valid JSON object.
    """
    
    prompt = f"""
    Extract the specific disclosure requirements that mutual funds must follow from the following regulatory text.
    Focus on concrete requirements for what must be disclosed in fund documents.
    
    TEXT FROM PAGES {page_range}:
    {text}
    """
    
    try:
        # Use OpenAI for extraction
        result = call_openai_with_structure(prompt, system_prompt, ChecklistItemsExtract)
        
        if result and hasattr(result, 'items'):
            return result.model_dump()
        
        # If failed, return empty checklist
        return {"items": []}
    except Exception as e:
        print(f"Error extracting requirements: {e}")
        return {"items": []}


def consolidate_checklist(raw_extractions: List[Dict]) -> Dict:
    """
    Consolidate multiple checklist extractions into a single unified checklist using OpenAI.
    """
    system_prompt = """
    You are an expert in mutual fund regulations and compliance. Your task is to consolidate multiple extracted disclosure requirements into a unified checklist.
    
    Remove duplicates, combine similar requirements, and ensure the final list is comprehensive.
    
    Structure your response as a valid JSON object with these fields:
    - items: A list of consolidated requirements, each with:
      - checklist_title: A short title for the requirement
      - checklist_description: Detailed description of what needs to be disclosed
      - rationale: The rationale from the regulatory document
      - page_numbers: The page numbers in the regulatory document where this requirement is found
    
    Format your response ONLY as a valid JSON object.
    """
    
    # Prepare list of extractions for the prompt
    extractions_text = json.dumps(raw_extractions, indent=2)
    
    prompt = f"""
    Consolidate the following extracted disclosure requirements into a unified checklist.
    Remove duplicates, combine similar requirements, and ensure the final list is comprehensive.
    
    EXTRACTED REQUIREMENTS:
    {extractions_text}
    """
    
    try:
        # Use OpenAI for consolidation
        result = call_openai_with_structure(prompt, system_prompt, ChecklistItemsExtract)
        
        if result and hasattr(result, 'items'):
            return result.model_dump()
        
        # If failed, return empty items list
        return {"items": []}
    except Exception as e:
        print(f"Error consolidating checklist: {e}")
        return {"items": []}


def evaluate_document_against_checklist(document_text: str, checklist_item: Dict) -> Dict:
    """
    Evaluate a document against a single checklist item using OpenAI.
    """
    reset_rate_limits_if_needed()
    
    system_prompt = """
    You are an expert in mutual fund regulations and compliance. 
    Your task is to evaluate whether a document properly addresses a specific disclosure requirement.
    
    Look for evidence in the document that specifically addresses the requirement.
    
    Return your assessment as a valid JSON object with these fields:
    - findings_summary: Brief assessment of compliance (1-2 sentences)
    - citations: Direct quotes from the document supporting your conclusion (if found)
    
    Format your response ONLY as a valid JSON object.
    """
    
    prompt = f"""
    Evaluate whether the following mutual fund document properly addresses this SEBI/AMFI disclosure requirement:
    
    REQUIREMENT TITLE: {checklist_item['checklist_title']}
    REQUIREMENT DESCRIPTION: {checklist_item['checklist_description']}
    
    Look for evidence in the document that addresses this specific requirement.
    
    DOCUMENT TEXT:
    {document_text[:8000]}  # Limit to respect context window
    """
    
    try:
        # Use OpenAI for evaluation
        result = call_openai_with_structure(prompt, system_prompt, DocumentComplianceExtract)
        
        if result and hasattr(result, 'findings_summary') and hasattr(result, 'citations'):
            return result.model_dump()
        
        # If failed, return default values
        return {
            "findings_summary": "Could not evaluate compliance",
            "citations": "No citations found"
        }
    except Exception as e:
        print(f"Error in document evaluation: {e}")
        return {
            "findings_summary": f"Evaluation error: {str(e)}",
            "citations": "Error occurred during evaluation"
        } 