import os
import base64
import json
import requests
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from abc import ABC, abstractmethod
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
OPENAI_INF_MODEL = os.getenv("OPENAI_INF_MODEL")

GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL")

from json_models import (
    ChecklistItem, ExtractedChecklistItem, HyDEQuery,
    SearchResult, Citation, ComplianceStatus, EvaluationResult,
    GapAnalysisResult, ComprehensiveSearchResult
)

# Abstract base class for LLM services
class LLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response from a prompt"""
        pass
    
    @abstractmethod
    def generate_structured_output(self, prompt: str, system_prompt: Optional[str] = None, 
                                  output_schema: Any = None) -> Any:
        """Generate structured output based on a schema"""
        pass


class OpenAIService(LLMService):
    """Service for interacting with OpenAI's GPT models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = OPENAI_INF_MODEL):
        """Initialize OpenAI service with API key and model name"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=OPENAI_BASE_URL)
        self.last_call_time = 0
        self.calls_per_minute = 0
        self.max_calls_per_minute = 30  # Adjust based on your rate limits
    
    def _manage_rate_limits(self):
        """Manage API rate limits"""
        current_time = time.time()
        if current_time - self.last_call_time >= 60:
            # Reset counter if a minute has passed
            self.calls_per_minute = 0
            self.last_call_time = current_time
        elif self.calls_per_minute >= self.max_calls_per_minute:
            # Wait until the minute is up
            sleep_time = 60 - (current_time - self.last_call_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls_per_minute = 0
            self.last_call_time = time.time()
        
        self.calls_per_minute += 1
    
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response from GPT model"""
        self._manage_rate_limits()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI text generation: {e}")
            return f"Error: {str(e)}"
    
    def generate_structured_output(self, prompt: str, system_prompt: Optional[str] = None, 
                                  output_schema: Any = None) -> Any:
        """Generate structured output based on a pydantic schema"""
        self._manage_rate_limits()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            response_content = response.choices[0].message.content
            json_data = json.loads(response_content)
            
            # Convert to Pydantic model if schema provided
            if output_schema:
                return output_schema(**json_data)
            
            return json_data
        except Exception as e:
            print(f"Error in OpenAI structured output generation: {e}")
            if output_schema:
                # Create an empty instance if possible
                try:
                    return output_schema()
                except:
                    pass
            return {"error": str(e)}
    
    def create_embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Create embeddings for text using OpenAI models"""
        self._manage_rate_limits()
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model,  # Choose appropriate embedding model
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in OpenAI embedding creation: {e}")
            # Return empty embedding as fallback
            return [0.0] * 1536  # Common dimension for OpenAI embeddings


class GeminiService(LLMService):
    """Service for interacting with Google's Gemini models via OpenAI-compatible API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_FLASH_MODEL):
        """Initialize Gemini service with API key and model name"""
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        self.model_name = model_name
        # Initialize OpenAI client with Gemini base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=GEMINI_BASE_URL
        )
        self.last_call_time = 0
        self.calls_per_minute = 0
        self.max_calls_per_minute = 10  # Adjust based on Gemini's rate limits
    
    def _manage_rate_limits(self):
        """Manage API rate limits"""
        current_time = time.time()
        if current_time - self.last_call_time >= 60:
            # Reset counter if a minute has passed
            self.calls_per_minute = 0
            self.last_call_time = current_time
        elif self.calls_per_minute >= self.max_calls_per_minute:
            # Wait until the minute is up
            sleep_time = 60 - (current_time - self.last_call_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.calls_per_minute = 0
            self.last_call_time = time.time()
        
        self.calls_per_minute += 1
    
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response from Gemini model"""
        self._manage_rate_limits()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Gemini text generation: {e}")
            return f"Error: {str(e)}"
    
    def generate_structured_output(self, prompt: str, system_prompt: Optional[str] = None, 
                                   output_schema: Any = None) -> Any:
        """Generate structured output based on a pydantic schema"""
        self._manage_rate_limits()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            response_content = response.choices[0].message.content
            json_data = json.loads(response_content)
            
            # Convert to Pydantic model if schema provided
            if output_schema:
                return output_schema(**json_data)
            
            return json_data
        except Exception as e:
            print(f"Error in Gemini structured output generation: {e}")
            if output_schema:
                # Create an empty instance if possible
                try:
                    return output_schema()
                except:
                    pass
            return {"error": str(e)}
    
    def generate_multimodal_text(self, text_prompt: str, image_data: bytes) -> str:
        """Generate text from both text and image inputs"""
        
        # For Gemini multimodal, we need to make direct API calls (not via OpenAI library)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": text_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0.2
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                headers=headers,
                json=payload
            )
            
            response_json = response.json()
            
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                text_response = ""
                for part in response_json["candidates"][0]["content"]["parts"]:
                    if "text" in part:
                        text_response += part["text"]
                return text_response
            else:
                return "No response generated"
        except Exception as e:
            print(f"Error in Gemini multimodal text generation: {e}")
            return f"Error: {str(e)}"


class HyDESearchService:
    """Service for Hypothetical Document Embedding (HyDE) search"""
    
    def __init__(self, llm_service: LLMService, embedding_service: OpenAIService):
        """Initialize HyDE search service with an LLM service and embedding service"""
        self.llm_service = llm_service
        self.embedding_service = embedding_service
    
    def generate_hypothetical_document(self, checklist_item: ChecklistItem) -> str:
        """Generate hypothetical document content that would satisfy the checklist item"""
        
        prompt = f"""
        Create a hypothetical document fragment (1-2 paragraphs) that would perfectly satisfy this compliance requirement:
        
        REQUIREMENT: {checklist_item.content.checklist_title}
        DETAILED DESCRIPTION: {checklist_item.content.checklist_description}
        RATIONALE: {checklist_item.content.rationale}
        
        Write this as if it were an excerpt from a perfectly compliant mutual fund disclosure document.
        Use specific language, terminology, and formatting typical of mutual fund disclosure documents.
        """
        
        system_prompt = """
        You are an expert in mutual fund regulations and compliance documentation.
        Your task is to create a hypothetical perfect document fragment that would satisfy the given compliance requirement.
        Focus on creating realistic content that a compliant document would contain.
        """
        
        hypothetical_document = self.llm_service.generate_text(prompt, system_prompt)
        return hypothetical_document
    
    def create_search_query(self, hypothetical_document: str, original_requirement: str) -> str:
        """Extract a search query from the hypothetical document"""
        
        prompt = f"""
        Based on this hypothetical document fragment and original requirement:
        
        HYPOTHETICAL DOCUMENT:
        {hypothetical_document}
        
        ORIGINAL REQUIREMENT:
        {original_requirement}
        
        Create a search query (1-2 sentences) that would effectively find similar content in a real document.
        Focus on key terms, phrases, and concepts that are most distinctive and relevant.
        """
        
        system_prompt = """
        You are an expert in information retrieval and search query optimization.
        Your task is to extract the most effective search query from a hypothetical document.
        Focus on distinctive terminology and concepts that would yield high-precision search results.
        """
        
        search_query = self.llm_service.generate_text(prompt, system_prompt)
        return search_query
    
    def generate_hyde_query(self, checklist_item: ChecklistItem) -> HyDEQuery:
        """Generate a complete HyDE query for the given checklist item"""
        
        # Generate hypothetical document
        hypothetical_document = self.generate_hypothetical_document(checklist_item)
        
        # Create search query from hypothetical document
        original_requirement = f"{checklist_item.content.checklist_title}: {checklist_item.content.checklist_description}"
        search_query = self.create_search_query(hypothetical_document, original_requirement)
        
        # Create and return HyDE query
        return HyDEQuery(
            checklist_item_id=str(id(checklist_item)),  # Use a proper ID in production
            generated_document=hypothetical_document,
            search_query=search_query,
            iteration=1
        )
    
    def analyze_search_gap(self, checklist_item: ChecklistItem, 
                          current_findings: List[Dict[str, Any]]) -> GapAnalysisResult:
        """Analyze gaps between current findings and the search objective"""
        
        # Format the current findings for analysis
        findings_text = ""
        if current_findings:
            for i, finding in enumerate(current_findings):
                findings_text += f"\nFinding {i+1}:\n"
                findings_text += f"- Summary: {finding.get('findings_summary', 'No summary')}\n"
                findings_text += f"- Citations: {finding.get('citations', 'No citations')}\n"
                findings_text += f"- Pages: {finding.get('matched_pages', [])}\n"
        else:
            findings_text = "No findings yet."
        
        prompt = f"""
        COMPLIANCE OBJECTIVE:
        {checklist_item.content.checklist_title}
        
        DETAILED REQUIREMENT:
        {checklist_item.content.checklist_description}
        
        RATIONALE:
        {checklist_item.content.rationale}
        
        CURRENT FINDINGS:
        {findings_text}
        
        Please analyze:
        1. What specific information is missing from the current findings to determine compliance?
        2. Suggest a focused search query to find this missing information
        3. Determine if the current findings are already sufficient (true/false)
        
        Format your response as a JSON object with these fields:
        - gap_description: Detailed description of what information is missing
        - suggested_query: A specific search query to find the missing information
        - is_sufficient: Boolean indicating if current findings are sufficient (true) or need more search (false)
        """
        
        system_prompt = """
        You are an expert in regulatory compliance document analysis.
        Your task is to analyze the gap between current search findings and the compliance objective.
        Identify what specific information is missing that would help determine compliance.
        Suggest a focused search query to find the missing information.
        """
        
        # Generate gap analysis
        result = self.llm_service.generate_structured_output(prompt, system_prompt, GapAnalysisResult)
        return result


class ComplianceEvaluationService:
    """Service for evaluating compliance based on search results"""
    
    def __init__(self, llm_service: LLMService):
        """Initialize compliance evaluation service with an LLM service"""
        self.llm_service = llm_service
    
    def evaluate_compliance(self, checklist_item: ChecklistItem, 
                           search_results: List[SearchResult]) -> EvaluationResult:
        """Evaluate compliance for a checklist item based on search results"""
        
        # Format search results for evaluation
        context = ""
        for result in search_results:
            context += f"[Page {result.page_number}] {result.content}\n\n"
        
        page_numbers = [result.page_number for result in search_results]
        
        prompt = f"""
        Evaluate whether the following mutual fund document properly addresses this SEBI/AMFI disclosure requirement:
        
        REQUIREMENT: {checklist_item.content.checklist_title}
        DETAILED DESCRIPTION: {checklist_item.content.checklist_description}
        RATIONALE: {checklist_item.content.rationale}
        
        DOCUMENT EXCERPTS:
        {context}
        
        Evaluate the compliance status as one of:
        - "Compliant": The document fully satisfies the requirement
        - "Partially Compliant": The document addresses some aspects but lacks complete coverage
        - "Non-Compliant": The document clearly fails to address the requirement
        - "Unknown": There is insufficient information to determine compliance
        
        Format your response as a JSON object with these fields:
        - compliance_status: The status from the list above
        - confidence: A number between 0 and 1 indicating your confidence in this determination
        - findings_summary: Brief assessment of compliance (1-2 sentences)
        - citations: A list of direct quotes from the document supporting your conclusion
        """
        
        system_prompt = """
        You are an expert in mutual fund regulations and compliance.
        Your task is to evaluate whether a document properly addresses a specific disclosure requirement.
        
        Be flexible in your evaluation - the document may satisfy requirements using different 
        wording or structure than what's literally specified in the requirement.
        
        For Partially Compliant: Use when core requirements are addressed but some details are missing.
        For Non-Compliant: Use ONLY when the requirement is clearly absent or contradicted.
        When in doubt, prefer "Partially Compliant" over "Non-Compliant".
        """
        
        # Generate evaluation
        result_json = self.llm_service.generate_structured_output(prompt, system_prompt)
        
        # Extract needed fields
        compliance_status = ComplianceStatus(
            status=result_json.get("compliance_status", "Unknown"),
            confidence=result_json.get("confidence", 0.5)
        )
        
        # Process citations
        citations = []
        if "citations" in result_json:
            citation_texts = result_json["citations"]
            if isinstance(citation_texts, list):
                for text in citation_texts:
                    # Try to find page number in the citation text
                    # This is a simplified approach - in production you might need more robust parsing
                    for page in page_numbers:
                        if f"Page {page}" in text:
                            citations.append(Citation(text=text, page_number=page))
                            break
                    else:
                        # If no page number found, use the first page as default
                        if page_numbers:
                            citations.append(Citation(text=text, page_number=page_numbers[0]))
            else:
                # Handle case where citations is a string
                if page_numbers:
                    citations.append(Citation(text=citation_texts, page_number=page_numbers[0]))
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            checklist_item=checklist_item,
            compliance_status=compliance_status,
            findings_summary=result_json.get("findings_summary", "No findings summary available"),
            citations=citations,
            matched_pages=page_numbers,
            search_iterations=1
        )
        
        return evaluation_result


# Factory for creating LLM services
class LLMServiceFactory:
    """Factory for creating different LLM services"""
    
    @staticmethod
    def create_openai_service(api_key: Optional[str] = None, model_name: str = OPENAI_INF_MODEL) -> OpenAIService:
        """Create an OpenAI service"""
        return OpenAIService(api_key, model_name)
    
    @staticmethod
    def create_gemini_service(api_key: Optional[str] = None, model_name: str = GEMINI_FLASH_MODEL) -> GeminiService:
        """Create a Gemini service"""
        return GeminiService(api_key, model_name)
    
    @staticmethod
    def create_hyde_search_service(
        llm_service: LLMService, 
        embedding_service: OpenAIService
    ) -> HyDESearchService:
        """Create a HyDE search service"""
        return HyDESearchService(llm_service, embedding_service)
    
    @staticmethod
    def create_compliance_evaluation_service(llm_service: LLMService) -> ComplianceEvaluationService:
        """Create a compliance evaluation service"""
        return ComplianceEvaluationService(llm_service)
