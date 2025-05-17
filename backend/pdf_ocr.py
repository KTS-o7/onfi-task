import os
import io
import time
import uuid
import base64
import hashlib
import chromadb
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import pypdf
from openai import OpenAI
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import concurrent.futures
from functools import partial

MAX_WORKERS = 5

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
OPENAI_INF_MODEL = os.getenv("OPENAI_INF_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chromadb_data")
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds

class PDFProcessor:
    """Class to process PDFs and extract text and images"""
    
    def __init__(self, dpi: int = 300):
        """Initialize PDF processor with given DPI for image extraction"""
        self.dpi = dpi
    
    def extract_from_path(self, pdf_path: str) -> Tuple[List[str], List[Image.Image]]:
        """Extract text and images from a PDF file path"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract images
        images = self._extract_images_from_path(pdf_path)
        
        # Extract text
        text_pages = self._extract_text_from_path(pdf_path)
        
        return text_pages, images
    
    def extract_from_bytes(self, pdf_bytes: bytes) -> Tuple[List[str], List[Image.Image]]:
        """Extract text and images from PDF bytes"""
        logger.info("Processing PDF from bytes")
        
        # Extract images
        images = self._extract_images_from_bytes(pdf_bytes)
        
        # Extract text
        text_pages = self._extract_text_from_bytes(pdf_bytes)
        
        return text_pages, images
    
    def _extract_images_from_path(self, pdf_path: str) -> List[Image.Image]:
        """Extract images from PDF file path"""
        try:
            images = convert_from_path(pdf_path, dpi=self.dpi)
            logger.info(f"Extracted {len(images)} page images from PDF")
            return images
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            raise
    
    def _extract_images_from_bytes(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Extract images from PDF bytes"""
        try:
            images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            logger.info(f"Extracted {len(images)} page images from PDF bytes")
            return images
        except Exception as e:
            logger.error(f"Error extracting images from PDF bytes: {e}")
            raise
    
    def _extract_text_from_path(self, pdf_path: str) -> List[str]:
        """Extract text from PDF file path"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text_pages = []
                
                for i in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    text_pages.append(text)
                
                logger.info(f"Extracted text from {len(text_pages)} PDF pages")
                return text_pages
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_text_from_bytes(self, pdf_bytes: bytes) -> List[str]:
        """Extract text from PDF bytes"""
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text_pages = []
            
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                text_pages.append(text)
            
            logger.info(f"Extracted text from {len(text_pages)} PDF pages")
            return text_pages
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {e}")
            raise


class VisionLLMProcessor:
    """Process images with Vision LLMs to extract detailed information"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_VISION_MODEL):
        """Initialize Vision LLM processor with API key and model"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=OPENAI_BASE_URL)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        self.requests_this_minute = 0
        self.max_requests_per_minute = 20  # Adjust based on your rate limits
        self.minute_start_time = time.time()
    
    def _manage_rate_limits(self):
        """Manage API rate limits with exponential backoff"""
        current_time = time.time()
        
        # Enforce minimum interval between requests
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        # Check if we need to reset the minute counter
        if current_time - self.minute_start_time >= 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time
        
        # Check if we've hit the rate limit for this minute
        if self.requests_this_minute >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            self.requests_this_minute = 0
            self.minute_start_time = time.time()
        
        self.last_request_time = time.time()
        self.requests_this_minute += 1
    
    def process_image(self, image: Image.Image, page_number: int, page_text: str) -> Dict[str, Any]:
        """Process an image with Vision LLM to extract detailed information"""
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                self._manage_rate_limits()
                
                # Convert image to bytes for API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Create base64 encoded image
                base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
                
                # Prepare prompt for vision LLM
                system_prompt = """
                You are an expert financial document analyzer.
                Extract key information from this page of a mutual fund Scheme Information Document (SID).
                Pay special attention to:
                1. Risk indicators (like riskometer)
                2. Fee structures and expense ratios
                3. Fund objectives and strategies
                4. Performance metrics
                5. Tables, charts, and visual elements
                6. Important disclosures
                
                For visual elements, describe what you see in detail.
                """
                
                prompt = f"""
                This is page {page_number} of a mutual fund SID document.
                
                The extracted text content is:
                ---
                {page_text}
                ---
                
                Analyze both the image and text together. 
                
                In particular, identify and describe in detail any visual elements like charts, tables, 
                riskometers, logos, etc. that are critical for compliance requirements.
                
                Also identify any disclosures or statements related to SEBI requirements.
                
                Format your response as JSON with these fields:
                - visual_elements: List of visual elements found and their descriptions
                - key_disclosures: List of key disclosures found on this page
                - page_summary: A concise summary of what this page covers
                - compliance_elements: Any specific elements related to regulatory compliance
                """
                
                # Call Vision API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                # Parse and return the response
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                
                # Add additional metadata
                result["page_number"] = page_number
                result["raw_text"] = page_text
                
                logger.info(f"Successfully processed page {page_number} with Vision LLM")
                
                return result
                
            except Exception as e:
                logger.warning(f"Error processing image (attempt {attempt+1}/{MAX_RETRY_ATTEMPTS}): {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Exponential backoff
                    sleep_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to process image after {MAX_RETRY_ATTEMPTS} attempts")
                    # Return basic info for this page
                    return {
                        "page_number": page_number,
                        "raw_text": page_text,
                        "visual_elements": [],
                        "key_disclosures": [],
                        "page_summary": "Error processing page with Vision LLM",
                        "compliance_elements": []
                    }


class EmbeddingProcessor:
    """Create and manage embeddings for PDF content"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_EMBEDDING_MODEL):
        """Initialize embedding processor with API key and model"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=OPENAI_BASE_URL)
        
        # Cache to avoid redundant embedding calculations
        self._embedding_cache = {}
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
        self.requests_this_minute = 0
        self.max_requests_per_minute = 60  # Adjust based on OpenAI's rate limits
        self.minute_start_time = time.time()
    
    def _manage_rate_limits(self):
        """Manage API rate limits with exponential backoff"""
        current_time = time.time()
        
        # Enforce minimum interval between requests
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        # Check if we need to reset the minute counter
        if current_time - self.minute_start_time >= 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time
        
        # Check if we've hit the rate limit for this minute
        if self.requests_this_minute >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            self.requests_this_minute = 0
            self.minute_start_time = time.time()
        
        self.last_request_time = time.time()
        self.requests_this_minute += 1
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text with caching"""
        # Create a hash of the text for caching
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Return from cache if available
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Get new embedding from API
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                self._manage_rate_limits()
                
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                self._embedding_cache[text_hash] = embedding
                return embedding
                
            except Exception as e:
                logger.warning(f"Error getting embedding (attempt {attempt+1}/{MAX_RETRY_ATTEMPTS}): {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Exponential backoff
                    sleep_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to get embedding after {MAX_RETRY_ATTEMPTS} attempts")
                    # Return zero vector as fallback
                    return [0.0] * EMBEDDING_DIMENSION
    
    def create_colpali_embeddings(self, text: str, vision_data: Dict[str, Any]) -> List[float]:
        """
        Create ColPaLI-style embeddings that combine text and visual information
        Uses an approach similar to the ColPaLI paper, adapted for our use case
        """
        # Combine text with visual elements from vision LLM
        visual_elements = vision_data.get("visual_elements", [])
        visual_elements_text = " ".join([str(elem) for elem in visual_elements]) if visual_elements else ""
        
        # Extract key disclosures
        key_disclosures = vision_data.get("key_disclosures", [])
        key_disclosures_text = " ".join([str(disc) for disc in key_disclosures]) if key_disclosures else ""
        
        # Extract compliance elements
        compliance_elements = vision_data.get("compliance_elements", [])
        compliance_elements_text = " ".join([str(elem) for elem in compliance_elements]) if compliance_elements else ""
        
        # Create a combined representation with explicit markers for different information types
        combined_text = (
            f"PAGE TEXT: {text} "
            f"VISUAL ELEMENTS: {visual_elements_text} "
            f"KEY DISCLOSURES: {key_disclosures_text} "
            f"COMPLIANCE ELEMENTS: {compliance_elements_text} "
            f"PAGE SUMMARY: {vision_data.get('page_summary', '')}"
        )
        
        # Get embedding for the combined representation
        return self.get_embedding(combined_text)


class ChromaDBManager:
    """Manage ChromaDB for storing and retrieving embeddings"""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        """Initialize ChromaDB manager with a persistence directory"""
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"ChromaDB initialized with persistence directory: {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def get_or_create_collection(self, collection_name: str) -> Any:
        """Get or create a collection in ChromaDB"""
        try:
            # Use ChromaDB's built-in OpenAI embedding function
            import chromadb.utils.embedding_functions as embedding_functions
            
            # Create embedding function using the proper API parameters
            embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-3-small",
                api_base=os.environ.get("OPENAI_BASE_URL"), 
            )
            
            # First, try to get the collection
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_func
                )
                logger.info(f"Retrieved existing collection: {collection_name}")
                return collection
            except Exception as e:
                logger.info(f"Collection {collection_name} not found or incompatible, will create new one: {str(e)}")
                
                # Try to delete the collection if it exists but with wrong dimensions
                try:
                    self.client.delete_collection(name=collection_name)
                    logger.warning(f"Deleted existing collection {collection_name} with incompatible dimensions")
                except Exception:
                    # Collection doesn't exist, which is fine
                    pass
                    
                # Create a new collection with our embedding function
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func,
                    metadata={"created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                )
                logger.info(f"Created new collection: {collection_name}")
                return collection
                
        except Exception as e:
            logger.error(f"Error getting or creating collection: {e}")
            raise
    
    def add_document(self, 
                    collection: Any, 
                    doc_id: str, 
                    page_num: int, 
                    embedding: List[float], 
                    metadata: Dict[str, Any], 
                    text_content: str) -> None:
        """Add a document page to the ChromaDB collection"""
        try:
            # Create a unique ID for this page
            page_id = f"{doc_id}_page_{page_num}"
            
            # Sanitize metadata - convert complex objects to strings
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized_metadata[key] = value
                else:
                    # Convert complex objects to JSON strings
                    sanitized_metadata[key + "_json"] = json.dumps(value)
            
            collection.add(
                ids=[page_id],
                embeddings=[embedding],
                metadatas=[sanitized_metadata],  # Use sanitized metadata
                documents=[text_content]
            )
            logger.info(f"Added document page {page_num} to ChromaDB with ID: {page_id}")
        except Exception as e:
            logger.error(f"Error adding document to ChromaDB: {e}")
            raise
    
    def get_document_fingerprint(self, document_text: str) -> str:
        """Generate a fingerprint for a document to detect changes"""
        return hashlib.md5(document_text.encode('utf-8')).hexdigest()
    
    def document_exists(self, collection: Any, doc_id: str) -> bool:
        """Check if a document already exists in the collection"""
        try:
            # Try to query for the first page of the document
            result = collection.get(
                ids=[f"{doc_id}_page_0"],
                include=["metadatas"]
            )
            return len(result["ids"]) > 0
        except:
            return False


class SIDProcessor:
    """Main class for processing SID documents with ColPaLI-style embeddings"""
    
    def __init__(self, 
                openai_api_key: Optional[str] = None, 
                vision_model: str = "gpt-4o",
                embedding_model: str = "text-embedding-3-small"):
        """Initialize SID processor with API keys and models"""
        self.pdf_processor = PDFProcessor()
        self.vision_processor = VisionLLMProcessor(api_key=openai_api_key, model=vision_model)
        self.embedding_processor = EmbeddingProcessor(api_key=openai_api_key, model=embedding_model)
        self.db_manager = ChromaDBManager()
    
    def process_page(self, page_num: int, image: Image.Image, text: str, doc_id: str, doc_title: str, num_pages: int, collection: Any) -> Dict[str, Any]:
        """Process a single page of the document"""
        try:
            # Process image with Vision LLM
            vision_data = self.vision_processor.process_image(
                image, 
                page_num, 
                text
            )
            
            # Create ColPaLI-style embedding
            embedding = self.embedding_processor.create_colpali_embeddings(
                text,
                vision_data
            )
            
            # Page metadata
            page_metadata = {
                "doc_id": doc_id,
                "title": doc_title,
                "page_number": page_num,
                "total_pages": num_pages,
                "has_visual_elements": len(vision_data.get("visual_elements", [])) > 0,
                "vision_data": json.dumps(vision_data)
            }
            
            # Combine text with vision data for storage
            combined_text = (
                f"{text}\n\n"
                f"VISION ANALYSIS:\n{json.dumps(vision_data, indent=2)}"
            )
            
            # Add to ChromaDB
            self.db_manager.add_document(
                collection,
                doc_id,
                page_num,
                embedding,
                page_metadata,
                combined_text
            )
            
            return {
                "page_num": page_num,
                "status": "completed",
                "vision_data": vision_data
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return {
                "page_num": page_num,
                "status": "failed",
                "error": str(e)
            }

    def process_pdf(self, 
                   pdf_path: str, 
                   doc_id: Optional[str] = None, 
                   doc_title: Optional[str] = None, 
                   force_refresh: bool = False) -> Dict[str, Any]:
        """Process a PDF document and store embeddings in ChromaDB"""
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Generate doc_title if not provided
        if not doc_title:
            doc_title = os.path.basename(pdf_path)
        
        # Get ChromaDB collection
        collection_name = "sid_documents"
        collection = self.db_manager.get_or_create_collection(collection_name)
        
        # Check if document already exists and hasn't changed
        if not force_refresh:
            exists = self.db_manager.document_exists(collection, doc_id)
            if exists:
                logger.info(f"Document '{doc_title}' already exists in ChromaDB. Use force_refresh=True to reprocess.")
                return {
                    "doc_id": doc_id,
                    "title": doc_title,
                    "status": "skipped"
                }
        
        # Extract text and images from PDF
        text_pages, images = self.pdf_processor.extract_from_path(pdf_path)
        
        # Ensure we have the same number of text pages and images
        if len(text_pages) != len(images):
            logger.warning(f"Mismatch in number of text pages ({len(text_pages)}) and images ({len(images)})")
        
        # Process each page
        num_pages = min(len(text_pages), len(images))
        
        # Document metadata
        doc_metadata = {
            "doc_id": doc_id,
            "title": doc_title,
            "source_path": pdf_path,
            "page_count": num_pages,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fingerprint": self.db_manager.get_document_fingerprint("".join(text_pages))
        }

        # Create a partial function with the common arguments
        process_page = partial(
            self.process_page,
            doc_id=doc_id,
            doc_title=doc_title,
            num_pages=num_pages,
            collection=collection
        )

        # Process pages in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all page processing tasks
            future_to_page = {
                executor.submit(process_page, page_num, images[page_num], text_pages[page_num]): page_num 
                for page_num in range(num_pages)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing page {page_num + 1}/{num_pages}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    results.append({
                        "page_num": page_num,
                        "status": "failed",
                        "error": str(e)
                    })

        # Sort results by page number
        results.sort(key=lambda x: x["page_num"])
        
        # Check if all pages were processed successfully
        failed_pages = [r for r in results if r["status"] == "failed"]
        if failed_pages:
            logger.warning(f"Failed to process {len(failed_pages)} pages")
        
        logger.info(f"Successfully processed document: {doc_title}")
        
        return {
            "doc_id": doc_id,
            "title": doc_title,
            "page_count": num_pages,
            "status": "completed",
            "failed_pages": len(failed_pages),
            "results": results
        }


def process_sid_document(pdf_path: str, doc_id: Optional[str] = None, doc_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to process an SID document without directly instantiating the classes
    
    Args:
        pdf_path: Path to the PDF file
        doc_id: Optional document ID (will be generated if not provided)
        doc_title: Optional document title (will be derived from filename if not provided)
        
    Returns:
        Dictionary with processing results
    """
    processor = SIDProcessor()
    return processor.process_pdf(pdf_path, doc_id, doc_title)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process SID documents with ColPaLI-style embeddings")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--doc-id", help="Optional document ID")
    parser.add_argument("--doc-title", help="Optional document title")
    parser.add_argument("--force-refresh", action="store_true", help="Force reprocessing even if document exists")
    
    args = parser.parse_args()
    
    processor = SIDProcessor()
    result = processor.process_pdf(
        args.pdf_path, 
        args.doc_id, 
        args.doc_title, 
        args.force_refresh
    )
    
    print(json.dumps(result, indent=2))
