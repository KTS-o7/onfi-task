import os
import time
import uuid
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import logging
import chromadb
from pydantic import BaseModel
from typing import List
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import concurrent.futures
from functools import partial

MAX_WORKERS = 5

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

MAX_HYDE_ITERATIONS = 1

# Import local modules
from pdf_ocr import SIDProcessor, process_sid_document
from llm_service import (
    LLMServiceFactory, 
)
from json_models import (
    ExtractedChecklistItem, 
    ChecklistItem,
    ExtractedChecklistItemList, 
    SearchResult, 
    FinalComplianceReport,
    ComplianceReportResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Global constants
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
ALLOWED_EXTENSIONS = {'pdf'}
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chromadb_data")
CHECKLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "checklist.json")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"), exist_ok=True)

# Flask configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Background task status tracking
processing_tasks = {}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_services():
    """Create and return services for LLM processing"""
    # Create OpenAI and Gemini services
    openai_service = LLMServiceFactory.create_openai_service(model_name=OPENAI_INF_MODEL)
    gemini_service = LLMServiceFactory.create_gemini_service(model_name=GEMINI_PRO_MODEL)
    logger.info(f"Using OpenAI service: {openai_service}")
    logger.info(f"Using Gemini service: {gemini_service}")
    
    # Create HyDE search service using Gemini for queries and OpenAI for embeddings
    hyde_search_service = LLMServiceFactory.create_hyde_search_service(
        llm_service=openai_service, # Can be changed to gemini_service
        embedding_service=openai_service
    )
    
    # Create compliance evaluation service using OpenAI
    compliance_service = LLMServiceFactory.create_compliance_evaluation_service(
        llm_service=openai_service
    )
    
    return {
        "openai": openai_service,
        "gemini": gemini_service,
        "hyde_search": hyde_search_service,
        "compliance": compliance_service
    }

class ChecklistGenerator:
    """Class to handle checklist generation from master circular"""
    
    def __init__(self, llm_service):
        """Initialize checklist generator with LLM service"""
        self.llm_service = llm_service
    
    def extract_checklist_from_text(self, text: str, page_range: str) -> List[ExtractedChecklistItem]:
        """Extract checklist items from text"""
        prompt = f"""
        Extract mutual fund disclosure requirements from this SEBI master circular text. 
        Focus on specific requirements that must be disclosed in a mutual fund's Scheme Information Document (SID).
        
        TEXT:
        ---
        {text}
        ---
        
        For each requirement, provide:
        1. A short title summarizing the requirement
        2. A detailed description of what needs to be disclosed
        3. The rationale from the SEBI circular
        
        You must format your response in this EXACT JSON structure:
        {{
          "item_list": [
            {{
              "checklist_title": "Example Title",
              "checklist_description": "Example Description",
              "rationale": "Example Rationale"
            }},
          ]
        }}
        
        This structure is required for processing. The output must be a single JSON object with only the 'item_list' key containing an array of items.
        """
        
        system_prompt = """
        You are an expert in SEBI regulations for mutual funds.
        Your task is to identify disclosure requirements that mutual funds must follow in their 
        Scheme Information Documents (SIDs).
        
        Focus on extracting actionable and specific requirements.
        Ensure the detailed description clearly specifies what needs to be disclosed.
        
        Your response MUST be a valid JSON object with a single property named 'item_list' that contains an array of checklist items.
        """
        
        try:
            result = self.llm_service.generate_structured_output(
                prompt, 
                system_prompt, 
                output_schema=ExtractedChecklistItemList
            )
            
            # Access the list via the item_list attribute
            if hasattr(result, "item_list"):
                return result.item_list
            
            # Fall back to checking if result is a dict with an item_list field
            if isinstance(result, dict) and "item_list" in result:
                items = result["item_list"]
                if isinstance(items, list):
                    return [ExtractedChecklistItem(**item) for item in items]
            
            # If we can't process it, raise an exception
            raise ValueError(f"Could not extract a valid item_list from LLM output: {result}")
            
        except Exception as e:
            logger.error(f"Error in extract_checklist_from_text: {e}")
            # Return empty list instead of propagating error
            return []
    
    def generate_checklist_from_pdf(self, pdf_path: str) -> List[ChecklistItem]:
        """Generate checklist from PDF document"""
        from pypdf import PdfReader
        
        # Read PDF text
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            checklist_items = []
            
            # Process pages in batches of 5
            batch_size = 40
            total_pages = len(pdf.pages)
            
            for batch_start in range(0, total_pages, batch_size):
                # Get the end index for this batch
                batch_end = min(batch_start + batch_size, total_pages)
                
                # Combine text from all pages in this batch
                batch_text = ""
                page_numbers = []
                total_text_length = 0
                
                for i in range(batch_start, batch_end):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    
                    # Skip pages with very little content
                    if len(text.strip()) < 100:
                        continue

                    total_text_length += len(text.strip())
                    batch_text += f"\n--- Page {i+1} ---\n{text}\n"
                    page_numbers.append(i+1)
                
                print("--------------------------------")
                print("Total text length",total_text_length)
                print("--------------------------------")
                # Skip if no valid content in this batch
                if not batch_text.strip():
                    continue
                
                # Create page range string
                page_range = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
                
                try:
                    # Extract requirements from combined batch text
                    extracted_items = self.extract_checklist_from_text(batch_text, page_range)
                    logger.info(f"Extracted {len(extracted_items)} items from {page_range}")
                    
                    # Convert to ChecklistItem objects
                    for item in extracted_items:
                        checklist_item = ChecklistItem(
                            content=item,
                            page_numbers=page_range,
                            source_document="SEBI Master Circular"
                        )
                        checklist_items.append(checklist_item)
                        
                except Exception as e:
                    logger.error(f"Error extracting requirements from {page_range}: {e}")
            
            return checklist_items

def process_master_circular(task_id: str, pdf_path: str):
    """Background task to process master circular and generate checklist"""
    try:
        processing_tasks[task_id]["status"] = "processing"
        
        # Get OpenAI service for checklist generation
        services = get_services()
        checklist_generator = ChecklistGenerator(services["openai"])
        
        # Generate checklist
        checklist_items = checklist_generator.generate_checklist_from_pdf(pdf_path)
        
        # Save checklist to file
        checklist_data = [item.model_dump() for item in checklist_items]
        with open(CHECKLIST_PATH, 'w') as f:
            json.dump(checklist_data, f, indent=2)
        
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = {
            "checklist_count": len(checklist_items),
            "checklist_path": CHECKLIST_PATH
        }
        
    except Exception as e:
        logger.error(f"Error processing master circular: {e}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)

def process_checklist_item(checklist_item, hyde_service, compliance_service, collection, doc_result):
    """Process a single checklist item and return its evaluation results"""
    # Initialize findings for this checklist item
    all_findings = []
    max_iterations = MAX_HYDE_ITERATIONS
    
    # Start with initial HyDE query
    hyde_query = hyde_service.generate_hyde_query(checklist_item)
    query_text = hyde_query.search_query
    
    # Track all queries used
    queries_used = [f"Initial query: {query_text}"]
    
    # Iterative HyDE search
    for iteration in range(max_iterations):
        # Search ChromaDB with the query
        search_results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={"doc_id": doc_result["doc_id"]}
        )
        
        # Convert search results to SearchResult objects
        result_objects = []
        if search_results["documents"] and len(search_results["documents"][0]) > 0:
            for j, (doc, metadata, distance) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["distances"][0]
            )):
                result_objects.append(SearchResult(
                    content=doc,
                    page_number=metadata["page_number"],
                    relevance_score=1.0 - distance
                ))
        
        # Evaluate compliance based on search results
        evaluation = compliance_service.evaluate_compliance(checklist_item, result_objects)
        
        # Add to findings
        finding = {
            "iteration": iteration + 1,
            "query": query_text,
            "findings_summary": evaluation.findings_summary,
            "citations": [c.model_dump() for c in evaluation.citations],
            "compliance_status": evaluation.compliance_status.status,
            "matched_pages": evaluation.matched_pages
        }
        all_findings.append(finding)
        
        # Check if we need to continue with more iterations
        if evaluation.compliance_status.status == "Compliant" or iteration == max_iterations - 1:
            break
        
        # Analyze gap and generate improved query for next iteration
        gap_analysis = hyde_service.analyze_search_gap(checklist_item, all_findings)
        
        if gap_analysis.is_sufficient:
            break
        
        # Generate new query for next iteration
        query_text = gap_analysis.suggested_query
        queries_used.append(f"Iteration {iteration+2}: {query_text}")
    
    # Use the final evaluation as the result
    final_evaluation = compliance_service.evaluate_compliance(checklist_item, result_objects)
    
    # Create and return report item
    return {
        "report_item": FinalComplianceReport(
            compliance_requirement=f"{checklist_item.content.checklist_title}: {checklist_item.content.checklist_description}",
            rationale=checklist_item.content.rationale,
            source=f"{checklist_item.source_document} {checklist_item.page_numbers}",
            compliance_status=final_evaluation.compliance_status.status,
            findings_summary=final_evaluation.findings_summary,
            findings_citations=[citation.page_number for citation in final_evaluation.citations]
        ),
        "status": final_evaluation.compliance_status.status
    }

def process_sid_evaluation(task_id: str, pdf_path: str, checklist_items: list[ChecklistItem]):
    """Background task to process SID document and evaluate against checklist"""
    try:
        processing_tasks[task_id]["status"] = "processing"
        
        # Step 1: Process SID document with OCR and generate embeddings
        sid_processor = SIDProcessor()
        doc_result = sid_processor.process_pdf(pdf_path)
        
        # Update task status
        processing_tasks[task_id]["status"] = "evaluating"
        processing_tasks[task_id]["progress"] = 0.5
        
        # Get services for compliance evaluation
        services = get_services()
        hyde_service = services["hyde_search"]
        compliance_service = services["compliance"]
        
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Create embedding function
        embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL
        )

        # Get collection with the explicit embedding function
        collection = chroma_client.get_collection(
            name="sid_documents",
            embedding_function=embedding_func
        )
        
        # Create a partial function with the common arguments
        process_item = partial(
            process_checklist_item,
            hyde_service=hyde_service,
            compliance_service=compliance_service,
            collection=collection,
            doc_result=doc_result
        )
        
        # Process items in parallel
        report_items = []
        compliant_count = 0
        non_compliant_count = 0
        partial_compliant_count = 0
        unknown_count = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_item, item): item 
                for item in checklist_items
            }
            
            # Process results as they complete
            total_items = len(checklist_items)
            completed_items = 0
            
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    report_items.append(result["report_item"])
                    
                    # Update counts
                    if result["status"] == "Compliant":
                        compliant_count += 1
                    elif result["status"] == "Non-Compliant":
                        non_compliant_count += 1
                    elif result["status"] == "Partially Compliant":
                        partial_compliant_count += 1
                    else:
                        unknown_count += 1
                    
                    # Update progress
                    completed_items += 1
                    processing_tasks[task_id]["progress"] = 0.5 + (completed_items / total_items) * 0.5
                    
                except Exception as e:
                    logger.error(f"Error processing checklist item: {e}")
                    # Continue with other items even if one fails
        
        # Step 3: Prepare final report
        compliance_report = ComplianceReportResponse(
            report_items=report_items,
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            partial_compliant_count=partial_compliant_count,
            unknown_count=unknown_count
        )
        
        # Update task status
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = compliance_report.model_dump()
        
    except Exception as e:
        logger.error(f"Error processing SID document: {e}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/upload/master-circular', methods=['POST'])
def upload_master_circular():
    """Endpoint to upload master circular PDF and generate checklist"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if user submitted an empty form
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Create a task ID
        #task_id = str(uuid.uuid4())
        task_id = "123"
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # Create task entry
        processing_tasks[task_id] = {
            "type": "master_circular",
            "filename": filename,
            "path": file_path,
            "status": "queued",
            "created_at": datetime.now().isoformat()
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_master_circular,
            args=(task_id, file_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "task_id": task_id, 
            "status": "queued",
            "message": "Master circular upload received and processing started"
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/upload/sid-document', methods=['POST'])
def upload_sid_document():
    """Endpoint to upload SID document PDF for evaluation"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if user submitted an empty form
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if we have a checklist to evaluate against
    if not os.path.exists(CHECKLIST_PATH):
        return jsonify({"error": "No checklist available. Please upload a master circular first"}), 400
    
    if file and allowed_file(file.filename):
        # Create a task ID
        #task_id = str(uuid.uuid4())
        task_id="456"
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        print("--------------------------------")
        print(filename)
        print("--------------------------------")
        
        # Check if this file already exists in the collection
        try:
            # Connect to ChromaDB to check for existing document
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            collection = chroma_client.get_collection("sid_documents")
            
            # Query for documents with this filename
            results = collection.get(
                where={"title": {"$in": [filename]}},
                limit=1
            )
            print(results)
            
            # If we found a match, use its doc_id directly for evaluation
            if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                existing_doc_id = results["metadatas"][0].get("doc_id")
                
                if existing_doc_id:
                    logger.info(f"Document already processed, using existing doc_id: {existing_doc_id}")
                    
                    # Create task entry for direct evaluation
                    processing_tasks[task_id] = {
                        "type": "sid_document_evaluation",
                        "filename": filename,
                        "path": file_path,
                        "status": "queued",
                        "created_at": datetime.now().isoformat(),
                        "progress": 0.25,  # Skip the OCR/embedding part
                        "doc_id": existing_doc_id
                    }
                    
                    # Load checklist
                    with open(CHECKLIST_PATH, 'r') as f:
                        checklist_data = json.load(f)
                    
                    # Convert to ChecklistItem objects
                    checklist_items = [ChecklistItem.model_validate(item) for item in checklist_data]
                    
                    # Start background evaluation directly using existing doc_id
                    thread = threading.Thread(
                        target=process_sid_evaluation,
                        args=(task_id, existing_doc_id, checklist_items)
                    )
                    thread.daemon = True
                    thread.start()
                    
                    return jsonify({
                        "task_id": task_id, 
                        "status": "queued",
                        "message": "Document already processed, skipping to evaluation"
                    })
        
        except Exception as e:
            logger.warning(f"Error checking for existing document: {e}, proceeding with full processing")
        
        # If we get here, either the document doesn't exist or there was an error checking
        # So we proceed with normal processing
        processing_tasks[task_id] = {
            "type": "sid_document",
            "filename": filename,
            "path": file_path,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "progress": 0.0
        }
        
        # Load checklist
        with open(CHECKLIST_PATH, 'r') as f:
            checklist_data = json.load(f)
        
        # Convert to ChecklistItem objects
        checklist_items = [ChecklistItem.model_validate(item) for item in checklist_data]
        
        # Start background processing
        thread = threading.Thread(
            target=process_sid_evaluation,
            args=(task_id, file_path, checklist_items)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "task_id": task_id, 
            "status": "queued",
            "message": "SID document upload received and processing started"
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Endpoint to check the status of a processing task"""
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = processing_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "type": task.get("type"),
        "status": task.get("status"),
        "created_at": task.get("created_at")
    }
    
    # Include progress if available
    if "progress" in task:
        response["progress"] = task["progress"]
    
    # Include result if completed
    if task.get("status") == "completed" and "result" in task:
        response["result"] = task["result"]
    
    # Include error if failed
    if task.get("status") == "failed" and "error" in task:
        response["error"] = task["error"]
    
    return jsonify(response)

@app.route('/api/checklist', methods=['GET'])
def get_checklist():
    """Endpoint to get the generated checklist"""
    if not os.path.exists(CHECKLIST_PATH):
        return jsonify({"error": "No checklist available"}), 404
    
    with open(CHECKLIST_PATH, 'r') as f:
        checklist_data = json.load(f)
    
    return jsonify({"checklist": checklist_data})

@app.route('/api/documents', methods=['GET'])
def get_processed_documents():
    """Endpoint to get a list of processed documents in ChromaDB"""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = chroma_client.get_collection("sid_documents")
        
        # Get all unique doc_ids
        results = collection.get(
            include=["metadatas"],
            limit=1000  # Adjust as needed
        )
        
        # Extract unique documents from metadatas
        documents = {}
        for metadata in results["metadatas"]:
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "page_count": metadata.get("total_pages", 0)
                }
        
        return jsonify({"documents": list(documents.values())})
    
    except Exception as e:
        logger.error(f"Error getting processed documents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate-checklist/<doc_id>', methods=['GET'])
def evaluate_document_with_id(doc_id):
    """Endpoint to evaluate a specific document against the checklist"""
    # Check if we have a checklist to evaluate against
    if not os.path.exists(CHECKLIST_PATH):
        return jsonify({"error": "No checklist available. Please upload a master circular first"}), 400
    
    try:
        # Load checklist
        with open(CHECKLIST_PATH, 'r') as f:
            checklist_data = json.load(f)
        
        # Convert to ChecklistItem objects
        checklist_items = [ChecklistItem.model_validate(item) for item in checklist_data]
        
        # Create a task ID
        #task_id = str(uuid.uuid4())
        task_id="456"
        # Create task entry
        processing_tasks[task_id] = {
            "type": "sid_evaluation",
            "doc_id": doc_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "progress": 0.0
        }
        
        # Get document path from ChromaDB
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = chroma_client.get_collection("sid_documents")
        
        # Get metadata for the first page to find the file path
        results = collection.get(
            ids=[f"{doc_id}_page_0"],
            include=["metadatas"]
        )
        
        if not results["metadatas"] or len(results["metadatas"]) == 0:
            return jsonify({"error": "Document not found in the database"}), 404
        
        # Start background processing using doc_id directly
        thread = threading.Thread(
            target=process_sid_evaluation,
            args=(task_id, doc_id, checklist_items)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "task_id": task_id, 
            "status": "queued",
            "message": "Document evaluation started"
        })
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-tasks', methods=['POST'])
def clear_completed_tasks():
    """Endpoint to clear completed or failed tasks"""
    to_remove = []
    
    for task_id, task in processing_tasks.items():
        if task["status"] in ["completed", "failed"]:
            # Delete the file if it exists
            if "path" in task and os.path.exists(task["path"]):
                try:
                    os.remove(task["path"])
                except Exception as e:
                    logger.warning(f"Could not remove file {task['path']}: {e}")
            
            to_remove.append(task_id)
    
    # Remove tasks
    for task_id in to_remove:
        del processing_tasks[task_id]
    
    return jsonify({
        "message": f"Cleared {len(to_remove)} completed or failed tasks",
        "cleared_count": len(to_remove)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
