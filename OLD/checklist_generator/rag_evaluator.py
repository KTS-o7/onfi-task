#!/usr/bin/env python3
"""
RAG-based document evaluation for mutual fund disclosure compliance checker
Uses ChromaDB for vector search and embedding storage
"""

import os
import re
import uuid
import json
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import asyncio
import aiohttp
import time
import hashlib

# Import local modules
from models import ChecklistItem, ChecklistEvaluation
# Comment out Groq and Gemini imports and use OpenAI instead
# from llm_client import call_groq_with_structure, call_gemini_with_structure, RAGEvaluationResponse
from llm_client import call_openai_with_structure, RAGEvaluationResponse

# Load environment variables
load_dotenv()

# Global variables
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromadb_data")

# Initialize OpenAI API client for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-004")

# Create OpenAI client
embedding_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Create async OpenAI client
async_openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Add embedding cache
embedding_cache = {}


class RAGDocument(BaseModel):
    """Model for a document and its chunks"""
    doc_id: str
    title: str
    chunks: List[Dict[str, Any]]


class RAGResult(BaseModel):
    """Model for RAG search results"""
    checklist_item: Dict[str, Any]
    findings_summary: str 
    citations: str
    compliance_status: str
    matched_pages: List[int]


def ensure_chroma_dir():
    """Ensure the ChromaDB persistence directory exists"""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


def extract_page_number(text: str) -> int:
    """Extract page number from page marker"""
    match = re.search(r"--- Page (\d+) ---", text)
    if match:
        return int(match.group(1))
    return 0


def split_document_into_pages(document_text: str) -> List[Dict[str, Any]]:
    """Split document into pages with metadata"""
    pages = []
    current_page = ""
    page_num = 0
    
    lines = document_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("--- Page"):
            if current_page and page_num > 0:
                pages.append({
                    "content": current_page.strip(),
                    "metadata": {"page_number": page_num}
                })
            current_page = ""
            page_num = extract_page_number(line)
        else:
            current_page += line + "\n"
    
    # Add the last page
    if current_page and page_num > 0:
        pages.append({
            "content": current_page.strip(),
            "metadata": {"page_number": page_num}
        })
    
    return pages


def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI's embedding model with caching"""
    # Use cache if available
    cache_key = hash(text[:1000])  # Use first 1000 chars as key to avoid memory issues
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    try:
        # Use OpenAI client with embedding model
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        # Get embedding and cache it
        embedding = response.data[0].embedding
        embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero embedding as fallback
        return [0.0] * 1536  # Common OpenAI embedding dimension


async def get_embedding_async(text: str) -> List[float]:
    """Get embedding asynchronously with caching"""
    # Use cache if available
    cache_key = hash(text[:1000])
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    try:
        # Use async OpenAI client
        response = await async_openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        # Cache and return embedding
        embedding = response.data[0].embedding
        embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"Error getting async embedding: {e}")
        return [0.0] * 1536


def get_document_fingerprint(document_text: str) -> str:
    """Generate a fingerprint for a document to detect changes"""
    # Create a hash of the document content
    return hashlib.md5(document_text.encode('utf-8')).hexdigest()


def create_document_embedding(doc_id: str, title: str, document_text: str, force_refresh: bool = False):
    """Create embeddings for a document and store in ChromaDB"""
    # Add global declaration at the beginning
    global CHROMA_PERSIST_DIR, client
    
    ensure_chroma_dir()
    
    # Generate a document fingerprint
    doc_fingerprint = get_document_fingerprint(document_text)
    
    # Create a collection for the document
    collection_name = f"document_{doc_id}"
    
    try:
        # Check if collection already exists
        collection = client.get_collection(collection_name)
        
        # Check if this is the same document (compare fingerprints)
        collection_metadata = collection.metadata
        if collection_metadata and 'doc_fingerprint' in collection_metadata:
            if not force_refresh and collection_metadata['doc_fingerprint'] == doc_fingerprint:
                # Same document, reuse existing embeddings
                print(f"Reusing existing embeddings for document '{title}' (unchanged content)")
                return {
                    "doc_id": doc_id,
                    "title": title,
                    "num_pages": collection_metadata.get('page_count', 0),
                    "status": "reused"
                }
            else:
                # Different document or forced refresh, delete existing collection
                print(f"Recreating embeddings for document '{title}' (content changed or forced refresh)")
                client.delete_collection(collection_name)
    except Exception as e:
        # Collection doesn't exist or error occurred
        print(f"Creating new collection for document '{title}' (no existing collection)")
    
    # Split document into pages
    pages = split_document_into_pages(document_text)
    
    try:
        # Create fresh collection with document fingerprint in metadata
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "document_title": title,
                "page_count": len(pages),
                "doc_fingerprint": doc_fingerprint,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as db_error:
        if "readonly database" in str(db_error).lower():
            # Remove duplicate global declaration here
            import tempfile
            
            # Use a temporary directory instead
            old_dir = CHROMA_PERSIST_DIR
            CHROMA_PERSIST_DIR = tempfile.mkdtemp(prefix="chromadb_")
            print(f"Database is readonly, switching to temporary directory: {CHROMA_PERSIST_DIR}")
            
            # Create a new client
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            
            # Create collection with the new client
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "document_title": title,
                    "page_count": len(pages),
                    "doc_fingerprint": doc_fingerprint,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        else:
            raise
    
    print(f"Creating embeddings for document '{title}' ({len(pages)} pages)")
    
    # Process pages in batches for better performance
    batch_size = 5
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i+batch_size]
        ids = []
        docs = []
        embeddings = []
        metadatas = []
        
        # Process batch
        for page in batch:
            chunk_id = f"{doc_id}_page_{page['metadata']['page_number']}"
            # Get embedding for the page content
            embedding = get_embedding(page["content"])
            
            ids.append(chunk_id)
            docs.append(page["content"])
            embeddings.append(embedding)
            metadatas.append({"page_number": page["metadata"]["page_number"]})
        
        # Add batch to ChromaDB
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    return {
        "doc_id": doc_id,
        "title": title,
        "num_pages": len(pages),
        "status": "created"
    }


def search_relevant_pages(doc_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant pages in a document using embeddings"""
    # Get the collection for the document
    collection_name = f"document_{doc_id}"
    try:
        collection = client.get_collection(collection_name)
    except:
        raise ValueError(f"Document {doc_id} not found in ChromaDB")
    
    # Get embedding for query
    query_embedding = get_embedding(query)
    
    # Search for relevant pages
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Process results
    relevant_pages = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0],
        results["distances"][0]
    )):
        relevant_pages.append({
            "content": doc,
            "page_number": metadata["page_number"],
            "relevance_score": 1.0 - distance  # Convert distance to similarity score
        })
    
    # Sort by page number
    relevant_pages.sort(key=lambda x: x["page_number"])
    
    return relevant_pages


async def call_openai_with_structure_async(prompt, system_prompt, schema_model):
    """Async version of call_openai_with_structure"""
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Use async OpenAI client
        completion = await async_openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        # Parse JSON response
        response_content = completion.choices[0].message.content
        json_data = json.loads(response_content)
        
        # Preprocess the JSON data - THIS IS THE KEY FIX
        # Convert citations from list to string if needed
        if 'citations' in json_data and isinstance(json_data['citations'], list):
            json_data['citations'] = '\n\n'.join(str(item) for item in json_data['citations'])
            
        # Ensure all required fields exist
        if 'findings_summary' not in json_data:
            json_data['findings_summary'] = "No findings summary available"
        if 'citations' not in json_data:
            json_data['citations'] = "No citations found"
        if 'compliance_status' not in json_data:
            json_data['compliance_status'] = "Unknown"
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
        
        # Create and return model instance
        try:
            return schema_model(**json_data)
        except Exception as e:
            print(f"Error creating schema model: {e}")
            # Create a default model as fallback
            return schema_model(
                findings_summary="Could not process model",
                citations="Error in model creation",
                compliance_status="Unknown",
                matched_pages=[]
            )
    except Exception as e:
        print(f"Error in async OpenAI call: {e}")
        # Return default model instance
        return schema_model(
            findings_summary="Error in API call",
            citations="No citations available due to error",
            compliance_status="Unknown",
            matched_pages=[]
        )


async def evaluate_item_with_rag_async(doc_id: str, checklist_item: Dict[str, Any]) -> RAGResult:
    """Async version of evaluate_item_with_rag"""
    # Create query from checklist item
    query = f"{checklist_item['checklist_title']}: {checklist_item['checklist_description']}"
    
    # Search for relevant pages
    try:
        relevant_pages = search_relevant_pages(doc_id, query, top_k=7)
    except Exception as e:
        print(f"Error searching relevant pages: {e}")
        return RAGResult(
            checklist_item=checklist_item,
            findings_summary="Error searching document",
            citations="",
            compliance_status="Unknown",
            matched_pages=[]
        )
    
    if not relevant_pages:
        return RAGResult(
            checklist_item=checklist_item,
            findings_summary="No relevant pages found",
            citations="",
            compliance_status="Unknown",
            matched_pages=[]
        )
    
    # Extract content and page numbers
    context = "\n\n".join([f"[Page {p['page_number']}] {p['content']}" for p in relevant_pages])
    page_numbers = [p["page_number"] for p in relevant_pages]
    
    # Prepare system prompt
    system_prompt = """
    You are an expert in mutual fund regulations and compliance. 
    Your task is to evaluate whether a document properly addresses a specific disclosure requirement.
    
    Be flexible in your evaluation - the document may satisfy requirements using different 
    wording or structure than what's literally specified in the requirement.
    
    For Partially Compliant: Use when core requirements are addressed but some details are missing.
    For Non-Compliant: Use ONLY when the requirement is clearly absent or contradicted.
    When in doubt, prefer "Partially Compliant" over "Non-Compliant".
    
    Return your assessment as a valid JSON object with these fields:
    - findings_summary: Brief assessment of compliance (1-2 sentences)
    - citations: Direct quotes from the document supporting your conclusion (if found)
    - compliance_status: Either "Compliant", "Partially Compliant", or "Non-Compliant"
    - matched_pages: List of page numbers where relevant information was found
    
    Format your response ONLY as a valid JSON object.
    """
    
    # Prepare evaluation prompt
    prompt = f"""
    Evaluate whether the following mutual fund document properly addresses this SEBI/AMFI disclosure requirement:
    
    REQUIREMENT TITLE: {checklist_item['checklist_title']}
    REQUIREMENT DESCRIPTION: {checklist_item['checklist_description']}
    
    The following text was extracted from pages {page_numbers} of the document:
    
    {context}
    
    Evaluate whether these pages adequately address the requirement. Include specific citations as direct quotes.
    For matched_pages, only include pages that actually contain relevant information.
    """
    
    # Call OpenAI for evaluation using async method
    try:
        result = await call_openai_with_structure_async(prompt, system_prompt, RAGEvaluationResponse)
        
        if hasattr(result, 'findings_summary') and hasattr(result, 'citations'):
            return RAGResult(
                checklist_item=checklist_item,
                findings_summary=result.findings_summary,
                citations=result.citations,
                compliance_status=result.compliance_status,
                matched_pages=result.matched_pages
            )
    except Exception as e:
        print(f"Error evaluating with RAG async: {e}")
    
    # Default response if everything fails
    return RAGResult(
        checklist_item=checklist_item,
        findings_summary="Could not evaluate compliance",
        citations="No citations found",
        compliance_status="Unknown",
        matched_pages=page_numbers
    )


def evaluate_item_with_rag(doc_id: str, checklist_item: Dict[str, Any]) -> RAGResult:
    """Evaluate a checklist item against a document using RAG"""
    # Create query from checklist item
    query = f"{checklist_item['checklist_title']}: {checklist_item['checklist_description']}"
    
    # Search for relevant pages
    relevant_pages = search_relevant_pages(doc_id, query, top_k=7)
    
    if not relevant_pages:
        return RAGResult(
            checklist_item=checklist_item,
            findings_summary="No relevant pages found",
            citations="",
            compliance_status="Unknown",
            matched_pages=[]
        )
    
    # Extract content and page numbers
    context = "\n\n".join([f"[Page {p['page_number']}] {p['content']}" for p in relevant_pages])
    page_numbers = [p["page_number"] for p in relevant_pages]
    
    # Prepare system prompt
    system_prompt = """
    You are an expert in mutual fund regulations and compliance. 
    Your task is to evaluate whether a document properly addresses a specific disclosure requirement.
    
    Be flexible in your evaluation - the document may satisfy requirements using different 
    wording or structure than what's literally specified in the requirement.
    
    For Partially Compliant: Use when core requirements are addressed but some details are missing.
    For Non-Compliant: Use ONLY when the requirement is clearly absent or contradicted.
    When in doubt, prefer "Partially Compliant" over "Non-Compliant".
    
    Return your assessment as a valid JSON object with these fields:
    - findings_summary: Brief assessment of compliance (1-2 sentences)
    - citations: Direct quotes from the document supporting your conclusion (if found)
    - compliance_status: Either "Compliant", "Partially Compliant", or "Non-Compliant"
    - matched_pages: List of page numbers where relevant information was found
    
    Format your response ONLY as a valid JSON object.
    """
    
    # Prepare evaluation prompt
    prompt = f"""
    Evaluate whether the following mutual fund document properly addresses this SEBI/AMFI disclosure requirement:
    
    REQUIREMENT TITLE: {checklist_item['checklist_title']}
    REQUIREMENT DESCRIPTION: {checklist_item['checklist_description']}
    
    The following text was extracted from pages {page_numbers} of the document:
    
    {context}
    
    Evaluate whether these pages adequately address the requirement. Include specific citations as direct quotes.
    For matched_pages, only include pages that actually contain relevant information.
    """
    
    # Call OpenAI for evaluation
    try:
        result = call_openai_with_structure(prompt, system_prompt, RAGEvaluationResponse)
        
        if hasattr(result, 'findings_summary') and hasattr(result, 'citations'):
            return RAGResult(
                checklist_item=checklist_item,
                findings_summary=result.findings_summary,
                citations=result.citations,
                compliance_status=result.compliance_status,
                matched_pages=result.matched_pages
            )
    except Exception as e:
        print(f"Error evaluating with RAG: {e}")
    
    # Default response if everything fails
    return RAGResult(
        checklist_item=checklist_item,
        findings_summary="Could not evaluate compliance",
        citations="No citations found",
        compliance_status="Unknown",
        matched_pages=page_numbers
    )


async def batch_evaluate_with_rag_async(doc_id: str, checklist_items: List[Dict[str, Any]], batch_size: int = 10) -> List[RAGResult]:
    """Async evaluation of multiple checklist items against a document using RAG"""
    # Use a semaphore to avoid overwhelming the API
    semaphore = asyncio.Semaphore(8)  # Allow 8 concurrent API calls
    all_results = [None] * len(checklist_items)  # Pre-allocate results list
    
    async def process_item(index, item):
        async with semaphore:
            try:
                return await evaluate_item_with_rag_async(doc_id, item)
            except Exception as e:
                print(f"Error processing item {index}: {e}")
                return RAGResult(
                    checklist_item=item,
                    findings_summary="Error during processing",
                    citations="An error occurred",
                    compliance_status="Unknown",
                    matched_pages=[]
                )
    
    # Create all tasks at once
    all_tasks = []
    for i, item in enumerate(checklist_items):
        all_tasks.append(process_item(i, item))
    
    # Process in true batches for progress reporting
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    print(f"Processing {len(all_tasks)} items in {total_batches} batches with up to 8 concurrent operations")
    
    # Process all tasks with progress reporting
    completed = 0
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:min(i+batch_size, len(all_tasks))]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        
        # Store results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"Task error: {result}")
                idx = i + j
                all_results[idx] = RAGResult(
                    checklist_item=checklist_items[idx],
                    findings_summary="Task failed with error",
                    citations="An exception occurred",
                    compliance_status="Unknown",
                    matched_pages=[]
                )
            else:
                all_results[i + j] = result
        
        # Update progress
        completed += len(batch)
        print(f"Progress: {completed}/{len(all_tasks)} items completed ({(completed/len(all_tasks)*100):.1f}%)")
    
    # Remove any None values (shouldn't happen, but just in case)
    return [r for r in all_results if r is not None]


def batch_evaluate_with_rag(doc_id: str, checklist_items: List[Dict[str, Any]], batch_size: int = 10) -> List[RAGResult]:
    """Evaluate multiple checklist items against a document using RAG"""
    if len(checklist_items) == 0:
        return []
        
    print(f"Starting evaluation of {len(checklist_items)} items using async processing")
    
    # Use the asyncio event loop to run the async function
    try:
        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function and get the results
        results = loop.run_until_complete(batch_evaluate_with_rag_async(doc_id, checklist_items, batch_size))
        print(f"Async processing completed successfully for {len(results)}/{len(checklist_items)} items")
        return results
    except Exception as e:
        print(f"Error in async batch evaluation: {e}")
        
        # Fallback to synchronous processing if async fails
        print("Falling back to synchronous processing...")
        results = []
        for i in range(0, len(checklist_items), batch_size):
            end_idx = min(i + batch_size, len(checklist_items))
            batch = checklist_items[i:end_idx]
            print(f"Processing items {i+1}-{end_idx} of {len(checklist_items)}...")
            
            for item in batch:
                try:
                    result = evaluate_item_with_rag(doc_id, item)
                    results.append(result)
                except Exception as item_e:
                    print(f"Error evaluating item: {item_e}")
                    results.append(RAGResult(
                        checklist_item=item,
                        findings_summary="Error in evaluation",
                        citations="An error occurred during evaluation",
                        compliance_status="Unknown",
                        matched_pages=[]
                    ))
        
        return results 