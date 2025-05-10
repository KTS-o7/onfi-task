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
from openai import OpenAI

# Import local modules
from models import ChecklistItem, ChecklistEvaluation
from llm_client import call_groq_with_structure, call_gemini_with_structure, RAGEvaluationResponse

# Load environment variables
load_dotenv()

# Global variables
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chromadb_data")

# Initialize Google API client for embeddings using OpenAI client
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("LLM_TEXT_EMBEDDING", "text-embedding-004")

# Create OpenAI client with Google API base URL
embedding_client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


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
    """Get embedding for a text using Google's embedding model through OpenAI client"""
    try:
        # Use OpenAI client with Google's embedding model
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        # Return the embedding values
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero embedding as fallback
        return [0.0] * 768  # Common embedding dimension


def create_document_embedding(doc_id: str, title: str, document_text: str):
    """Create embeddings for a document and store in ChromaDB"""
    ensure_chroma_dir()
    
    # Split document into pages
    pages = split_document_into_pages(document_text)
    
    # Create a collection for the document
    collection_name = f"document_{doc_id}"
    try:
        collection = client.get_collection(collection_name)
        # If collection exists, delete it to refresh content
        client.delete_collection(collection_name)
    except:
        pass
    
    # Create fresh collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"document_title": title}
    )
    
    # Add each page to the collection
    for i, page in enumerate(pages):
        chunk_id = f"{doc_id}_page_{page['metadata']['page_number']}"
        
        # Get embedding for the page content
        embedding = get_embedding(page["content"])
        
        # Add to ChromaDB
        collection.add(
            ids=[chunk_id],
            documents=[page["content"]],
            embeddings=[embedding],
            metadatas=[{"page_number": page["metadata"]["page_number"]}]
        )
    
    return {
        "doc_id": doc_id,
        "title": title,
        "num_pages": len(pages)
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


def evaluate_item_with_rag(doc_id: str, checklist_item: Dict[str, Any]) -> RAGResult:
    """Evaluate a checklist item against a document using RAG"""
    # Create query from checklist item
    query = f"{checklist_item['checklist_title']}: {checklist_item['checklist_description']}"
    
    # Search for relevant pages
    relevant_pages = search_relevant_pages(doc_id, query, top_k=3)
    
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
    
    # Call LLM for evaluation
    try:
        result = call_groq_with_structure(prompt, system_prompt, RAGEvaluationResponse)
        
        if hasattr(result, 'findings_summary') and hasattr(result, 'citations'):
            return RAGResult(
                checklist_item=checklist_item,
                findings_summary=result.findings_summary,
                citations=result.citations,
                compliance_status=result.compliance_status,
                matched_pages=result.matched_pages
            )
        
        # Fallback to Gemini if Groq fails
        result = call_gemini_with_structure(prompt, system_prompt, RAGEvaluationResponse)
        
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


def batch_evaluate_with_rag(doc_id: str, checklist_items: List[Dict[str, Any]], batch_size: int = 10) -> List[RAGResult]:
    """Evaluate multiple checklist items against a document using RAG"""
    results = []
    
    # Process in batches
    for i in range(0, len(checklist_items), batch_size):
        batch = checklist_items[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)...")
        
        # Process each item in the batch
        for item in batch:
            result = evaluate_item_with_rag(doc_id, item)
            results.append(result)
    
    return results 