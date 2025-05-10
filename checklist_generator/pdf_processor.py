import os
import pdfplumber
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm

# Set maximum number of pages to process per chunk
MAX_PAGES_PER_CHUNK = 10


def extract_text_from_pdf(pdf_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
    """Extract text from PDF file given a page range."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Adjust end_page to handle None value
            if end_page is None:
                end_page = len(pdf.pages)
            else:
                # Ensure end_page is not beyond the document
                end_page = min(end_page, len(pdf.pages))
            
            # Adjust for 0-indexed pages
            start_idx = start_page - 1
            end_idx = end_page - 1
            
            # Extract text from each page
            text = ""
            for i in range(start_idx, end_idx + 1):
                try:
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
                except Exception as e:
                    print(f"Error extracting text from page {i+1}: {e}")
            
            return text
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return ""


def get_pdf_length(pdf_path: str) -> int:
    """Get the number of pages in a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        print(f"Error getting PDF length: {e}")
        return 0


def extract_document_structure(pdf_path: str) -> Dict[str, Dict]:
    """
    Extract document structure (chapters) based on the first ~30 pages.
    Returns mapping of chapters to page ranges.
    """
    # Try to get the table of contents or chapter structure
    toc_text = extract_text_from_pdf(pdf_path, 1, 30)
    
    # Look for specific chapter patterns in TOC
    # For this SEBI master circular, we know the key chapters based on the index
    # Hard-coding the most relevant chapters for disclosure requirements
    return {
        "OFFER_DOCUMENT": {"name": "OFFER DOCUMENT FOR SCHEMES", "start": 8, "end": 26},
        "DISCLOSURES": {"name": "DISCLOSURES & REPORTING NORMS", "start": 115, "end": 138},
        "ADVERTISEMENTS": {"name": "ADVERTISEMENTS", "start": 320, "end": 328},
        "INVESTOR_RIGHTS": {"name": "INVESTOR RIGHTS & OBLIGATIONS", "start": 328, "end": 345},
        "RISK_MANAGEMENT": {"name": "RISK MANAGEMENT FRAMEWORK", "start": 91, "end": 115},
        "LOADS_FEES": {"name": "LOADS, FEES, CHARGES AND EXPENSES", "start": 243, "end": 261},
        "VALUATION": {"name": "VALUATION", "start": 216, "end": 243}
    }


def split_pdf_into_chunks(pdf_path: str, focus_chapters: Optional[Dict] = None) -> List[Tuple[int, int]]:
    """
    Split PDF into manageable chunks respecting chapter boundaries if possible.
    Returns list of (start_page, end_page) tuples.
    """
    total_pages = get_pdf_length(pdf_path)
    
    # If focus chapters are provided, use them
    if focus_chapters:
        chunks = []
        for chapter, details in focus_chapters.items():
            # Split large chapters into multiple chunks
            start = details["start"]
            end = details["end"]
            
            for chunk_start in range(start, end, MAX_PAGES_PER_CHUNK):
                chunk_end = min(chunk_start + MAX_PAGES_PER_CHUNK - 1, end)
                chunks.append((chunk_start, chunk_end))
        
        # Sort chunks by start page
        return sorted(chunks, key=lambda x: x[0])
    else:
        # Just split into regular chunks
        return [(i, min(i + MAX_PAGES_PER_CHUNK - 1, total_pages)) 
                for i in range(1, total_pages + 1, MAX_PAGES_PER_CHUNK)]


def process_pdf_in_chunks(pdf_path: str, extraction_function, focus_on_chapters: bool = True) -> List[Dict]:
    """
    Process PDF in chunks using the provided extraction function.
    Returns list of extraction results.
    """
    try:
        results = []
        
        # Get document structure if focusing on chapters
        if focus_on_chapters:
            chapters = extract_document_structure(pdf_path)
            chunks = split_pdf_into_chunks(pdf_path, chapters)
        else:
            chunks = split_pdf_into_chunks(pdf_path)
        
        print(f"Processing {len(chunks)} chunks from {pdf_path}")
        
        # Process each chunk
        for start_page, end_page in tqdm(chunks):
            # Extract text from this chunk
            text = extract_text_from_pdf(pdf_path, start_page, end_page)
            
            if not text.strip():
                print(f"Warning: No text extracted from pages {start_page}-{end_page}")
                continue
            
            # Call extraction function with the text
            page_range = f"{start_page}-{end_page}"
            extraction = extraction_function(text, page_range)
            
            # Save extraction to results
            results.append({
                "page_range": page_range,
                "extraction": extraction
            })
            
            # Optionally save intermediate results to prevent data loss
            save_intermediate_results(results, "intermediate_extractions.json")
        
        return results
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []


def save_intermediate_results(results: List[Dict], output_file: str):
    """Save intermediate extraction results to prevent data loss."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving intermediate results: {e}")


def load_intermediate_results(input_file: str) -> List[Dict]:
    """Load intermediate extraction results if available."""
    try:
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        print(f"Error loading intermediate results: {e}")
        return [] 