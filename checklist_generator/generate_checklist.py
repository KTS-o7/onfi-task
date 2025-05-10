#!/usr/bin/env python3
"""
Checklist Generator for Mutual Fund Disclosure Requirements
Processes SEBI and AMFI regulatory documents to extract disclosure requirements
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import time
from dotenv import load_dotenv
import logging

# Import local modules
from pdf_processor import process_pdf_in_chunks, load_intermediate_results
from llm_client import extract_requirements_from_text, consolidate_checklist
from models import ChecklistItem, RawExtraction, MutualFundChecklist

# Load environment variables
load_dotenv()

# Default file paths
SEBI_PDF_PATH = "../SEBI_master_circular.pdf"
OUTPUT_DIR = "../circular_evals"
INTERMEDIATE_FILE = "intermediate_extractions.json"
RAW_CHECKLIST_FILE = "raw_checklist_items.json"
FINAL_CHECKLIST_FILE = "mutual_fund_disclosure_checklist.json"

# Suppress pdfminer warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)


def ensure_output_dir(dir_path: str):
    """Ensure output directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def process_regulatory_document(
    pdf_path: str, 
    output_dir: str,
    resume_from_intermediate: bool = True
) -> List[Dict]:
    """Process regulatory document to extract disclosure requirements"""
    ensure_output_dir(output_dir)
    
    intermediate_file_path = os.path.join(output_dir, INTERMEDIATE_FILE)
    
    # Check if we can resume from intermediate results
    if resume_from_intermediate and os.path.exists(intermediate_file_path):
        print(f"Resuming from intermediate results in {intermediate_file_path}")
        return load_intermediate_results(intermediate_file_path)
    
    # Process PDF in chunks using extract_requirements_from_text
    return process_pdf_in_chunks(pdf_path, extract_requirements_from_text, focus_on_chapters=True)


def batch_consolidate(items: List[Dict], batch_size: int = 20) -> List[Dict]:
    """Consolidate items in smaller batches to avoid token limits"""
    if len(items) <= batch_size:
        # If we have fewer items than batch size, just consolidate directly
        result = consolidate_checklist({"items": items})
        return result.get("items", [])
    
    consolidated_items = []
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        print(f"Consolidating batch {i//batch_size + 1} ({len(batch)} items)...")
        
        try:
            batch_result = consolidate_checklist({"items": batch})
            if "items" in batch_result and batch_result["items"]:
                consolidated_items.extend(batch_result["items"])
        except Exception as e:
            print(f"Error consolidating batch: {e}")
            # If consolidation fails, include the raw items
            consolidated_items.extend(batch)
    
    # Final consolidation pass if needed and if we have a reasonable number of items
    if len(consolidated_items) > batch_size:
        print(f"Performing final consolidation on {len(consolidated_items)} items...")
        # Make smaller batches for final consolidation to ensure success
        final_batch_size = batch_size // 2
        return batch_consolidate(consolidated_items, final_batch_size)
    
    return consolidated_items


def create_consolidated_checklist(
    extractions: List[Dict],
    output_dir: str
) -> MutualFundChecklist:
    """Create consolidated checklist from extractions"""
    # Convert to proper format for consolidation
    checklist = MutualFundChecklist()
    
    # First collect all items
    raw_items = []
    for extraction in extractions:
        raw_items_batch = extraction["extraction"]["items"]
        page_range = extraction["page_range"]
        
        # Create RawExtraction object
        raw_extraction = RawExtraction(page_range=page_range)
        
        # Add items to RawExtraction
        for item in raw_items_batch:
            checklist_item = ChecklistItem(
                checklist_title=item["checklist_title"],
                checklist_description=item["checklist_description"],
                rationale=item["rationale"],
                page_numbers=item.get("page_numbers", page_range)
            )
            raw_extraction.items.append(checklist_item)
            raw_items.append(item)
        
        # Add RawExtraction to checklist
        checklist.add_extraction(raw_extraction)
    
    print(f"Collected {len(checklist.items)} raw checklist items")
    
    # Save raw checklist items
    raw_checklist_path = os.path.join(output_dir, RAW_CHECKLIST_FILE)
    with open(raw_checklist_path, 'w') as f:
        json.dump(raw_items, f, indent=2)
    
    print(f"Saved raw checklist items to {raw_checklist_path}")
    
    # Now deduplicate and consolidate
    print("Deduplicating and consolidating checklist items...")
    
    # Use batch consolidation for large checklists
    consolidated_items = batch_consolidate(raw_items)
    
    # Create new consolidated checklist
    final_checklist = MutualFundChecklist()
    for item in consolidated_items:
        final_checklist.add_item(ChecklistItem(
            checklist_title=item["checklist_title"],
            checklist_description=item["checklist_description"],
            rationale=item["rationale"],
            page_numbers=item["page_numbers"]
        ))
    
    print(f"Final checklist contains {len(final_checklist.items)} items after deduplication")
    
    # Save consolidated checklist
    checklist_path = os.path.join(output_dir, FINAL_CHECKLIST_FILE)
    with open(checklist_path, 'w') as f:
        json.dump(final_checklist.to_list(), f, indent=2)
    
    print(f"Saved consolidated checklist to {checklist_path}")
    
    return final_checklist


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate a checklist of mutual fund disclosure requirements from regulatory documents"
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default=SEBI_PDF_PATH,
        help="Path to the regulatory PDF document"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--no-resume", 
        action="store_true",
        help="Do not resume from intermediate results"
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Skip the consolidation step and use raw items directly"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Processing {args.pdf} to extract disclosure requirements")
    extractions = process_regulatory_document(
        args.pdf, 
        args.output,
        resume_from_intermediate=not args.no_resume
    )
    
    if args.skip_consolidation:
        # Create checklist from raw items without consolidation
        checklist = MutualFundChecklist()
        for extraction in extractions:
            for item in extraction["extraction"]["items"]:
                checklist.add_item(ChecklistItem(
                    checklist_title=item["checklist_title"],
                    checklist_description=item["checklist_description"],
                    rationale=item["rationale"],
                    page_numbers=item.get("page_numbers", extraction["page_range"])
                ))
        
        # Save raw checklist without consolidation
        checklist_path = os.path.join(args.output, FINAL_CHECKLIST_FILE)
        with open(checklist_path, 'w') as f:
            json.dump(checklist.to_list(), f, indent=2)
        
        print(f"Saved checklist with {len(checklist.items)} items (no consolidation)")
    else:
        # Normal flow with consolidation
        checklist = create_consolidated_checklist(extractions, args.output)
    
    end_time = time.time()
    print(f"Processing completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Generated checklist with {len(checklist.items)} items")


if __name__ == "__main__":
    main()
