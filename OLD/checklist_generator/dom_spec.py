#!/usr/bin/env python3
"""
Domain-Specific Checklist Generator for Mutual Fund Disclosure Requirements
Processes SEBI and AMFI regulatory documents to extract disclosure requirements by category
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any
import time
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, Field, ValidationError

# Import local modules
from pdf_processor import process_pdf_in_chunks, load_intermediate_results
from llm_client import (
    call_openai_with_structure,
    ChecklistItemsExtract,
    reset_rate_limits_if_needed
)
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

# LLM model choice - Only OpenAI is available now
DEFAULT_LLM = "openai"

def get_mutual_fund_disclosure_categories():
    """Return the core disclosure categories for mutual funds based on SEBI regulations."""
    return {
        "Portfolio Disclosure": {
            "description": "Requirements for disclosing portfolio holdings, composition, and related metrics",
            "keywords": ["portfolio", "holdings", "composition", "asset allocation", "securities"]
        },
        "NAV and Performance Disclosure": {
            "description": "Requirements for disclosing NAV calculation, performance metrics, and benchmarks",
            "keywords": ["nav", "performance", "returns", "benchmark", "cagr", "index"]
        },
        "Risk Disclosure": {
            "description": "Requirements for disclosing various risks and the risk management framework",
            "keywords": ["risk", "risk-o-meter", "credit risk", "liquidity risk", "market risk"]
        },
        "Fee and Expense Disclosure": {
            "description": "Requirements for disclosing fees, expenses, TER, and associated charges",
            "keywords": ["fee", "expense", "ter", "total expense ratio", "load", "charge"]
        },
        "Scheme Information Disclosure": {
            "description": "Requirements for disclosing scheme characteristics, objectives, and structure",
            "keywords": ["scheme", "objective", "structure", "policy", "strategy"]
        },
        "Investor Communication": {
            "description": "Requirements for communication with investors regarding various fund matters",
            "keywords": ["investor", "communication", "report", "statement", "email", "notification"]
        },
        "Valuation Disclosure": {
            "description": "Requirements for disclosing valuation methodology and related information",
            "keywords": ["valuation", "fair value", "pricing", "valuation policy"]
        },
        "Dividend and Distribution": {
            "description": "Requirements for disclosing dividend declaration and distribution information",
            "keywords": ["dividend", "distribution", "payout", "income distribution"]
        },
        "Redemption and Exit": {
            "description": "Requirements for disclosing redemption procedures, exit loads, and related information",
            "keywords": ["redemption", "exit", "repurchase", "liquidity"]
        },
        "Regulatory Compliance": {
            "description": "Requirements for disclosing compliance with regulatory norms and guidelines",
            "keywords": ["compliance", "regulatory", "sebi", "amfi", "circular"]
        }
    }

def ensure_output_dir(dir_path: str):
    """Ensure output directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class DomainChecklistItemExtract(BaseModel):
    """Schema for extracting a single domain-specific checklist item"""
    checklist_title: str = Field(..., description="Short title of the checklist item")
    checklist_description: str = Field(..., description="Detailed description of what needs to be disclosed")
    rationale: str = Field(..., description="Rationale from SEBI circular or AMFI email")
    page_numbers: str = Field(..., description="Page numbers in the regulatory document")
    category: str = Field(..., description="Disclosure category this item belongs to")

class DomainChecklistItemsExtract(BaseModel):
    """Schema for extracting a list of domain-specific checklist items"""
    items: List[DomainChecklistItemExtract] = Field(default_factory=list, description="List of checklist items")

def extract_domain_specific_requirements(text: str, page_range: str) -> Dict:
    """
    Extract domain-specific disclosure requirements from text using OpenAI.
    Focus only on predefined important categories.
    """
    reset_rate_limits_if_needed()
    
    # Get domain categories
    categories = get_mutual_fund_disclosure_categories()
    category_descriptions = "\n".join([f"- {cat}: {details['description']}" for cat, details in categories.items()])
    
    system_prompt = f"""
    You are an expert in mutual fund regulations and compliance. Your task is to extract ONLY the most important disclosure requirements from SEBI and AMFI regulatory circulars.
    
    Focus ONLY on the following key disclosure categories:
    {category_descriptions}
    
    Extract ONLY specific, mandatory disclosure requirements that mutual funds MUST follow in their documents.
    Ignore procedural instructions or requirements that don't directly impact what gets disclosed to investors.
    
    Prioritize requirements that:
    1. Have a direct impact on investor decision-making
    2. Are explicitly mandated by SEBI/AMFI (not just recommendations)
    3. Apply to most mutual fund types (not just specialized schemes)
    4. Include specific, clear details about what must be disclosed
    
    Structure your response as a valid JSON object with these fields:
    - items: A list of extracted requirements, each with:
      - checklist_title: A short title for the requirement
      - checklist_description: Detailed description of what needs to be disclosed
      - rationale: The rationale from the regulatory document
      - page_numbers: The page numbers in the regulatory document where this requirement is found
      - category: The disclosure category this item belongs to (from the list above)
    
    Format your response ONLY as a valid JSON object.
    """
    
    prompt = f"""
    Extract ONLY the most important disclosure requirements that mutual funds must follow from the following regulatory text.
    Focus on concrete, mandatory requirements for what must be disclosed in fund documents.
    
    TEXT FROM PAGES {page_range}:
    {text}
    """
    
    try:
        # Use OpenAI for extraction with domain-specific model
        result = call_openai_with_structure(prompt, system_prompt, DomainChecklistItemsExtract)
        
        if result and hasattr(result, 'items'):
            return result.model_dump()
        
        # If failed, return empty checklist
        return {"items": []}
    except Exception as e:
        print(f"Error extracting domain-specific requirements: {e}")
        return {"items": []}

def process_regulatory_document(
    pdf_path: str, 
    output_dir: str,
    resume_from_intermediate: bool = True,
    focus_on_chapters: bool = True
) -> List[Dict]:
    """Process regulatory document to extract disclosure requirements"""
    ensure_output_dir(output_dir)
    
    intermediate_file_path = os.path.join(output_dir, INTERMEDIATE_FILE)
    
    # Check if we can resume from intermediate results
    if resume_from_intermediate and os.path.exists(intermediate_file_path):
        print(f"Resuming from intermediate results in {intermediate_file_path}")
        return load_intermediate_results(intermediate_file_path)
    
    # Process PDF in chunks using the domain-specific extraction function
    return process_pdf_in_chunks(pdf_path, extract_domain_specific_requirements, focus_on_chapters=focus_on_chapters)

def consolidate_by_category(raw_items: List[Dict]) -> List[Dict]:
    """Consolidate checklist items by category to reduce API calls."""
    # Group items by category
    categorized_items = {}
    
    # Handle items that may not have a category (from earlier extractions)
    for item in raw_items:
        category = item.get("category", "Uncategorized")
        if category not in categorized_items:
            categorized_items[category] = []
        categorized_items[category].append(item)
    
    print(f"Grouped items into {len(categorized_items)} categories")
    
    # Consolidate within each category
    consolidated_items = []
    for category, items in categorized_items.items():
        print(f"Consolidating {len(items)} items in category: {category}")
        
        # If we have few items, just use them directly
        if len(items) <= 3:
            consolidated_items.extend(items)
            continue
            
        # For larger categories, consolidate with limited batches
        batch_size = min(20, len(items))
        
        # Only consolidate if we have enough items to warrant it
        if len(items) > 10:
            # Use the existing consolidation function but with a specific prompt
            system_prompt = f"""
            You are an expert in mutual fund regulations and compliance. Your task is to consolidate disclosure requirements 
            in the '{category}' category into a unified checklist.
            
            Combine similar requirements, remove duplicates, and keep only the most important, mandatory disclosures.
            Limit your output to the 5-7 most important items in this category.
            
            Structure your response as a valid JSON object with these fields:
            - items: A list of consolidated requirements, each with:
              - checklist_title: A short title for the requirement
              - checklist_description: Detailed description of what needs to be disclosed
              - rationale: The rationale from the regulatory document
              - page_numbers: The page numbers in the regulatory document where this requirement is found
              - category: The disclosure category (should be '{category}')
            
            Format your response ONLY as a valid JSON object.
            """
            
            extractions_text = json.dumps({"items": items}, indent=2)
            prompt = f"""
            Consolidate the following extracted disclosure requirements in the '{category}' category into a unified checklist.
            Keep only the 5-7 most important, mandatory disclosure requirements.
            
            EXTRACTED REQUIREMENTS:
            {extractions_text}
            """
            
            try:
                result = call_openai_with_structure(prompt, system_prompt, DomainChecklistItemsExtract)
                if result and hasattr(result, 'items'):
                    # Add category if it was dropped
                    for item in result.items:
                        if not hasattr(item, 'category'):
                            item.category = category
                    consolidated_items.extend(result.items)
                else:
                    # If consolidation fails, add a subset of the original items
                    consolidated_items.extend(items[:5])
            except Exception as e:
                print(f"Error consolidating {category} category: {e}")
                # If exception occurs, include a subset of items
                consolidated_items.extend(items[:5])
        else:
            # For smaller groups, just take the items directly
            consolidated_items.extend(items)
    
    print(f"Consolidated to {len(consolidated_items)} items across {len(categorized_items)} categories")
    return consolidated_items

def select_top_n_per_category(raw_items: List[Dict], n: int = 3) -> List[Dict]:
    """
    Select the top N most important items per category.
    This is a simpler alternative to consolidation that uses fewer API calls.
    """
    # Group items by category
    categorized_items = {}
    
    for item in raw_items:
        category = item.get("category", "Uncategorized")
        if category not in categorized_items:
            categorized_items[category] = []
        categorized_items[category].append(item)
    
    print(f"Grouped items into {len(categorized_items)} categories")
    
    # Select top N for each category
    selected_items = []
    for category, items in categorized_items.items():
        print(f"Selecting top {n} items from {len(items)} in category: {category}")
        
        # If we have few items, just use them all
        if len(items) <= n:
            selected_items.extend(items)
            continue
        
        # Use OpenAI to select top N most important items
        system_prompt = f"""
        You are an expert in mutual fund regulations and compliance. Your task is to select the {n} MOST IMPORTANT 
        disclosure requirements from the '{category}' category.
        
        Prioritize requirements that:
        1. Have the greatest investor protection impact
        2. Are legally mandatory (not just recommended)
        3. Contain specific, actionable disclosure guidance
        4. Apply to most mutual fund types (not specialized/niche requirements)
        
        Structure your response as a valid JSON object with these fields:
        - items: A list of EXACTLY {n} selected requirements, maintaining their original structure
        
        Format your response ONLY as a valid JSON object.
        """
        
        items_text = json.dumps({"items": items}, indent=2)
        prompt = f"""
        Review these mutual fund disclosure requirements in the '{category}' category and select the {n} MOST IMPORTANT ones:
        
        {items_text}
        
        IMPORTANT: Return EXACTLY {n} items that are most critical for investor protection and regulatory compliance.
        """
        
        try:
            result = call_openai_with_structure(prompt, system_prompt, DomainChecklistItemsExtract)
            if result and hasattr(result, 'items') and len(result.items) > 0:
                # Add selected items to our result list
                selected_items.extend(result.items)
            else:
                # Fallback if OpenAI selection fails: manually select top N items
                selected_items.extend(items[:n])
                print(f"Warning: Selection failed for {category}, using first {n} items")
        except Exception as e:
            print(f"Error selecting from {category}: {e}")
            # Fallback if exception occurs
            selected_items.extend(items[:n])
    
    print(f"Selected {len(selected_items)} items from {len(categorized_items)} categories")
    return selected_items

def create_domain_consolidated_checklist(
    extractions: List[Dict],
    output_dir: str,
    method: str = "select"
) -> MutualFundChecklist:
    """
    Create domain-specific consolidated checklist from extractions
    
    Parameters:
    - extractions: List of extraction results
    - output_dir: Directory to save output files
    - method: Consolidation method, either "consolidate" (API-heavy) or "select" (fewer API calls)
    """
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
                page_numbers=item.get("page_numbers", page_range),
                # Include category if available
                category=item.get("category", "Uncategorized")
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
    
    # Process items based on selected method
    if method == "consolidate":
        print("Consolidating checklist items by category using OpenAI...")
        processed_items = consolidate_by_category(raw_items)
    else:  # "select" method
        print("Selecting top items per category using OpenAI...")
        processed_items = select_top_n_per_category(raw_items, n=3)
    
    # Create new consolidated checklist
    final_checklist = MutualFundChecklist()
    for item in processed_items:
        # Fix: Access attributes directly instead of using dictionary subscription
        # Check if item is a Pydantic model or a dictionary and handle accordingly
        if hasattr(item, 'model_dump'):  # Pydantic model
            checklist_item = ChecklistItem(
                checklist_title=item.checklist_title,
                checklist_description=item.checklist_description,
                rationale=item.rationale,
                page_numbers=item.page_numbers,
                category=getattr(item, 'category', "Uncategorized")
            )
        else:  # Dictionary
            checklist_item = ChecklistItem(
                checklist_title=item["checklist_title"],
                checklist_description=item["checklist_description"],
                rationale=item["rationale"],
                page_numbers=item["page_numbers"],
                category=item.get("category", "Uncategorized")
            )
            
        final_checklist.add_item(checklist_item)
    
    print(f"Final checklist contains {len(final_checklist.items)} items after processing")
    
    # Save consolidated checklist
    checklist_path = os.path.join(output_dir, FINAL_CHECKLIST_FILE)
    with open(checklist_path, 'w') as f:
        json.dump(final_checklist.to_list(), f, indent=2)
    
    print(f"Saved final checklist to {checklist_path}")
    
    return final_checklist

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate a domain-specific checklist of mutual fund disclosure requirements"
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
        "--process-entire-pdf",
        action="store_true",
        help="Process the entire PDF instead of focusing on specific chapters"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["consolidate", "select"],
        default="select",
        help="Method to process items: 'consolidate' (more API calls) or 'select' (fewer API calls)"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Processing {args.pdf} to extract domain-specific disclosure requirements using OpenAI model")
    extractions = process_regulatory_document(
        pdf_path=args.pdf,
        output_dir=args.output,
        resume_from_intermediate=not args.no_resume,
        focus_on_chapters=not args.process_entire_pdf
    )
    
    checklist = create_domain_consolidated_checklist(
        extractions=extractions, 
        output_dir=args.output,
        method=args.method
    )
    
    end_time = time.time()
    print(f"Processing completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Generated checklist with {len(checklist.items)} items")
    
    # Print summary by category
    categories = {}
    for item in checklist.items:
        category = getattr(item, "category", "Uncategorized")
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    print("\nItems by category:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} items")

if __name__ == "__main__":
    main()