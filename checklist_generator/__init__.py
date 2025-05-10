"""
Mutual Fund Disclosure Checklist Generator
-----------------------------------------

This package provides tools to generate and evaluate checklists
for mutual fund scheme information document disclosure requirements.
"""

# Version
__version__ = "0.1.0"

# Import main components
from .models import (
    ChecklistItem,
    RawExtraction,
    ChecklistEvaluation,
    MutualFundChecklist,
    DocumentEvaluation
)

# Import PDF processing utilities
from .pdf_processor import (
    extract_text_from_pdf,
    get_pdf_length,
    process_pdf_in_chunks
)

# Import LLM client functions
from .llm_client import (
    extract_requirements_from_text,
    consolidate_checklist,
    evaluate_document_against_checklist
) 