from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ChecklistItem(BaseModel):
    """A single item in the disclosure checklist."""
    checklist_title: str = Field(..., description="Short title of the checklist item")
    checklist_description: str = Field(..., description="Detailed description of what needs to be disclosed")
    rationale: str = Field(..., description="Rationale from SEBI circular or AMFI email")
    page_numbers: str = Field(..., description="Page numbers in the regulatory document")
    source_document: str = Field(default="SEBI Master Circular", description="Source document (SEBI/AMFI)")
    category: Optional[str] = None  # Optional category for grouping
    
    def __str__(self):
        return f"{self.checklist_title} | {self.checklist_description[:50]}..."


class RawExtraction(BaseModel):
    """Raw extraction from a chunk of the regulatory document."""
    page_range: str
    items: List[ChecklistItem] = Field(default_factory=list)


class ChecklistEvaluation(BaseModel):
    """Evaluation of a document against a checklist item."""
    checklist_title: str
    checklist_description: str
    rationale: str
    page_numbers: str
    findings_summary: str = Field(..., description="Brief assessment of compliance")
    citations: str = Field(..., description="Direct quotes from the document supporting findings")
    compliance_status: str = "Unknown"  # "Compliant", "Partially Compliant", "Non-Compliant", or "Unknown"
    matched_pages: List[int] = []  # List of page numbers where relevant information was found


class MutualFundChecklist(BaseModel):
    """Complete checklist for mutual fund disclosure requirements."""
    items: List[ChecklistItem] = Field(default_factory=list)
    extractions: List[RawExtraction] = []
    
    def add_item(self, item: ChecklistItem):
        self.items.append(item)
    
    def add_extraction(self, extraction: RawExtraction):
        self.extractions.append(extraction)
        self.items.extend(extraction.items)
    
    def to_list(self) -> List[Dict]:
        return [item.model_dump() for item in self.items]
    
    def __len__(self):
        return len(self.items)


class DocumentEvaluation(BaseModel):
    """Complete evaluation of a document against the checklist."""
    evaluations: List[ChecklistEvaluation] = Field(default_factory=list)
    
    def to_list(self) -> List[Dict]:
        return [eval.model_dump() for eval in self.evaluations] 