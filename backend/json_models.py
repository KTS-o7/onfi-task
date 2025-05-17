from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


# PART 1: Checklist Generation Models

class ExtractedChecklistItem(BaseModel):
    """Model for an individual checklist item extracted from regulatory document"""
    checklist_title: str = Field(..., description="Short title of the checklist item")
    checklist_description: str = Field(..., description="Detailed description of what needs to be disclosed")
    rationale: str = Field(..., description="Rationale from SEBI circular or AMFI email")


class ChecklistItem(BaseModel):
    """Complete checklist item with source information"""
    content: ExtractedChecklistItem = Field(..., description="Extracted checklist item from the LLM")
    page_numbers: str = Field(..., description="Page numbers in the regulatory document")
    source_document: str = Field(default="SEBI Master Circular", description="Source document (SEBI/AMFI)")
    category: Optional[str] = Field(None, description="Optional category for grouping requirements")
    

# PART 2: HyDE Search and Evaluation Models

class HyDEQuery(BaseModel):
    """Model for HyDE search query generation"""
    checklist_item_id: str = Field(..., description="Identifier for the checklist item being searched")
    generated_document: str = Field(..., description="Hypothetical document content that would satisfy the requirement")
    search_query: str = Field(..., description="The search query derived from the hypothetical document")
    iteration: int = Field(default=1, description="The iteration number in the search process")


class SearchResult(BaseModel):
    """Model for search results from document"""
    content: str = Field(..., description="The content found from the search")
    page_number: int = Field(..., description="Page number where the content was found")
    relevance_score: float = Field(..., description="Relevance score of the search result (0-1)")


class Citation(BaseModel):
    """Model for content citations from the document"""
    text: str = Field(..., description="Cited text from the document")
    page_number: int = Field(..., description="Page number where citation is found")


class ComplianceStatus(BaseModel):
    """Model for compliance evaluation status"""
    status: Literal["Compliant", "Partially Compliant", "Non-Compliant", "Unknown"] = Field(..., description="Compliance status")
    confidence: float = Field(..., description="Confidence level in the compliance determination (0-1)")


class EvaluationResult(BaseModel):
    """Model for the evaluation of a single checklist item"""
    checklist_item: ChecklistItem = Field(..., description="The checklist item being evaluated")
    compliance_status: ComplianceStatus = Field(..., description="The compliance status determination")
    findings_summary: str = Field(..., description="A brief summary of findings related to compliance")
    citations:list[Citation] = Field(default_factory=list, description="Citations from the document supporting findings")
    matched_pages:list[int] = Field(default_factory=list, description="Page numbers where relevant information was found")
    search_iterations: int = Field(default=1, description="Number of search iterations performed")


# PART 3: Comprehensive Evaluation Output Models

class GapAnalysisResult(BaseModel):
    """Model for gap analysis between findings and objective"""
    gap_description: str = Field(..., description="Description of the information gap between findings and objectives")
    suggested_query: str = Field(..., description="Refined search query to find missing information")
    is_sufficient: bool = Field(..., description="Whether current findings are sufficient or need additional searches")


class ComprehensiveSearchResult(BaseModel):
    """Model for comprehensive iterative search results"""
    checklist_item: ChecklistItem = Field(..., description="The checklist item being evaluated")
    all_findings:list[dict] = Field(default_factory=list, description="Results from all search iterations")
    final_summary: str = Field(..., description="Final consolidated findings summary")
    final_citations:list[Citation] = Field(default_factory=list, description="Final consolidated citations")
    compliance_status: str = Field(..., description="Final compliance status")
    matched_pages:list[int] = Field(default_factory=list, description="All matched pages across iterations")
    queries_used:list[str] = Field(default_factory=list, description="Search queries used across iterations")
    iterations_performed: int = Field(default=1, description="Number of iterations performed")


class FinalComplianceReport(BaseModel):
    """Model for final compliance report to be displayed in table format"""
    compliance_requirement: str = Field(..., description="Description of the compliance requirement")
    source: str = Field(..., description="Regulation document and page number where requirement is found")
    rationale: str = Field(..., description="Rationale for the compliance status from SEBI/AMFI circular")
    compliance_status: str = Field(..., description="Compliance status (Compliant, Non-Compliant, etc.)")
    findings_summary: str = Field(..., description="Brief summary of findings related to the requirement")
    findings_citations:list[int] = Field(default_factory=list, description="Page numbers where findings are cited")


class ComplianceReportResponse(BaseModel):
    """Model for the complete compliance report response"""
    report_items:list[FinalComplianceReport] = Field(..., description="List of all compliance report items")
    compliant_count: int = Field(..., description="Count of compliant items")
    non_compliant_count: int = Field(..., description="Count of non-compliant items")
    partial_compliant_count: int = Field(..., description="Count of partially compliant items")
    unknown_count: int = Field(..., description="Count of items with unknown compliance")


# PART 4: Document Processing Models

class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    doc_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    page_count: int = Field(..., description="Number of pages in the document")
    created_at: str = Field(..., description="Timestamp when document was processed")
    fingerprint: str = Field(..., description="Document fingerprint to detect changes")


class ProcessedPage(BaseModel):
    """Model for a processed page from document"""
    content: str = Field(..., description="Extracted text content from the page")
    page_number: int = Field(..., description="Page number")
    has_tables: bool = Field(default=False, description="Whether the page contains tables")
    has_charts: bool = Field(default=False, description="Whether the page contains charts or figures")
    

class ExtractedChecklistItemList(BaseModel):
    item_list: List[ExtractedChecklistItem]

    