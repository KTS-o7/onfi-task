#!/usr/bin/env python3
"""
Streamlit UI for Mutual Fund Disclosure Compliance Checker
"""

import os
import json
import uuid
import requests
import pandas as pd
import streamlit as st
import pdfplumber
from io import BytesIO
from typing import Dict, List, Optional
from dotenv import load_dotenv
import hashlib

# Import local modules
from models import ChecklistItem, MutualFundChecklist, ChecklistEvaluation, DocumentEvaluation
from pdf_processor import extract_text_from_pdf
from llm_client import evaluate_document_against_checklist
from rag_evaluator import create_document_embedding, batch_evaluate_with_rag

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_CHECKLIST_PATH = "Domspec/mutual_fund_disclosure_checklist.json"
OUTPUT_DIR = "Domspec"


def ensure_output_dir(dir_path: str):
    """Ensure output directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_checklist(path: str) -> MutualFundChecklist:
    """Load checklist from file"""
    try:
        with open(path, 'r') as f:
            items_data = json.load(f)
        
        # Create MutualFundChecklist object
        checklist = MutualFundChecklist()
        for item_data in items_data:
            item = ChecklistItem(
                checklist_title=item_data["checklist_title"],
                checklist_description=item_data["checklist_description"],
                rationale=item_data["rationale"],
                page_numbers=item_data["page_numbers"],
                category=item_data.get("category")
            )
            checklist.add_item(item)
        
        return checklist
    except Exception as e:
        st.error(f"Error loading checklist: {e}")
        return MutualFundChecklist()


def download_pdf_from_url(url: str) -> BytesIO:
    """Download PDF from URL"""
    try:
        response = requests.get(url, verify=False)  # Added verify=False to handle certificate issues
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            st.error(f"Error downloading PDF: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf_binary(pdf_binary: BytesIO) -> str:
    """Extract text from PDF binary data"""
    try:
        text = ""
        with pdfplumber.open(pdf_binary) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def evaluate_document_traditional(document_text: str, checklist: MutualFundChecklist) -> DocumentEvaluation:
    """Evaluate document against checklist using traditional method (one item at a time)"""
    evaluation = DocumentEvaluation()
    
    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(checklist.items):
        # Update progress
        progress = (i + 1) / len(checklist.items)
        progress_bar.progress(progress)
        status_text.text(f"Evaluating {i+1}/{len(checklist.items)}: {item.checklist_title}")
        
        # Evaluate document against checklist item
        result = evaluate_document_against_checklist(document_text, item.model_dump())
        
        # Create ChecklistEvaluation object
        eval_item = ChecklistEvaluation(
            checklist_title=item.checklist_title,
            checklist_description=item.checklist_description,
            rationale=item.rationale,
            page_numbers=item.page_numbers,
            findings_summary=result["findings_summary"],
            citations=result["citations"]
        )
        
        # Add to evaluation
        evaluation.evaluations.append(eval_item)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return evaluation


def evaluate_document_with_rag(document_text: str, checklist: MutualFundChecklist, batch_size: int = 10) -> DocumentEvaluation:
    """Evaluate document against checklist using RAG approach (with batching)"""
    evaluation = DocumentEvaluation()
    
    # Generate a document ID based on content hash rather than random UUID
    doc_id = hashlib.md5(document_text[:5000].encode('utf-8')).hexdigest()[:32]
    doc_title = f"Mutual Fund Document {doc_id}"
    
    # Show embedding progress
    with st.spinner("Creating document embeddings... This may take a minute."):
        embedding_result = create_document_embedding(doc_id, doc_title, document_text)
        if embedding_result["status"] == "reused":
            st.success("Reusing existing document embeddings - skipping embedding creation.")
    
    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create batches
    items_data = [item.model_dump() for item in checklist.items]
    num_batches = (len(items_data) + batch_size - 1) // batch_size
    
    # Process in batches
    all_results = []
    for i in range(0, len(items_data), batch_size):
        # Update progress bar
        batch_num = i // batch_size + 1
        progress_bar.progress(batch_num / num_batches)
        status_text.text(f"Processing batch {batch_num}/{num_batches} ({min(batch_size, len(items_data) - i)} items)")
        
        # Process batch with actual batch size (not forcing to 1)
        batch = items_data[i:i+batch_size]
        batch_results = batch_evaluate_with_rag(doc_id, batch, batch_size=batch_size)
        all_results.extend(batch_results)
    
    # Create evaluation objects
    for result in all_results:
        item_data = result.checklist_item
        eval_item = ChecklistEvaluation(
            checklist_title=item_data["checklist_title"],
            checklist_description=item_data["checklist_description"],
            rationale=item_data["rationale"],
            page_numbers=item_data["page_numbers"],
            findings_summary=result.findings_summary,
            citations=result.citations,
            compliance_status=result.compliance_status,
            matched_pages=result.matched_pages
        )
        evaluation.evaluations.append(eval_item)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return evaluation


def create_evaluation_dataframe(evaluation: DocumentEvaluation) -> pd.DataFrame:
    """Create pandas DataFrame from evaluation results"""
    # Extract data for DataFrame
    data = []
    for eval_item in evaluation.evaluations:
        data.append({
            "Checklist Title": eval_item.checklist_title,
            "Checklist Description": eval_item.checklist_description,
            "Rationale": eval_item.rationale,
            "Page Numbers": eval_item.page_numbers,
            "Findings Summary": eval_item.findings_summary,
            "Citations": eval_item.citations,
            "Compliance Status": eval_item.compliance_status,
            "Matched Pages": ", ".join(map(str, eval_item.matched_pages)) if eval_item.matched_pages else "N/A"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def save_evaluation(evaluation: DocumentEvaluation, pdf_name: str, output_dir: str):
    """Save evaluation results"""
    ensure_output_dir(output_dir)
    
    # Create filename from PDF name
    if pdf_name.endswith('.pdf'):
        pdf_name = pdf_name[:-4]
    filename = f"{pdf_name}_evaluation.json"
    
    # Save as JSON
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(evaluation.to_list(), f, indent=2)
    
    return filepath


def display_compliance_summary(df: pd.DataFrame):
    """Display compliance summary statistics"""
    if "Compliance Status" not in df.columns:
        return
    
    status_counts = df["Compliance Status"].value_counts()
    
    # Create columns for summary
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    compliant = status_counts.get("Compliant", 0)
    partially = status_counts.get("Partially Compliant", 0)
    non_compliant = status_counts.get("Non-Compliant", 0)
    unknown = status_counts.get("Unknown", 0)
    
    col1.metric("Total Requirements", total)
    col2.metric("Compliant", compliant, f"{compliant/total*100:.1f}%" if total > 0 else None)
    col3.metric("Partially Compliant", partially, f"{partially/total*100:.1f}%" if total > 0 else None)
    col4.metric("Non-Compliant", non_compliant, f"{non_compliant/total*100:.1f}%" if total > 0 else None)
    
    # Create a simple bar chart
    st.bar_chart(status_counts)


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Mutual Fund Disclosure Compliance Checker",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("Mutual Fund Disclosure Compliance Checker")
    st.markdown(
        """
        Upload or provide a URL to a mutual fund Scheme Information Document (SID) PDF, 
        and this tool will evaluate it against SEBI and AMFI disclosure requirements.
        """
    )
    
    # Sidebar
    st.sidebar.title("Options")
    checklist_path = st.sidebar.text_input(
        "Checklist path",
        value=DEFAULT_CHECKLIST_PATH
    )
    
    # Load checklist
    checklist = load_checklist(checklist_path)
    st.sidebar.info(f"Loaded {len(checklist.items)} checklist items")
    
    # Evaluation options
    evaluation_method = st.sidebar.radio(
        "Evaluation Method",
        ["RAG-based (Faster)", "Traditional (Slower)"]
    )
    
    batch_size = 10
    if evaluation_method == "RAG-based (Faster)":
        batch_size = st.sidebar.slider("Batch Size", 1, 30, 10)
    
    # Filter by category if available
    categories = []
    for item in checklist.items:
        if item.category and item.category not in categories:
            categories.append(item.category)
    
    selected_category = None
    if categories:
        st.sidebar.subheader("Filter by Category")
        show_all = st.sidebar.checkbox("Show All Categories", value=True)
        if not show_all:
            selected_category = st.sidebar.selectbox("Category", categories)
    
    # Main content
    # Input options
    input_method = st.radio(
        "Select document input method",
        options=["URL", "Upload PDF"],
        horizontal=True
    )
    
    document_text = None
    pdf_name = None
    evaluation_complete = False
    
    if input_method == "URL":
        pdf_url = st.text_input("Enter PDF URL")
        if pdf_url:
            analyze_button = st.button("Download and Analyze")
            if analyze_button:
                with st.spinner("Downloading PDF..."):
                    pdf_binary = download_pdf_from_url(pdf_url)
                    if pdf_binary:
                        document_text = extract_text_from_pdf_binary(pdf_binary)
                        pdf_name = pdf_url.split("/")[-1]
                        st.info(f"Document text extracted: {len(document_text)} characters")
                        
                        # Evaluate document
                        if document_text:
                            with st.spinner("Evaluating document against disclosure requirements..."):
                                # Filter checklist if category selected
                                if selected_category:
                                    filtered_checklist = MutualFundChecklist()
                                    for item in checklist.items:
                                        if item.category == selected_category:
                                            filtered_checklist.add_item(item)
                                    st.info(f"Filtering to {len(filtered_checklist.items)} items in category: {selected_category}")
                                    active_checklist = filtered_checklist
                                else:
                                    active_checklist = checklist
                                
                                # Choose evaluation method
                                if evaluation_method == "RAG-based (Faster)":
                                    evaluation = evaluate_document_with_rag(document_text, active_checklist, batch_size)
                                else:
                                    evaluation = evaluate_document_traditional(document_text, active_checklist)
                                
                                evaluation_complete = True
                                
                                # Display compliance summary
                                df = create_evaluation_dataframe(evaluation)
                                st.subheader("Compliance Summary")
                                display_compliance_summary(df)
                                
                                # Display results
                                st.subheader("Detailed Evaluation Results")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download results as CSV
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "Download results as CSV",
                                    csv,
                                    f"{pdf_name}_evaluation.csv",
                                    "text/csv",
                                    key="download-csv"
                                )
                                
                                # Save results
                                save_button = st.button("Save evaluation results")
                                if save_button:
                                    filepath = save_evaluation(evaluation, pdf_name, OUTPUT_DIR)
                                    st.success(f"Evaluation saved to {filepath}")
    
    else:  # Upload PDF
        uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
        if uploaded_file:
            analyze_button = st.button("Process and Analyze")
            if analyze_button:
                with st.spinner("Processing uploaded document..."):
                    document_text = extract_text_from_pdf_binary(BytesIO(uploaded_file.read()))
                    pdf_name = uploaded_file.name
                    st.info(f"Document text extracted: {len(document_text)} characters")
                    
                    # Evaluate document
                    if document_text:
                        with st.spinner("Evaluating document against disclosure requirements..."):
                            # Filter checklist if category selected
                            if selected_category:
                                filtered_checklist = MutualFundChecklist()
                                for item in checklist.items:
                                    if item.category == selected_category:
                                        filtered_checklist.add_item(item)
                                st.info(f"Filtering to {len(filtered_checklist.items)} items in category: {selected_category}")
                                active_checklist = filtered_checklist
                            else:
                                active_checklist = checklist
                            
                            # Choose evaluation method
                            if evaluation_method == "RAG-based (Faster)":
                                evaluation = evaluate_document_with_rag(document_text, active_checklist, batch_size)
                            else:
                                evaluation = evaluate_document_traditional(document_text, active_checklist)
                            
                            evaluation_complete = True
                            
                            # Display compliance summary
                            df = create_evaluation_dataframe(evaluation)
                            st.subheader("Compliance Summary")
                            display_compliance_summary(df)
                            
                            # Display results
                            st.subheader("Detailed Evaluation Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download results as CSV
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download results as CSV",
                                csv,
                                f"{pdf_name}_evaluation.csv",
                                "text/csv",
                                key="download-csv"
                            )
                            
                            # Save results
                            save_button = st.button("Save evaluation results")
                            if save_button:
                                filepath = save_evaluation(evaluation, pdf_name, OUTPUT_DIR)
                                st.success(f"Evaluation saved to {filepath}")
    
    if not document_text and not evaluation_complete:
        st.info("Please provide a PDF document to analyze")


if __name__ == "__main__":
    main() 