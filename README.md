# Mutual Fund Disclosure Compliance Checker

This project provides a tool to check mutual fund Scheme Information Documents (SIDs) for compliance with SEBI and AMFI disclosure requirements.

## Overview

The system has two main components:

1. **Checklist Generator**: Processes SEBI/AMFI regulatory documents to extract a checklist of required disclosures
2. **Compliance Checker**: Evaluates mutual fund documents against the generated checklist
3. **Domain-Specific Extractor**: Efficiently processes large PDFs by focusing on key mutual fund disclosure categories

## Features

- Extracts disclosure requirements from 828-page SEBI master circular
- Focuses on relevant chapters/sections for disclosure requirements
- Processes documents in chunks to work within LLM context limitations
- Supports multiple LLM options: OpenAI, Groq, and Google Gemini with rate limit handling
- Provides structured output through Pydantic models and native JSON response validation
- Web-based UI with Streamlit for document evaluation
- RAG-based evaluation using ChromaDB for document analysis
- Domain-specific extraction for efficient processing of large regulatory documents

## Installation

```bash
# Clone the repository
git clone https://github.com/KTS-o7/onfi-task.git
cd onfi-task

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your API keys. You can use OpenAI, Groq, or Gemini as your LLM provider:

```
# OpenAI Configuration (Recommended)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1/  # or your custom endpoint
OPENAI_MODEL=gpt-4o  # or other OpenAI models
OPENAI_EMBEDDING_MODEL=text-embedding-004

# Alternative LLM Providers (Optional)
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

# Model configurations (optional)
LLM_GROQ_LARGE=llama-3.3-70b-versatile
LLM_GEMINI_LARGE=gemini-2.0-flash
```

### Example .env for OneAPI Integration

If you're using OneAPI or a similar API gateway, your `.env` file might look like:

```
OPENAI_BASE_URL=https://openai.com/v1/
OPENAI_API_KEY=sk-1234567890
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small         
```

## Usage

### Step 1: Generate Checklist

#### Standard Checklist Generation
```bash
cd checklist_generator
python generate_checklist.py --pdf ../SEBI_master_circular.pdf --output ../circular_evals
```

#### Domain-Specific Extraction (More Efficient)
For large PDFs (1000+ pages), use domain-specific extraction to process the document more efficiently:

```bash
cd checklist_generator
python dom_spec.py --pdf ../SEBI_master_circular.pdf --output ../domain_evals
```

This approach:
- Focuses only on predefined disclosure categories
- Uses less API tokens by prioritizing or consolidating within categories
- Produces a more organized, categorized checklist

Options for dom_spec.py:
- `--pdf`: Path to the regulatory PDF document
- `--output`: Directory to save output files
- `--no-resume`: Do not resume from intermediate results
- `--process-entire-pdf`: Process the entire PDF instead of focusing on specific chapters
- `--method`: Processing method, either "select" (fewer API calls) or "consolidate" (more comprehensive)

#### Standard Checklist Options
- `--pdf`: Path to the regulatory PDF document (default: `../SEBI_master_circular.pdf`)
- `--output`: Directory to save output files (default: `../circular_evals`)
- `--no-resume`: Do not resume from intermediate results
- `--skip-consolidation`: Skip the consolidation step and use raw items directly
- `--process-entire-pdf`: Process the entire PDF instead of focusing on specific chapters

### Step 2: Run Compliance Checker UI

```bash
cd checklist_generator
streamlit run streamlit_app.py
```

This will start a web server, typically at http://localhost:8501, where you can:

1. Upload or provide a URL to a mutual fund SID PDF
2. Run compliance checks against the generated checklist
3. View detailed evaluation results
4. Download results as CSV

### Step 3: RAG-Based Document Evaluation (Command Line)

```bash
cd checklist_generator
python rag_eval_cli.py --doc ../your_document.pdf --checklist ../circular_evals/mutual_fund_disclosure_checklist.json --output ../evaluation_results
```

Options:
- `--doc`: Path to the document to evaluate
- `--checklist`: Path to the checklist JSON file
- `--output`: Directory to save evaluation results
- `--batch-size`: Number of items to process in a batch (default: 10)
- `--force-refresh`: Force refresh of document embeddings

## Architecture

The system is built with these key components:

1. **PDF Processing** (`pdf_processor.py`):
   - Extracts text from PDF documents
   - Splits large documents into manageable chunks
   - Focuses on relevant chapters for disclosure requirements

2. **LLM Integration** (`llm_client.py`):
   - Manages API calls to OpenAI, Groq, and Gemini
   - Implements rate limiting to stay within API limits
   - Uses Pydantic for structured data validation
   - Provides fallback mechanisms between providers

3. **Structured Data Models** (`models.py`):
   - Defines Pydantic models for all data structures
   - Ensures consistent data handling throughout the application

4. **Checklist Generation** (`generate_checklist.py`):
   - Orchestrates the extraction of disclosure requirements
   - Manages the deduplication and consolidation process

5. **Domain-Specific Extraction** (`dom_spec.py`):
   - Provides category-focused extraction and processing
   - More efficient for large documents

6. **RAG Evaluation** (`rag_evaluator.py`):
   - Uses ChromaDB for vector search and document evaluation
   - Provides semantic search for relevant document sections
   - Evaluates compliance using retrieval-augmented generation

7. **Web UI** (`streamlit_app.py`):
   - Provides user-friendly interface for document evaluation
   - Visualizes compliance results in tabular format

## Technical Details

### LLM Integration

The system supports multiple LLM providers:

1. **OpenAI Integration**:
   ```python
   completion = openai_client.chat.completions.create(
       model=OPENAI_MODEL,
       messages=messages,
       response_format={"type": "json_object"},
       temperature=0.2
   )
   json_data = json.loads(completion.choices[0].message.content)
   result = schema_model(**json_data)  # Pydantic validation
   ```

2. **Groq Integration**:
   ```python
   response = groq_client.chat.completions.create(
       model=GROQ_MODEL,
       messages=messages,
       response_format={"type": "json_object"}
   )
   json_data = json.loads(response.choices[0].message.content)
   result = schema_model(**json_data)  # Pydantic validation
   ```

3. **Gemini Integration** (via OpenAI-compatible client):
   ```python
   completion = gemini_client.beta.chat.completions.parse(
       model=GEMINI_MODEL,
       messages=messages,
       response_format=schema_model,
   )
   result = completion.choices[0].message.parsed
   ```

### RAG-based Evaluation

The system uses ChromaDB for vector search and document evaluation:

1. Document processing:
   - Split document into pages
   - Generate embeddings for each page
   - Store in ChromaDB with metadata

2. Evaluation process:
   - For each checklist item, construct a semantic query
   - Retrieve the most relevant pages using vector similarity
   - Evaluate compliance using the retrieved context
   - Generate structured evaluation results

### Domain-Specific Processing

The domain-specific extraction is optimized for large regulatory documents:

1. Pre-defined disclosure categories:
   - Portfolio Disclosure
   - NAV and Performance Disclosure
   - Risk Disclosure
   - Fee and Expense Disclosure
   - Scheme Information Disclosure
   - And more...

2. Processing methods:
   - "select": Choose top N most important items per category (fewer API calls)
   - "consolidate": More comprehensive consolidation within categories

### Processing Large Documents

The 828-page SEBI master circular is processed by:

1. Identifying key chapters related to disclosure requirements
2. Splitting each chapter into ~10-page chunks
3. Processing each chunk to extract disclosure requirements
4. Saving intermediate results to enable resuming if interrupted
5. Consolidating and deduplicating all extracted requirements

## Troubleshooting

### Common Issues

1. **RAG Evaluation Shows False Non-Compliance**:
   - Try increasing the `top_k` parameter in `search_relevant_pages` function
   - Modify the system prompt to be more flexible in compliance assessment
   - Check if document text extraction is complete and accurate

2. **API Rate Limits**:
   - The system includes built-in rate limiting
   - For faster processing, increase your API rate limits or use multiple providers

3. **Memory Issues with Large PDFs**:
   - Use domain-specific extraction with the "select" method
   - Process the PDF in focused chapters rather than the entire document

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
