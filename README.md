# Mutual Fund Disclosure Compliance Checker

This project provides a tool to check mutual fund Scheme Information Documents (SIDs) for compliance with SEBI and AMFI disclosure requirements.

## Overview

The system has two main components:

1. **Checklist Generator**: Processes SEBI/AMFI regulatory documents to extract a checklist of required disclosures
2. **Compliance Checker**: Evaluates mutual fund documents against the generated checklist

## Features

- Extracts disclosure requirements from 828-page SEBI master circular
- Focuses on relevant chapters/sections for disclosure requirements
- Processes documents in chunks to work within LLM context limitations
- Uses free LLM APIs (Groq and Google Gemini) with rate limit handling
- Provides structured output through Pydantic models and native JSON response validation
- Web-based UI with Streamlit for document evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mutual-fund-disclosure-checker.git
cd mutual-fund-disclosure-checker

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your API keys:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

# Model configurations (optional)
LLM_GROQ_LARGE=llama-3.3-70b-versatile
LLM_GEMINI_LARGE=gemini-2.0-flash
```

## Usage

### Step 1: Generate Checklist

```bash
cd checklist_generator
python generate_checklist.py --pdf ../SEBI_master_circular.pdf --output ../circular_evals
```

Options:

- `--pdf`: Path to the regulatory PDF document (default: `../SEBI_master_circular.pdf`)
- `--output`: Directory to save output files (default: `../circular_evals`)
- `--no-resume`: Do not resume from intermediate results

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

## Architecture

The system is built with these key components:

1. **PDF Processing** (`pdf_processor.py`):

   - Extracts text from PDF documents
   - Splits large documents into manageable chunks
   - Focuses on relevant chapters for disclosure requirements

2. **LLM Integration** (`llm_client.py`):

   - Manages API calls to Groq and Gemini
   - Implements rate limiting to stay within free tier limits
   - Uses Pydantic for structured data validation
   - Provides fallback mechanisms between providers

3. **Structured Data Models** (`models.py`):

   - Defines Pydantic models for all data structures
   - Ensures consistent data handling throughout the application

4. **Checklist Generation** (`generate_checklist.py`):

   - Orchestrates the extraction of disclosure requirements
   - Manages the deduplication and consolidation process

5. **Web UI** (`streamlit_app.py`):
   - Provides user-friendly interface for document evaluation
   - Visualizes compliance results in tabular format

## Technical Details

### LLM Integration

The system uses two different LLM providers with their JSON response formats and Pydantic for validation:

1. **Groq Integration**:

   ```python
   response = groq_client.chat.completions.create(
       model=GROQ_MODEL,
       messages=messages,
       response_format={"type": "json_object"}
   )
   json_data = json.loads(response.choices[0].message.content)
   result = schema_model(**json_data)  # Pydantic validation
   ```

2. **Gemini Integration** (via OpenAI-compatible client):
   ```python
   completion = gemini_client.beta.chat.completions.parse(
       model=GEMINI_MODEL,
       messages=messages,
       response_format=schema_model,
   )
   result = completion.choices[0].message.parsed
   ```

### LLM Rate Limits

The system works within these free tier rate limits:

- **Groq (llama-3.3-70b-versatile)**:

  - 30 RPM (Requests Per Minute)
  - 1,000 RPD (Requests Per Day)
  - 12,000 TPM (Tokens Per Minute)
  - 100,000 TPD (Tokens Per Day)

- **Gemini 2.0 Flash**:
  - 15 RPM
  - 1,500 RPD
  - 1,000,000 TPM

### Processing Large Documents

The 828-page SEBI master circular is processed by:

1. Identifying key chapters related to disclosure requirements
2. Splitting each chapter into ~10-page chunks
3. Processing each chunk to extract disclosure requirements
4. Saving intermediate results to enable resuming if interrupted
5. Consolidating and deduplicating all extracted requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
