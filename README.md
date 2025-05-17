# Mutual Fund Disclosure Compliance Checker

A full-stack application for checking mutual fund Scheme Information Documents (SIDs) for compliance with SEBI and AMFI disclosure requirements.

## Tech Stack

### Frontend
- **Framework**: Next.js 15 with TypeScript
- **UI Components**: 
  - Radix UI for accessible components
  - Shadcn for styled components
  - TailwindCSS for styling
  - Framer Motion for animations
- **PDF Processing**: PDF.js and React-PDF
- **Development**: Turbopack for fast development

### Backend
- **Framework**: Flask (Python)
- **LLM Integration**: 
  - OpenAI GPT-4
  - Groq
  - Google Gemini
- **Vector Database**: ChromaDB for RAG
- **PDF Processing**: Custom OCR and text extraction

## Features

### Frontend Features
- Modern, responsive UI with dark/light mode support
- Real-time PDF preview and analysis
- Interactive compliance dashboard
- Progress tracking and visualization
- Export functionality for reports

### Backend Features
- Extracts disclosure requirements from 828-page SEBI master circular
- Domain-specific extraction for efficient processing
- Multi-LLM support with automatic fallback
- RAG-based evaluation using ChromaDB
- Structured output through Pydantic models
- Rate limit handling and error recovery

## Project Structure

```
onfi-task/
├── frontend/                 # Next.js frontend application
│   ├── app/                 # Next.js app directory
│   ├── components/          # React components
│   ├── services/           # API services
│   ├── lib/                # Utility functions
│   └── types/              # TypeScript type definitions
│
└── backend/                # Flask backend application
    ├── flask_server.py     # Main Flask application
    ├── llm_service.py      # LLM integration service
    ├── pdf_ocr.py          # PDF processing utilities
    └── json_models.py      # Pydantic models
```

## Installation

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python flask_server.py
```

## Configuration

### Frontend Configuration
Create a `.env.local` file in the frontend directory:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Backend Configuration
Create a `.env` file in the backend directory:
```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1/
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Alternative LLM Providers
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

1. Start both frontend and backend servers
2. Access the application at `http://localhost:3000`
3. Upload a mutual fund SID PDF
4. View compliance analysis in real-time
5. Export results as needed



## API Documentation

### Frontend API Services
- `services/api.ts`: Handles all API communication
- `services/pdf.ts`: Manages PDF processing and preview
- `services/analysis.ts`: Handles compliance analysis

### Backend API Endpoints
- `POST /api/analyze`: Upload and analyze PDF
- `GET /api/status`: Check analysis status
- `GET /api/results`: Get analysis results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## This project is a part of Onfinance's task for the interview process.
