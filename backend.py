# backend.py - FIXED IMPORTS
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Optional
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# FIX: Use the correct import for Document
from langchain_core.documents import Document
from dotenv import load_dotenv
import sqlite3
import pandas as pd
from io import BytesIO
import re
import chromadb
import pdfplumber
import PyPDF2
load_dotenv()
import json
from pathlib import Path

# Create local storage directories on startup
Path("./local_pdf_storage").mkdir(exist_ok=True)
Path("./local_excel_storage").mkdir(exist_ok=True)
# Rest of your code remains the same...

# -------------------
# 1. LLM & Embeddings
# -------------------
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. Enhanced File Storage with Vector DB
# -------------------
UPLOADED_FILES: Dict[str, List[Dict]] = {}
VECTOR_STORES: Dict[str, Any] = {}  # Store vector stores per thread
EXCEL_DATA: Dict[str, Dict] = {}    # Store Excel data per thread

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_vector_store(thread_id: str):
    """Get or create vector store for a thread"""
    collection_name = f"thread_{thread_id}"
    
    if thread_id not in VECTOR_STORES:
        try:
            # Try to get existing collection
            collection = chroma_client.get_collection(collection_name)
            vector_store = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
        except:
            # Create new collection
            vector_store = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
        VECTOR_STORES[thread_id] = vector_store
    
    return VECTOR_STORES[thread_id]

def store_file(file_content: bytes, filename: str, thread_id: str):
    """Store uploaded file in memory with thread isolation"""
    if thread_id not in UPLOADED_FILES:
        UPLOADED_FILES[thread_id] = []
    
    # Check if file already exists in this thread
    existing_files = [f for f in UPLOADED_FILES[thread_id] if f['filename'] == filename]
    if existing_files:
        return {'success': False, 'message': f"File '{filename}' already exists in this chat."}
    
    file_info = {
        'filename': filename,
        'content': file_content,
        'file_type': filename.split('.')[-1].lower()
    }
    UPLOADED_FILES[thread_id].append(file_info)
    return {'success': True, 'file_info': file_info}

def get_uploaded_files(thread_id: str):
    """Get all uploaded files for a thread"""
    return UPLOADED_FILES.get(thread_id, [])

def remove_file(filename: str, thread_id: str):
    """Remove file from thread"""
    if thread_id in UPLOADED_FILES:
        UPLOADED_FILES[thread_id] = [f for f in UPLOADED_FILES[thread_id] if f['filename'] != filename]
        
        # Also remove from vector store if it's a PDF
        file_type = filename.split('.')[-1].lower()
        if file_type == 'pdf':
            vector_store = get_vector_store(thread_id)
            # Remove documents related to this file
            try:
                # This is a simplified approach - in production you'd want more sophisticated deletion
                pass
            except:
                pass
        
        # Remove from Excel data if it's an Excel file
        if file_type in ['xlsx', 'xls']:
            if thread_id in EXCEL_DATA and filename in EXCEL_DATA[thread_id]:
                del EXCEL_DATA[thread_id][filename]
        
        return True
    return False

def remove_uploaded_file(filename: str, thread_id: str):
    """Remove uploaded file from thread - EXPORTED FUNCTION"""
    try:
        success = remove_file(filename, thread_id)
        return {
            'success': success,
            'message': f"File '{filename}' removed successfully!" if success else f"File '{filename}' not found!"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Error removing file: {str(e)}"
        }

def get_chat_title(thread_id: str, first_message: str = "") -> str:
    """Generate chat title from first message or use default"""
    if first_message:
        words = first_message.split()[:6]
        title = ' '.join(words)
        if len(title) > 30:
            title = title[:30] + '...'
        return title
    return f"Chat {thread_id[:8]}"

# -------------------
# 4. Enhanced File Processing Agents with Storage
# -------------------
import json
from pathlib import Path

class PDFAgent:
    """Agent for processing PDF files with LOCAL vector storage"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.local_storage_path = Path("./local_pdf_storage")
        self.local_storage_path.mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content using multiple methods - FIXED VERSION"""
        all_text = ""
        
        # Method 1: Try pdfplumber first (better for modern PDFs)
        try:
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Try multiple extraction strategies
                    page_text = ""
                    
                    # Strategy 1: Simple extraction
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 10:
                            all_text += f"Page {i+1}:\n{page_text}\n\n"
                            continue
                    except:
                        pass
                    
                    # Strategy 2: Extraction with layout preservation
                    try:
                        page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                        if page_text and len(page_text.strip()) > 10:
                            all_text += f"Page {i+1}:\n{page_text}\n\n"
                            continue
                    except:
                        pass
                    
                    # Strategy 3: Use text flow
                    try:
                        page_text = page.extract_text(use_text_flow=True)
                        if page_text and len(page_text.strip()) > 10:
                            all_text += f"Page {i+1}:\n{page_text}\n\n"
                            continue
                    except:
                        pass
                    
                    # If all strategies fail, try raw extraction
                    if not page_text:
                        raw_text = page.extract_text() or ""
                        if raw_text.strip():
                            all_text += f"Page {i+1} (raw):\n{raw_text}\n\n"
            
            if all_text.strip():
                print(f"PDF extracted successfully via pdfplumber: {len(all_text)} characters")
                return all_text
        except Exception as e:
            print(f"pdfplumber failed: {str(e)}")
        
        # Method 2: Try PyPDF2 as fallback
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                try:
                    # Try empty password
                    if reader.decrypt(""):
                        print("PDF decrypted with empty password")
                    else:
                        return "PDF is encrypted and cannot be decrypted."
                except:
                    return "PDF is encrypted and cannot be decrypted."
            
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {i+1}:\n{page_text}\n\n"
                except Exception as e:
                    print(f"Error extracting page {i+1}: {str(e)}")
                    continue
            
            if text.strip():
                print(f"PDF extracted successfully via PyPDF2: {len(text)} characters")
                return text
        except Exception as e:
            print(f"PyPDF2 failed: {str(e)}")
        
        # Method 3: Try pymupdf if available (most robust)
        try:
            import fitz  # pymupdf
            pdf_file = BytesIO(pdf_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text += f"Page {i+1}:\n{page_text}\n\n"
            doc.close()
            
            if text.strip():
                print(f"PDF extracted successfully via pymupdf: {len(text)} characters")
                return text
        except ImportError:
            print("pymupdf not available, skipping...")
        except Exception as e:
            print(f"pymupdf failed: {str(e)}")
        
        return "No readable text could be extracted from this PDF. The PDF might be image-based, encrypted, or corrupted."

    def vectorize_pdf(self, pdf_content: bytes, filename: str, thread_id: str) -> str:
        """Extract text from PDF and store in LOCAL vector database - FIXED VERSION"""
        print(f"Starting PDF processing for: {filename}")
        pdf_text = self.extract_text_from_pdf(pdf_content)
        
        if "Error" in pdf_text or "encrypted" in pdf_text or "No readable text" in pdf_text:
            print(f"PDF extraction failed: {pdf_text}")
            return pdf_text
        
        # Store raw PDF locally
        pdf_storage_path = self.local_storage_path / thread_id
        pdf_storage_path.mkdir(exist_ok=True)
        
        # Save PDF file locally
        pdf_file_path = pdf_storage_path / filename
        with open(pdf_file_path, 'wb') as f:
            f.write(pdf_content)
        
        # Save extracted text locally
        text_file_path = pdf_storage_path / f"{filename}_text.txt"
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(pdf_text)
        
        # Save structured info
        structured_info = self.extract_structured_info(pdf_text)
        info_file_path = pdf_storage_path / f"{filename}_info.json"
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(structured_info, f, indent=2)
        
        # Split text into chunks and vectorize
        texts = self.text_splitter.split_text(pdf_text)
        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "chunk": i,
                    "thread_id": thread_id,
                    "file_type": "pdf",
                    "local_path": str(pdf_file_path)
                }
            )
            documents.append(doc)
        
        # Get vector store and add documents
        vector_store = get_vector_store(thread_id)
        vector_store.add_documents(documents)
        
        print(f"PDF processed successfully: {len(documents)} chunks created")
        return f"PDF processed and stored locally. Created {len(documents)} chunks. Saved at: {pdf_file_path}"
    
    def get_local_pdfs(self, thread_id: str) -> List[Dict]:
        """Get all locally stored PDFs for a thread"""
        pdf_storage_path = self.local_storage_path / thread_id
        if not pdf_storage_path.exists():
            return []
        
        local_pdfs = []
        for file_path in pdf_storage_path.glob("*.pdf"):
            info_file = pdf_storage_path / f"{file_path.stem}_info.json"
            structured_info = {}
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    structured_info = json.load(f)
            
            local_pdfs.append({
                'filename': file_path.name,
                'path': str(file_path),
                'structured_info': structured_info,
                'file_type': 'pdf'
            })
        
        return local_pdfs

    def query_pdf(self, question: str, thread_id: str, filename: str = None) -> str:
        """Query PDF content using vector similarity search - FIXED VERSION"""
        try:
            vector_store = get_vector_store(thread_id)
            
            # Build filter - FIXED ChromaDB filter syntax
            filter_dict = {}
            if filename:
                filter_dict["source"] = filename
            else:
                filter_dict["file_type"] = "pdf"
            
            # Perform similarity search with proper filter
            try:
                docs = vector_store.similarity_search(question, k=3, filter=filter_dict)
            except Exception as filter_error:
                # Fallback: try without filter if filter fails
                print(f"Filter error, trying without filter: {filter_error}")
                docs = vector_store.similarity_search(question, k=3)
            
            if not docs:
                return "No relevant information found in the PDFs."
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            Based on the following context from PDF documents, answer the question.
            
            Context from PDFs:
            {context}
            
            Question: {question}
            
            Answer based only on the provided context. If the information is not in the context, say so.
            
            Answer:
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error querying PDF: {str(e)}"

    def extract_structured_info(self, pdf_text: str) -> dict:
        """Extract structured information from PDF text"""
        info = {}
        
        patterns = {
            'name': [r'Name[:\s]*([^\n\r]+)', r'Candidate[:\s]*([^\n\r]+)', r'Participant[:\s]*([^\n\r]+)'],
            'date': [r'Date[:\s]*([^\n\r]+)', r'Issued[:\s]*([^\n\r]+)', r'Completed[:\s]*([^\n\r]+)'],
            'course': [r'Course[:\s]*([^\n\r]+)', r'Certificate[:\s]*([^\n\r]+)', r'Program[:\s]*([^\n\r]+)'],
            'organization': [r'Organization[:\s]*([^\n\r]+)', r'Issued by[:\s]*([^\n\r]+)', r'Institute[:\s]*([^\n\r]+)'],
            'duration': [r'Duration[:\s]*([^\n\r]+)', r'Period[:\s]*([^\n\r]+)'],
            'score': [r'Score[:\s]*([^\n\r]+)', r'Grade[:\s]*([^\n\r]+)', r'Percentage[:\s]*([^\n\r]+)']
        }
        
        for field, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, pdf_text, re.IGNORECASE)
                if match:
                    info[field] = match.group(1).strip()
                    break
        
        return info

class ExcelAgent:
    """Agent for processing Excel files with LOCAL storage"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.local_storage_path = Path("./local_excel_storage")
        self.local_storage_path.mkdir(exist_ok=True)
    
    def store_excel_data(self, excel_content: bytes, filename: str, thread_id: str) -> Dict[str, Any]:
        """Store Excel file data LOCALLY - FIXED VERSION"""
        try:
            print(f"Starting Excel processing for: {filename}")
            
            excel_storage_path = self.local_storage_path / thread_id
            excel_storage_path.mkdir(exist_ok=True)
            
            # Save Excel file locally
            excel_file_path = excel_storage_path / filename
            with open(excel_file_path, 'wb') as f:
                f.write(excel_content)
            
            # Read and analyze Excel file
            excel_file = BytesIO(excel_content)
            
            # Try reading with different engines
            excel_data = None
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
            except:
                try:
                    excel_file.seek(0)  # Reset stream
                    excel_data = pd.read_excel(excel_file, sheet_name=None, engine='xlrd')
                except:
                    try:
                        excel_file.seek(0)  # Reset stream
                        excel_data = pd.read_excel(excel_file, sheet_name=None)
                    except Exception as e:
                        return {'error': f"Cannot read Excel file: {str(e)}"}
            
            if not excel_data:
                return {'error': 'No data found in Excel file'}
            
            # Store analysis for each sheet
            analysis = {}
            for sheet_name, df in excel_data.items():
                # Convert all data to string to avoid serialization issues
                sample_data = []
                for record in df.head(5).to_dict('records'):
                    safe_record = {}
                    for key, value in record.items():
                        try:
                            # Convert to string, handle NaN/None
                            if pd.isna(value):
                                safe_record[str(key)] = ""
                            else:
                                safe_record[str(key)] = str(value)
                        except:
                            safe_record[str(key)] = "[Unserializable Data]"
                    sample_data.append(safe_record)
                
                sheet_analysis = {
                    'shape': list(df.shape),  # Convert tuple to list for JSON
                    'columns': [str(col) for col in df.columns.tolist()],
                    'data_types': df.dtypes.astype(str).to_dict(),
                    'sample_data': sample_data,
                    'summary': {},
                    'null_counts': df.isnull().sum().to_dict()
                }
                
                # Add numerical summary if there are numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    try:
                        summary_dict = numeric_df.describe().to_dict()
                        # Convert numpy types to Python types
                        safe_summary = {}
                        for col, stats in summary_dict.items():
                            safe_summary[str(col)] = {str(k): float(v) for k, v in stats.items()}
                        sheet_analysis['summary'] = safe_summary
                    except:
                        sheet_analysis['summary'] = {}
                
                analysis[sheet_name] = sheet_analysis
            
            # Save analysis locally
            analysis_file_path = excel_storage_path / f"{filename}_analysis.json"
            with open(analysis_file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Store in global Excel data (for quick access)
            if thread_id not in EXCEL_DATA:
                EXCEL_DATA[thread_id] = {}
            EXCEL_DATA[thread_id][filename] = analysis
            
            print(f"Excel processed successfully: {len(analysis)} sheets analyzed")
            return {
                'success': True,
                'filename': filename,
                'local_path': str(excel_file_path),
                'sheets': list(excel_data.keys()),
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"Excel processing error: {str(e)}")
            return {'error': str(e)}
    
    def get_local_excels(self, thread_id: str) -> List[Dict]:
        """Get all locally stored Excel files for a thread"""
        excel_storage_path = self.local_storage_path / thread_id
        if not excel_storage_path.exists():
            return []
        
        local_excels = []
        for file_path in excel_storage_path.glob("*.xlsx"):
            analysis_file = excel_storage_path / f"{file_path.stem}_analysis.json"
            analysis = {}
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
            
            local_excels.append({
                'filename': file_path.name,
                'path': str(file_path),
                'analysis': analysis,
                'file_type': 'excel'
            })
        
        return local_excels
    
    def load_excel_from_local(self, filename: str, thread_id: str) -> pd.DataFrame:
        """Load Excel data from local storage"""
        excel_storage_path = self.local_storage_path / thread_id
        excel_file_path = excel_storage_path / filename
        
        if excel_file_path.exists():
            return pd.read_excel(excel_file_path)
        else:
            raise FileNotFoundError(f"Excel file {filename} not found in local storage")
        
    def query_excel_data(self, question: str, thread_id: str, filename: str = None) -> str:
        """Query stored Excel data"""
        try:
            
            # First check if we have data in EXCEL_DATA
            if thread_id not in EXCEL_DATA or not EXCEL_DATA[thread_id]:
                # Try to load from local storage
                local_excels = self.get_local_excels(thread_id)
                
                if not local_excels:
                    return "No Excel data available for querying. The file may not have been processed correctly."
                
                # Load analysis from local storage
                for excel_info in local_excels:
                    if filename is None or excel_info['filename'] == filename:
                        if thread_id not in EXCEL_DATA:
                            EXCEL_DATA[thread_id] = {}
                        EXCEL_DATA[thread_id][excel_info['filename']] = excel_info['analysis']
            
            # Now proceed with querying
            if thread_id not in EXCEL_DATA or not EXCEL_DATA[thread_id]:
                return "No Excel data available for querying after checking local storage."
            
            # Get relevant Excel files
            excel_files = EXCEL_DATA[thread_id]
            if filename and filename in excel_files:
                excel_files = {filename: excel_files[filename]}
            elif filename and filename not in excel_files:
                return f"Excel file '{filename}' not found in stored data. Available files: {list(EXCEL_DATA[thread_id].keys())}"
            
            if not excel_files:
                return "No matching Excel files found."
            
            # Prepare context from all relevant Excel files
            excel_context = ""
            for file_name, file_analysis in excel_files.items():
                excel_context += f"\n\n=== Excel File: {file_name} ===\n"
                for sheet_name, analysis in file_analysis.items():
                    excel_context += f"\nSheet: {sheet_name}\n"
                    excel_context += f"Shape: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns\n"
                    excel_context += f"Columns: {', '.join(analysis['columns'])}\n"
                    
                    # Show sample data in a readable format
                    if analysis['sample_data']:
                        excel_context += f"Sample Data (first {len(analysis['sample_data'])} rows):\n"
                        for i, row in enumerate(analysis['sample_data']):
                            excel_context += f"  Row {i+1}: {row}\n"
            
            prompt = f"""
            Based on the following Excel data analysis, answer the question.
            
            Excel Data Analysis:
            {excel_context}
            
            Question: {question}
            
            Provide a detailed answer based on the Excel data. If the answer cannot be found in the data, say so.
            Be specific about which file and sheet the information comes from.
            
            Answer:
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error querying Excel data: {str(e)}"

# Add this enhanced agent class to your backend.py
class IntelligentCrossFileAgent:
    """Agent that understands relationships between uploaded files and answers combined questions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)  # Use GPT-4 for better reasoning
        self.pdf_agent = PDFAgent()
        self.excel_agent = ExcelAgent()
        
    def analyze_file_relationships(self, thread_id: str) -> dict:
        """Analyze relationships between PDF and Excel files"""
        available_files = get_uploaded_files(thread_id)
        
        if not available_files:
            return {"relationships": [], "suggestions": []}
        
        # Extract structured information from all files
        file_analyses = []
        
        for file_info in available_files:
            if file_info['file_type'] == 'pdf':
                # Extract text and analyze
                pdf_text = self.pdf_agent.extract_text_from_pdf(file_info['content'])
                structured_info = self.pdf_agent.extract_structured_info(pdf_text)
                
                # Extract names from PDF
                names = self._extract_names_from_pdf(pdf_text)
                
                file_analyses.append({
                    'filename': file_info['filename'],
                    'type': 'pdf',
                    'structured_info': structured_info,
                    'entities': {'names': names},
                    'summary': self._summarize_pdf_content(pdf_text)
                })
                
            elif file_info['file_type'] in ['xlsx', 'xls']:
                # Get Excel analysis
                if thread_id in EXCEL_DATA and file_info['filename'] in EXCEL_DATA[thread_id]:
                    analysis = EXCEL_DATA[thread_id][file_info['filename']]
                    
                    # Extract entities from Excel
                    entities = self._extract_entities_from_excel(analysis)
                    
                    file_analyses.append({
                        'filename': file_info['filename'],
                        'type': 'excel',
                        'analysis': analysis,
                        'entities': entities,
                        'summary': self._summarize_excel_content(analysis)
                    })
        
        # Find relationships between files
        relationships = self._find_relationships(file_analyses)
        
        return {
            "file_analyses": file_analyses,
            "relationships": relationships,
            "combined_context": self._create_combined_context(file_analyses, relationships)
        }
    
    def _extract_names_from_pdf(self, pdf_text: str) -> list:
        """Extract all names from PDF text"""
        # Enhanced regex for names (can be customized)
        name_patterns = [
            r'(?:Employee|Name|Person|Participant)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\(|\n|$)',
            r'\b([A-Z][a-z]+)\s+[A-Z][a-z]+\b'  # First + Last name patterns
        ]
        
        names = set()
        for pattern in name_patterns:
            matches = re.findall(pattern, pdf_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match.split()) <= 3:  # Avoid capturing sentences
                    names.add(match.strip())
        
        return list(names)
    
    def _extract_entities_from_excel(self, excel_analysis: dict) -> dict:
        """Extract entities from Excel analysis"""
        entities = {
            'names': set(),
            'projects': set(),
            'departments': set(),
            'dates': set(),
            'other': set()
        }
        
        for sheet_name, analysis in excel_analysis.items():
            columns = analysis.get('columns', [])
            sample_data = analysis.get('sample_data', [])
            
            # Look for common column patterns
            for row in sample_data:
                for key, value in row.items():
                    key_lower = str(key).lower()
                    value_str = str(value)
                    
                    # Extract names
                    if any(name_key in key_lower for name_key in ['name', 'employee', 'manager', 'person']):
                        if len(value_str.split()) <= 3 and any(c.isalpha() for c in value_str):
                            entities['names'].add(value_str)
                    
                    # Extract projects
                    if any(proj_key in key_lower for proj_key in ['project', 'task', 'assignment']):
                        entities['projects'].add(value_str)
                    
                    # Extract dates
                    if any(date_key in key_lower for date_key in ['date', 'time', 'deadline']):
                        entities['dates'].add(value_str)
        
        # Convert sets to lists
        return {k: list(v) for k, v in entities.items()}
    
    def _find_relationships(self, file_analyses: list) -> list:
        """Find relationships between different files"""
        relationships = []
        
        pdf_analyses = [f for f in file_analyses if f['type'] == 'pdf']
        excel_analyses = [f for f in file_analyses if f['type'] == 'excel']
        
        for pdf in pdf_analyses:
            pdf_names = set(pdf['entities'].get('names', []))
            
            for excel in excel_analyses:
                excel_names = set(excel['entities'].get('names', []))
                
                # Find common names
                common_names = pdf_names.intersection(excel_names)
                
                if common_names:
                    relationships.append({
                        'pdf_file': pdf['filename'],
                        'excel_file': excel['filename'],
                        'relationship_type': 'shared_employees',
                        'common_entities': list(common_names),
                        'strength': len(common_names),
                        'description': f"{pdf['filename']} and {excel['filename']} share {len(common_names)} common employee(s): {', '.join(common_names)}"
                    })
        
        return relationships
    
    def _create_combined_context(self, file_analyses: list, relationships: list) -> str:
        """Create a comprehensive context combining all files"""
        context_parts = []
        
        # Summary of each file
        for analysis in file_analyses:
            context_parts.append(f"=== {analysis['filename']} ({analysis['type'].upper()}) ===")
            context_parts.append(f"Summary: {analysis.get('summary', 'No summary available')}")
            if analysis['entities']:
                for entity_type, entities in analysis['entities'].items():
                    if entities:
                        context_parts.append(f"{entity_type.title()}: {', '.join(entities[:5])}{'...' if len(entities) > 5 else ''}")
            context_parts.append("")
        
        # Relationships between files
        if relationships:
            context_parts.append("=== RELATIONSHIPS BETWEEN FILES ===")
            for rel in relationships:
                context_parts.append(f"• {rel['description']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _summarize_pdf_content(self, pdf_text: str) -> str:
        """Create a concise summary of PDF content"""
        lines = pdf_text.split('\n')[:10]  # Get first 10 lines
        content_lines = [line for line in lines if line.strip() and len(line.strip()) > 10]
        
        if not content_lines:
            return "Contains tabular or structured data"
        
        # Look for key phrases
        summary = ""
        for line in content_lines[:3]:
            if any(keyword in line.lower() for keyword in ['table', 'employee', 'manager', 'project', 'department']):
                summary += line[:100] + "... "
        
        return summary.strip() or "Document with organizational data"
    
    def _summarize_excel_content(self, excel_analysis: dict) -> str:
        """Create a concise summary of Excel content"""
        sheets = list(excel_analysis.keys())
        if not sheets:
            return "Empty spreadsheet"
        
        sheet_info = []
        for sheet_name in sheets[:2]:  # Limit to first 2 sheets
            analysis = excel_analysis[sheet_name]
            rows, cols = analysis.get('shape', [0, 0])
            columns = analysis.get('columns', [])
            sheet_info.append(f"{sheet_name} ({rows}×{cols}): {', '.join(columns[:3])}")
        
        return f"Sheets: {'; '.join(sheet_info)}"
    
    def answer_combined_question(self, question: str, thread_id: str) -> str:
        """Answer questions that involve multiple files"""
        # First, analyze the files and their relationships
        analysis = self.analyze_file_relationships(thread_id)
        
        if not analysis.get('file_analyses'):
            return "No files uploaded for analysis."
        
        # Prepare enhanced prompt
        prompt = f"""You are an intelligent assistant analyzing multiple documents. You have access to:
        
{analysis['combined_context']}

User Question: {question}

INSTRUCTIONS:
1. Analyze ALL the available files together
2. Identify relationships between the files (shared names, projects, etc.)
3. Answer by combining information from ALL relevant files
4. If information is in multiple files, show correlations
5. If some information is missing from one file, note that
6. Be specific about which file each piece of information comes from

Format your answer as:
1. **Direct Answer**: [Concise answer to the question]
2. **Sources**: [Which files provided which information]
3. **Relationships**: [How the information connects across files]
4. **Details**: [Specific data points from each file]

ANSWER:
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Add metadata about which files were used
            files_used = [f['filename'] for f in analysis['file_analyses']]
            relationships = analysis['relationships']
            
            enhanced_response = f"""🔍 **Combined Analysis from {len(files_used)} files**

{response.content}

---
**Analysis Details:**
• Files analyzed: {', '.join(files_used)}
• Relationships found: {len(relationships)}
• Key connections: {', '.join([r['description'] for r in relationships[:2]]) if relationships else 'None detected'}
"""
            
            return enhanced_response
            
        except Exception as e:
            return f"Error analyzing combined files: {str(e)}"

class MultiSourceQueryAgent:
    """Agent that can query both PDF and Excel data from LOCAL storage"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.pdf_agent = PDFAgent()
        self.excel_agent = ExcelAgent()
    
    def cross_reference_query(self, question: str, thread_id: str) -> str:
        """Query both PDF and Excel data from LOCAL storage and provide integrated answers"""
        
        # Get available local files
        local_pdfs = self.pdf_agent.get_local_pdfs(thread_id)
        local_excels = self.excel_agent.get_local_excels(thread_id)
        
        if not local_pdfs and not local_excels:
            return "No PDF or Excel files found in local storage for this thread."
        
        # Query PDF data
        pdf_answers = []
        for pdf_info in local_pdfs:
            answer = self.pdf_agent.query_pdf(question, thread_id, pdf_info['filename'])
            pdf_answers.append(f"**From PDF '{pdf_info['filename']}':** {answer}")
        
        # Query Excel data
        excel_answers = []
        for excel_info in local_excels:
            answer = self.excel_agent.query_excel_data(question, thread_id, excel_info['filename'])
            excel_answers.append(f"**From Excel '{excel_info['filename']}':** {answer}")
        
        # Integrate answers
        all_answers = pdf_answers + excel_answers
        
        if len(all_answers) == 1:
            return all_answers[0]
        else:
            integration_prompt = f"""
            Question: {question}
            
            I have answers from different data sources:
            
            {'\n'.join(all_answers)}
            
            Please provide a comprehensive integrated answer that:
            1. Combines information from both sources if relevant
            2. Highlights any correlations or relationships between PDF and Excel data
            3. Notes any contradictions or differences between sources
            4. Provides a clear, unified answer to the original question
            5. Specifies which source each piece of information comes from
            
            If one source doesn't have relevant information, focus on the source that does.
            
            Integrated Answer:
            """
            
            response = self.llm.invoke(integration_prompt)
            return response.content
    
    def get_combined_analysis(self, thread_id: str) -> str:
        """Get comprehensive analysis of all local files"""
        local_pdfs = self.pdf_agent.get_local_pdfs(thread_id)
        local_excels = self.excel_agent.get_local_excels(thread_id)
        
        analysis_parts = []
        
        # Analyze PDFs
        if local_pdfs:
            analysis_parts.append("=== PDF Documents ===")
            for pdf_info in local_pdfs:
                analysis_parts.append(f"📄 {pdf_info['filename']}")
                if pdf_info['structured_info']:
                    for key, value in pdf_info['structured_info'].items():
                        analysis_parts.append(f"   - {key}: {value}")
        
        # Analyze Excel files
        if local_excels:
            analysis_parts.append("\n=== Excel Files ===")
            for excel_info in local_excels:
                analysis_parts.append(f"📊 {excel_info['filename']}")
                for sheet_name, analysis in excel_info['analysis'].items():
                    analysis_parts.append(f"   - Sheet: {sheet_name} ({analysis['shape'][0]} rows, {analysis['shape'][1]} columns)")
                    analysis_parts.append(f"     Columns: {', '.join(analysis['columns'][:5])}{'...' if len(analysis['columns']) > 5 else ''}")
        
        return "\n".join(analysis_parts) if analysis_parts else "No files in local storage."

    def compare_sources(self, thread_id: str) -> str:
        """Compare and analyze relationships between PDF and Excel data"""
        available_files = get_uploaded_files(thread_id)
        
        pdf_files = [f for f in available_files if f['file_type'] == 'pdf']
        excel_files = [f for f in available_files if f['file_type'] in ['xlsx', 'xls']]
        
        if not pdf_files or not excel_files:
            return "Need both PDF and Excel files to perform cross-source analysis."
        
        # Get summaries of both data types
        pdf_summaries = []
        for pdf_file in pdf_files:
            pdf_text = self.pdf_agent.extract_text_from_pdf(pdf_file['content'])
            structured_info = self.pdf_agent.extract_structured_info(pdf_text)
            pdf_summaries.append({
                'filename': pdf_file['filename'],
                'structured_info': structured_info
            })
        
        excel_summaries = []
        for excel_file in excel_files:
            if thread_id in EXCEL_DATA and excel_file['filename'] in EXCEL_DATA[thread_id]:
                analysis = EXCEL_DATA[thread_id][excel_file['filename']]
                excel_summaries.append({
                    'filename': excel_file['filename'],
                    'sheets': list(analysis.keys()),
                    'columns': [analysis[sheet]['columns'] for sheet in analysis]
                })
        
        comparison_prompt = f"""
        Analyze the relationship between PDF documents and Excel files in this conversation.
        
        PDF Documents:
        {pdf_summaries}
        
        Excel Files:
        {excel_summaries}
        
        Please analyze:
        1. Potential relationships between the PDF content and Excel data
        2. Common themes or subjects
        3. How the data might complement each other
        4. Suggestions for cross-referencing queries
        
        Analysis:
        """
        
        response = self.llm.invoke(comparison_prompt)
        return response.content

# -------------------
# 5. Enhanced Central Agent with Multi-Source Support
# -------------------
class CentralAgent:
    """Central agent that coordinates between specialized agents - FIXED VERSION"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.pdf_agent = PDFAgent()
        self.excel_agent = ExcelAgent()
        self.multi_source_agent = MultiSourceQueryAgent()
    
    def should_use_files(self, question: str, available_files: List[Dict]) -> bool:
        """Determine if the question should use file processing"""
        question_lower = question.lower()
        
        # Always use files if question asks about uploaded content
        if any(phrase in question_lower for phrase in [
            'upload', 'file', 'document', 'pdf', 'excel', 'spreadsheet',
            'what is in', 'tell me about', 'show me', 'explain',
            'whose', 'name', 'who', 'what', 'when', 'where', 'which',
            'analyze', 'summary', 'data', 'information'
        ]) and available_files:
            return True
        
        return False
    
    def determine_file_types_needed(self, question: str, available_files: List[Dict]) -> List[str]:
        """Determine which file types are needed to answer the question"""
        question_lower = question.lower()
        file_types_present = [f['file_type'] for f in available_files]
        
        # Check if question might require cross-referencing
        cross_reference_indicators = [
            'manager', 'project', 'employee', 'team', 'report',
            'compare', 'relationship', 'both', 'across', 'between',
            'connect', 'link', 'together', 'combined', 'integration'
        ]
        
        has_multiple_file_types = len(set(file_types_present)) > 1
        needs_cross_reference = any(indicator in question_lower for indicator in cross_reference_indicators)
        
        # If we have multiple file types AND question suggests cross-referencing, use multi-source
        if has_multiple_file_types and needs_cross_reference:
            return ['multi_source']
        
        # Otherwise, determine specific file types needed
        needed_types = []
        
        # Check for PDF keywords
        pdf_keywords = ['pdf', 'document', 'text', 'report', 'letter', 'certificate']
        if any(word in question_lower for word in pdf_keywords) and 'pdf' in file_types_present:
            needed_types.append('pdf')
        
        # Check for Excel keywords
        excel_keywords = ['excel', 'spreadsheet', 'table', 'data', 'sheet', 'row', 'column', 'calculate']
        if any(word in question_lower for word in excel_keywords):
            if any(ft in ['xlsx', 'xls'] for ft in file_types_present):
                needed_types.append('excel')
        
        # If no specific types detected but files exist, check question type
        if not needed_types and available_files:
            # Questions about people, projects, relationships often need multi-source
            if has_multiple_file_types and any(word in question_lower for word in ['who', 'what', 'which', 'whose']):
                return ['multi_source']
            
            # Default to all available file types
            for ft in set(file_types_present):
                if ft == 'pdf':
                    needed_types.append('pdf')
                elif ft in ['xlsx', 'xls']:
                    needed_types.append('excel')
        
        return needed_types
    
    def coordinate_agents(self, question: str, thread_id: str) -> str:
        """Coordinate between agents to answer complex questions - FIXED"""
        available_files = get_uploaded_files(thread_id)
        
        if not available_files:
            return "No files uploaded for this conversation. Please upload files first."
        
        needed_types = self.determine_file_types_needed(question, available_files)
        
        # Handle multi-source queries
        if 'multi_source' in needed_types or len(needed_types) > 1:
            return self.multi_source_agent.cross_reference_query(question, thread_id)
        
        # Handle specific file type queries
        if needed_types:
            agent_answers = []
            
            for file_type in needed_types:
                if file_type == 'pdf':
                    # Query all PDFs
                    local_pdfs = self.pdf_agent.get_local_pdfs(thread_id)
                    for pdf_info in local_pdfs:
                        answer = self.pdf_agent.query_pdf(question, thread_id, pdf_info['filename'])
                        agent_answers.append(f"**From PDF '{pdf_infoh8['filename']}':** {answer}")
                
                elif file_type == 'excel':
                    # Query all Excel files
                    local_excels = self.excel_agent.get_local_excels(thread_id)
                    for excel_info in local_excels:
                        answer = self.excel_agent.query_excel_data(question, thread_id, excel_info['filename'])
                        agent_answers.append(f"**From Excel '{excel_info['filename']}':** {answer}")
            
            if not agent_answers:
                return "No relevant information found in the uploaded files."
            
            if len(agent_answers) == 1:
                return agent_answers[0]
            else:
                # Synthesize multiple answers
                synthesis_prompt = f"""
                Question: {question}
                
                I have answers from different files:
                {'\n'.join(agent_answers)}
                
                Please provide a unified answer combining all relevant information.
                
                Unified Answer:
                """
                
                response = self.llm.invoke(synthesis_prompt)
                return response.content
        
        # If no specific file types needed, but files exist, check if question relates to files
        question_lower = question.lower()
        file_related_keywords = ['what', 'who', 'which', 'where', 'when', 'how many', 'how much']
        
        if any(keyword in question_lower for keyword in file_related_keywords):
            # Try multi-source query as default when files exist
            return self.multi_source_agent.cross_reference_query(question, thread_id)
        
        return "I'm not sure how to answer that with the uploaded files. Could you be more specific about what you want to know from the documents?"

# Initialize agents
pdf_agent = PDFAgent()
excel_agent = ExcelAgent()
multi_source_agent = MultiSourceQueryAgent()
central_agent = CentralAgent()

# -------------------
# 5. Enhanced State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 6. Enhanced Nodes
# -------------------
def chat_node(state: ChatState):
    """Enhanced LLM node that can handle file-based questions"""
    messages = state["messages"]
    
    if not messages:
        return {"messages": []}
    
    latest_message = messages[-1].content if messages else ""
    
    # Extract thread_id from the last message's metadata if available
    thread_id = "default"
    if messages and hasattr(messages[-1], 'additional_kwargs') and 'thread_id' in messages[-1].additional_kwargs:
        thread_id = messages[-1].additional_kwargs['thread_id']
    
    available_files = get_uploaded_files(thread_id)
    
    if available_files and central_agent.should_use_files(latest_message, available_files):
        response = central_agent.coordinate_agents(latest_message, thread_id)
        return {"messages": [AIMessage(content=response)]}
    else:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 7. SQLITE CHECKPOINTER
# -------------------
def get_checkpointer():
    """Get SQLite checkpointer using the OLD working approach"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # Use the same approach as in backend_old.py
        conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        
        return checkpointer
    except ImportError:
        return None
    except Exception:
        return None

checkpointer = get_checkpointer()

# -------------------
# 8. Chat History Functions - CLEAN WORKING VERSION
# -------------------
def get_chat_history(thread_id: str):
    """Get chat history for a specific thread - CLEAN WORKING VERSION"""
    if not checkpointer:
        return []
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        saved_state = checkpointer.get(config)
        
        if saved_state is None:
            return []
            
        # Recursive search for messages in the state structure
        def find_messages_recursive(obj):
            if isinstance(obj, list) and obj and hasattr(obj[0], 'content'):
                return obj
            elif isinstance(obj, dict):
                for value in obj.values():
                    result = find_messages_recursive(value)
                    if result:
                        return result
            elif hasattr(obj, '__dict__'):
                for value in obj.__dict__.values():
                    result = find_messages_recursive(value)
                    if result:
                        return result
            return None
        
        messages = find_messages_recursive(saved_state) or []
        return messages
        
    except Exception:
        return []

def retrieve_all_threads():
    """Retrieve all conversation threads with titles - CLEAN WORKING VERSION"""
    all_threads = {}
    
    if not checkpointer:
        return all_threads

    try:
        # Get all thread IDs from checkpointer
        thread_ids = set()
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config["configurable"]["thread_id"]
            thread_ids.add(thread_id)
        
        # Create titles for each thread
        for thread_id in thread_ids:
            try:
                history = get_chat_history(thread_id)
                title = get_chat_title(thread_id)
                
                # Try to get a better title from first message
                if history:
                    first_human_msg = None
                    for msg in history:
                        if hasattr(msg, 'type') and msg.type == 'human':
                            first_human_msg = msg.content
                            break
                        elif hasattr(msg, '__class__') and msg.__class__.__name__ == 'HumanMessage':
                            first_human_msg = msg.content
                            break
                        elif hasattr(msg, 'content'):
                            first_human_msg = msg.content
                            break
                    
                    if first_human_msg:
                        title = get_chat_title(thread_id, first_human_msg)
                
                all_threads[thread_id] = title
                
            except Exception:
                all_threads[thread_id] = get_chat_title(thread_id)
        
        # Also include threads that have files but might not have chat history yet
        for thread_id in UPLOADED_FILES.keys():
            if thread_id not in all_threads:
                all_threads[thread_id] = get_chat_title(thread_id)
            
        return all_threads
                
    except Exception:
        return {}

# -------------------
# 9. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# COMPILE WITH SQLITE CHECKPOINTER
if checkpointer:
    chatbot = graph.compile(checkpointer=checkpointer)
else:
    chatbot = graph.compile()

def handle_file_upload(file_content: bytes, filename: str, thread_id: str):
    """Handle file upload and processing with LOCAL storage - ENHANCED"""
    try:
        # Check for duplicates
        result = store_file(file_content, filename, thread_id)
        if not result['success']:
            return result
        
        file_info = result['file_info']
        file_type = file_info['file_type']
        
        # Process file based on type with LOCAL storage
        if file_type == 'pdf':
            # Vectorize and store PDF LOCALLY
            storage_result = pdf_agent.vectorize_pdf(file_content, filename, thread_id)
            
            if "successfully" in storage_result or "stored locally" in storage_result:
                word_count = len(pdf_agent.extract_text_from_pdf(file_content).split())
                structured_info = pdf_agent.extract_structured_info(pdf_agent.extract_text_from_pdf(file_content))
                info_summary = ""
                if structured_info:
                    info_summary = " Found: " + ", ".join([f"{k}: {v}" for k, v in structured_info.items()])
                
                summary = f"PDF processed and stored locally. Extracted {word_count} words.{info_summary}"
            else:
                summary = f"PDF processing issue: {storage_result}"
                return {
                    'success': False,
                    'filename': filename,
                    'file_type': file_type,
                    'summary': summary,
                    'message': f"Could not process PDF: {storage_result}"
                }
                
        elif file_type in ['xlsx', 'xls']:
            # Store Excel data LOCALLY
            storage_result = excel_agent.store_excel_data(file_content, filename, thread_id)
            
            if 'success' in storage_result and storage_result['success']:
                sheet_count = len(storage_result['sheets'])
                summary = f"Excel processed and stored locally. {sheet_count} sheets analyzed."

            else:
                summary = f"Excel processing error: {storage_result.get('error', 'Unknown error')}"
                return {
                    'success': False,
                    'filename': filename,
                    'file_type': file_type,
                    'summary': summary,
                    'message': f"Could not process Excel file: {storage_result.get('error', 'Unknown error')}"
                }
        else:
            summary = f"File type {file_type} uploaded successfully."
        
        return {
            'success': True,
            'filename': filename,
            'file_type': file_type,
            'summary': summary,
            'message': f"File '{filename}' uploaded and stored locally successfully!"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Error uploading file: {str(e)}"
        }

def get_chatbot_response(message: str, thread_id: Optional[str] = None):
    """Enhanced helper function to get chatbot response - CLEAN VERSION"""
    if not thread_id:
        thread_id = "default"
    
    # Add thread_id to message metadata for use in chat_node
    human_msg = HumanMessage(content=message, additional_kwargs={"thread_id": thread_id})
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Simple approach - let the graph handle everything
        input_state = {"messages": [human_msg]}
        
        # Invoke the chatbot
        result = chatbot.invoke(input_state, config=config)
        
        # Extract the AI response
        messages = []
        
        # Handle different result structures
        if hasattr(result, 'values') and hasattr(result.values, 'get'):
            messages = result.values.get('messages', [])
        elif isinstance(result, dict):
            messages = result.get('messages', [])
        elif hasattr(result, 'get'):
            messages = result.get('messages', [])
        
        # Find the last AI message
        for msg in reversed(messages):
            if (hasattr(msg, 'content') and msg.content and 
                ((hasattr(msg, 'type') and msg.type == 'ai') or 
                 (hasattr(msg, '__class__') and msg.__class__.__name__ in ['AIMessage', 'ChatMessage']))):
                return msg.content
        
        return "No response generated"
        
    except Exception as e:
        return f"Error: {str(e)}"