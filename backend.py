# backend.py - CLEAN WORKING VERSION WITH CHAT HISTORY

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import Optional
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import sqlite3
import os
import pandas as pd
import PyPDF2
import pdfplumber
from io import BytesIO
import json
import tempfile
import re

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-3.5-turbo")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. Enhanced File Storage with Thread Isolation
# -------------------
UPLOADED_FILES: Dict[str, List[Dict]] = {}

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
        # Use first 6 words of first message as title
        words = first_message.split()[:6]
        title = ' '.join(words)
        if len(title) > 30:
            title = title[:30] + '...'
        return title
    return f"Chat {thread_id[:8]}"

# -------------------
# 4. File Processing Agents
# -------------------
class PDFAgent:
    """Agent for processing PDF files"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content using multiple methods"""
        all_text = ""
        
        # Method 1: Try pdfplumber
        try:
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    strategies = [
                        lambda p: p.extract_text(),
                        lambda p: p.extract_text(x_tolerance=1, y_tolerance=1),
                        lambda p: p.extract_text(use_text_flow=True)
                    ]
                    
                    for strategy in strategies:
                        try:
                            page_text = strategy(page)
                            if page_text and len(page_text.strip()) > 10:
                                all_text += f"Page {i+1}:\n{page_text}\n\n"
                                break
                        except:
                            continue
                    
                    if not any(strategy(page) for strategy in strategies):
                        raw_text = page.extract_text()
                        if raw_text:
                            all_text += f"Page {i+1} (raw):\n{raw_text}\n\n"
            
            if all_text.strip():
                return all_text
        except Exception as e:
            pass
        
        # Method 2: Try PyPDF2
        try:
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            if reader.is_encrypted:
                try:
                    if reader.decrypt(""):
                        pass
                except:
                    return "PDF is encrypted and cannot be decrypted."
            
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {i+1}:\n{page_text}\n\n"
            
            if text.strip():
                return text
        except Exception as e:
            pass
        
        return "No readable text could be extracted from this PDF."

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
    
    def answer_question(self, question: str, pdf_content: bytes) -> str:
        """Answer question based on PDF content"""
        pdf_text = self.extract_text_from_pdf(pdf_content)
        
        if "Error" in pdf_text or "encrypted" in pdf_text or "No readable text" in pdf_text:
            return f"Unable to process PDF: {pdf_text}"
        
        structured_info = self.extract_structured_info(pdf_text)
        pdf_context = pdf_text[:8000]
        
        structured_context = ""
        if structured_info:
            structured_context = "Structured information found in the PDF:\n"
            for key, value in structured_info.items():
                structured_context += f"- {key.capitalize()}: {value}\n"
            structured_context += "\n"
        
        prompt = f"""
        You are analyzing a PDF certificate/document. Here is the extracted content:

        {structured_context}
        Full PDF Content (partial):
        {pdf_context}

        Question: {question}

        CRITICAL INSTRUCTIONS:
        1. FIRST check the structured information above - if the answer is there, use it directly
        2. THEN search through the full PDF content for the answer
        3. Be SPECIFIC and DIRECT in your answer
        4. If you find partial information, share what you found
        5. If the information is truly not in the PDF, say: "This information is not specified in the certificate"
        6. For dates, names, courses - be very precise
        7. If asking "what can you know", list ALL identifiable information from the PDF

        Answer format: Be direct and factual. No disclaimers unless information is truly missing.

        Answer:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def analyze_pdf_content(self, pdf_content: bytes) -> str:
        """Comprehensive analysis of PDF content"""
        pdf_text = self.extract_text_from_pdf(pdf_content)
        
        if "Error" in pdf_text or "encrypted" in pdf_text or "No readable text" in pdf_text:
            return pdf_text
        
        structured_info = self.extract_structured_info(pdf_text)
        
        prompt = f"""
        Analyze this PDF certificate/document and provide a comprehensive summary of ALL information available.

        PDF Content:
        {pdf_text[:6000]}

        Structured Information Found:
        {structured_info}

        Provide a detailed summary including:
        1. Type of document (certificate, report, etc.)
        2. Person/entity the document is about
        3. Dates mentioned
        4. Course/program/organization details
        5. Any scores, grades, or achievements
        6. Issuing authority
        7. Any other relevant information

        Be thorough and extract every piece of information available.

        Comprehensive Analysis:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing PDF: {str(e)}"

class ExcelAgent:
    """Agent for processing Excel files"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    def analyze_excel(self, excel_content: bytes) -> Dict[str, Any]:
        """Analyze Excel file and return summary"""
        try:
            excel_file = BytesIO(excel_content)
            df = pd.read_excel(excel_file)
            
            analysis = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'summary': df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else {}
            }
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def answer_question(self, question: str, excel_content: bytes) -> str:
        """Answer question based on Excel data"""
        analysis = self.analyze_excel(excel_content)
        
        if 'error' in analysis:
            return f"Error analyzing Excel file: {analysis['error']}"
        
        prompt = f"""
        Based on the following Excel file analysis, answer the question.
        
        Excel File Analysis:
        - Shape: {analysis['shape']}
        - Columns: {analysis['columns']}
        - Sample Data: {analysis['sample_data']}
        - Summary: {analysis['summary']}
        
        Question: {question}
        
        Provide a detailed answer based on the Excel data. If the answer cannot be found in the data, say so.
        
        Answer:
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class CentralAgent:
    """Central agent that coordinates between specialized agents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.pdf_agent = PDFAgent()
        self.excel_agent = ExcelAgent()
    
    def should_use_files(self, question: str, available_files: List[Dict]) -> bool:
        """Determine if the question should use file processing"""
        question_lower = question.lower()
        
        file_keywords = [
            'read', 'pdf', 'excel', 'document', 'file', 'upload', 'content',
            'what is in', 'tell me about', 'summary', 'analyze', 'data',
            'information', 'details', 'extract', 'show me', 'explain',
            'whose', 'name', 'who', 'what', 'when', 'where', 'which',
            'certificate', 'course', 'date', 'issued', 'completed',
            'organization', 'score', 'grade', 'duration', 'program',
            'what can you know', 'what information', 'what details',
            'list information', 'summary of', 'overview of'
        ]
        
        has_file_keywords = any(keyword in question_lower for keyword in file_keywords)
        has_uploaded_files = len(available_files) > 0
        
        return has_file_keywords and has_uploaded_files
    
    def determine_file_types_needed(self, question: str, available_files: List[Dict]) -> List[str]:
        """Determine which file types are needed to answer the question"""
        question_lower = question.lower()
        file_types_present = [f['file_type'] for f in available_files]
        
        needed_types = []
        
        if any(word in question_lower for word in ['pdf', 'document', 'report', 'text', 'certificate', 'name', 'date', 'course']):
            if 'pdf' in file_types_present:
                needed_types.append('pdf')
        
        if any(word in question_lower for word in ['excel', 'spreadsheet', 'data', 'table', 'numbers', 'calculate', 'analysis']):
            if any(ft in ['xlsx', 'xls'] for ft in file_types_present):
                needed_types.append('excel')
        
        if not needed_types and available_files:
            needed_types = list(set(file_types_present))
        
        return needed_types
    
    def coordinate_agents(self, question: str, thread_id: str) -> str:
        """Coordinate between agents to answer complex questions"""
        available_files = get_uploaded_files(thread_id)
        
        if not available_files:
            return "No files uploaded for this conversation. Please upload files first."
        
        question_lower = question.lower()
        
        # Handle "what can you know" type questions - prioritize PDFs
        if any(phrase in question_lower for phrase in ['what can you know', 'what information', 'what details', 'list information', 'summary of', 'overview of']):
            relevant_files = [f for f in available_files if f['file_type'] == 'pdf']
            if relevant_files:
                all_analyses = []
                for file_info in relevant_files:
                    analysis = self.pdf_agent.analyze_pdf_content(file_info['content'])
                    all_analyses.append(f"=== Analysis of '{file_info['filename']}' ===\n{analysis}")
                
                if len(all_analyses) == 1:
                    return all_analyses[0]
                else:
                    combined_analysis = "\n\n".join(all_analyses)
                    return f"Analysis of {len(all_analyses)} PDF files:\n\n{combined_analysis}"
            else:
                return "No PDF files available for comprehensive analysis."
        
        # For specific questions, process all relevant files
        needed_types = self.determine_file_types_needed(question, available_files)
        
        if not needed_types:
            return "I couldn't determine which files to use for your question. Please be more specific about which file you're referring to."
        
        # Process all relevant files
        agent_answers = []
        
        for file_type in needed_types:
            relevant_files = [f for f in available_files if f['file_type'] == file_type]
            
            for file_info in relevant_files:
                if file_type == 'pdf':
                    answer = self.pdf_agent.answer_question(question, file_info['content'])
                    agent_answers.append(f"**From PDF '{file_info['filename']}':** {answer}")
                
                elif file_type in ['xlsx', 'xls']:
                    answer = self.excel_agent.answer_question(question, file_info['content'])
                    agent_answers.append(f"**From Excel file '{file_info['filename']}':** {answer}")
        
        if not agent_answers:
            return "Could not process the uploaded files for this question."
        
        if len(agent_answers) == 1:
            return agent_answers[0]
        else:
            # Enhanced synthesis for multiple files
            synthesis_prompt = f"""
            Question: {question}
            
            I have answers from different file analyses:
            {chr(10).join(agent_answers)}

            Please provide a comprehensive synthesized answer combining all the above analyses. 
            - Highlight agreements between different files
            - Note any contradictions or differences
            - Provide a clear, unified answer
            - If information comes from multiple sources, mention that

            Comprehensive Answer:
            """
            
            response = self.llm.invoke(synthesis_prompt)
            return response.content

# Initialize agents
pdf_agent = PDFAgent()
excel_agent = ExcelAgent()
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
    """Handle file upload and processing with duplicate check"""
    try:
        # Check for duplicates
        result = store_file(file_content, filename, thread_id)
        if not result['success']:
            return result
        
        file_info = result['file_info']
        file_type = file_info['file_type']
        
        # Process file based on type
        if file_type == 'pdf':
            pdf_text = pdf_agent.extract_text_from_pdf(file_content)
            if "Error" not in pdf_text and "encrypted" not in pdf_text and "No readable text" not in pdf_text:
                word_count = len(pdf_text.split())
                
                structured_info = pdf_agent.extract_structured_info(pdf_text)
                info_summary = ""
                if structured_info:
                    info_summary = " Found: " + ", ".join([f"{k}: {v}" for k, v in structured_info.items()])
                
                summary = f"PDF processed successfully. Extracted {word_count} words.{info_summary}"
            else:
                summary = f"PDF processing issue: {pdf_text}"
                return {
                    'success': False,
                    'filename': filename,
                    'file_type': file_type,
                    'summary': summary,
                    'message': f"Could not process PDF: {pdf_text}"
                }
        elif file_type in ['xlsx', 'xls']:
            analysis = excel_agent.analyze_excel(file_content)
            if 'error' not in analysis:
                summary = f"Excel processed successfully. Dataset: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns."
            else:
                summary = f"Excel processing error: {analysis['error']}"
                return {
                    'success': False,
                    'filename': filename,
                    'file_type': file_type,
                    'summary': summary,
                    'message': f"Could not process Excel file: {analysis['error']}"
                }
        else:
            summary = f"File type {file_type} uploaded successfully."
        
        return {
            'success': True,
            'filename': filename,
            'file_type': file_type,
            'summary': summary,
            'message': f"File '{filename}' uploaded successfully!"
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