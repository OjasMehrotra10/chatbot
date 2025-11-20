# frontend.py - UPDATED WITH BETTER API INTEGRATION

import streamlit as st
import requests
import uuid
import json

# =========================== API Configuration ===========================
API_BASE_URL = "http://localhost:8000/api"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/chat/", timeout=5)
        return response.status_code == 200
    except:
        return False

def api_get_chat_history(thread_id: str):
    """Get chat history and uploaded files via API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/", 
            params={"thread_id": thread_id, "include_history": "true"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            
            # Update message history if available
            if 'message_history' in data:
                st.session_state["message_history"] = data['message_history']
            
            return data.get('uploaded_files', [])
        return []
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

def api_retrieve_all_threads():
    """Retrieve all threads via API"""
    try:
        response = requests.get(f"{API_BASE_URL}/chat/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('available_threads', {})
        return {}
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {}

def api_handle_file_upload(file_content: bytes, filename: str, thread_id: str):
    """Handle file upload via API"""
    try:
        files = {'file': (filename, file_content)}
        data = {'thread_id': thread_id}
        response = requests.post(f"{API_BASE_URL}/chat/", files=files, data=data, timeout=30)
        return response.json()
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"API Error: {str(e)}"
        }

def api_remove_uploaded_file(filename: str, thread_id: str):
    """Remove uploaded file via API"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/chat/",
            json={'filename': filename, 'thread_id': thread_id},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"API Error: {str(e)}"
        }

def api_get_chatbot_response(message: str, thread_id: str):
    """Get chatbot response via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json={'message': message, 'thread_id': thread_id},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('reply', 'No response received')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["uploaded_files"] = []
    st.session_state["file_uploader_key"] += 1

def add_thread(thread_id, title="New Chat"):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = {}
    
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title

def update_chat_title(thread_id, title):
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title

def get_chat_title(thread_id: str, first_message: str = "") -> str:
    """Generate chat title from first message or use default"""
    if first_message:
        words = first_message.split()[:6]
        title = ' '.join(words)
        if len(title) > 30:
            title = title[:30] + '...'
        return title
    return f"Chat {thread_id[:8]}"

def switch_chat(thread_id):
    """Switch to a different chat and load its history"""
    st.session_state["thread_id"] = thread_id
    
    # Load uploaded files AND chat history from API
    uploaded_files = api_get_chat_history(thread_id)
    
    st.session_state["uploaded_files"] = uploaded_files
    st.session_state["file_uploader_key"] += 1
    
    # FORCE RELOAD
    st.rerun()

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Initialize with backend threads via API
    backend_threads = api_retrieve_all_threads()
    st.session_state["chat_threads"] = backend_threads
    
    # Add current thread if it doesn't exist
    current_thread = st.session_state["thread_id"]
    if current_thread not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][current_thread] = "New Chat"

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# Load current uploaded files and history via API
current_thread = st.session_state["thread_id"]
st.session_state["uploaded_files"] = api_get_chat_history(current_thread)

# ============================ Sidebar ============================
st.sidebar.title("🤖 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

# Display chat threads
if st.session_state["chat_threads"]:
    for thread_id, title in list(st.session_state["chat_threads"].items())[::-1]:
        # Create a unique key for each button
        button_key = f"thread_{thread_id}"
        if st.sidebar.button(title, key=button_key, use_container_width=True):
            switch_chat(thread_id)
else:
    st.sidebar.info("No conversations yet. Start a new chat!")

# ============================ Main UI ============================
st.title("🤖 LangGraph Chatbot")

# Show current chat info
current_thread = st.session_state["thread_id"]
current_title = st.session_state["chat_threads"].get(current_thread, "New Chat")
st.caption(f"Current Chat: {current_title}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a PDF or Excel file", 
    type=['pdf', 'xlsx', 'xls','csv'],
    key=f"file_uploader_{st.session_state['file_uploader_key']}"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("📤 Upload File"):
            with st.spinner("Uploading and processing file..."):
                result = api_handle_file_upload(
                    uploaded_file.read(),
                    uploaded_file.name,
                    st.session_state["thread_id"]
                )
                
                if result['success']:
                    st.session_state["uploaded_files"] = api_get_chat_history(st.session_state["thread_id"])
                    
                    # Update chat title if this is the first file
                    if len(st.session_state["uploaded_files"]) == 1:
                        new_title = f"Chat with {uploaded_file.name.split('.')[0]}"
                        update_chat_title(st.session_state["thread_id"], new_title)
                else:
                    st.error(result['message'])
    with col2:
        st.write(f"Selected: **{uploaded_file.name}**")

# Display Uploaded Files for CURRENT CHAT ONLY
current_files = st.session_state["uploaded_files"]
if current_files:
    st.subheader("📎 Uploaded Files in this Chat")
    
    for file_info in current_files:
        col1, col3 = st.columns([3, 1])
        with col1:
            st.write(f"**{file_info['filename']}** ({file_info['file_type'].upper()})")
        with col3:
            if st.button("🗑️ Remove", key=f"remove_{file_info['filename']}_{st.session_state['thread_id']}"):
                with st.spinner("Removing file..."):
                    result = api_remove_uploaded_file(file_info['filename'], st.session_state["thread_id"])
                    if result['success']:
                        st.success(result['message'])
                        st.session_state["uploaded_files"] = api_get_chat_history(st.session_state["thread_id"])
                        st.rerun()
                    else:
                        st.error(result['message'])

# Chat History Display
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Create thread in sidebar on first message if it doesn't exist
    current_thread = st.session_state["thread_id"]
    if current_thread not in st.session_state["chat_threads"]:
        new_title = get_chat_title(current_thread, user_input)
        add_thread(current_thread, new_title)
    
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get chatbot response via API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = api_get_chatbot_response(user_input, st.session_state["thread_id"])
        st.write(response)
    
    # Save assistant message to frontend history
    st.session_state["message_history"].append({"role": "assistant", "content": response})
    
    # Update chat title with first message if this is a new chat
    if len(st.session_state["message_history"]) == 2:  # First user + assistant message
        new_title = get_chat_title(current_thread, user_input)
        update_chat_title(current_thread, new_title)
    
    # Rerun to update sidebar with new thread
    st.rerun()