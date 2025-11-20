# views.py - UPDATED WITH CHAT HISTORY SUPPORT

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import uuid

# Import your backend functions
from backend import (
    get_chatbot_response, 
    handle_file_upload, 
    get_uploaded_files, 
    remove_uploaded_file, 
    retrieve_all_threads,
    get_chat_history
)

def get_chat_history_via_api(thread_id: str):
    """Get chat history for API"""
    try:
        backend_messages = get_chat_history(thread_id)
        message_history = []
        
        if backend_messages:
            for msg in backend_messages:
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    role = "assistant"  # default to assistant
                    
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            role = "user"
                        elif msg.type == 'ai':
                            role = "assistant"
                    elif hasattr(msg, '__class__'):
                        class_name = msg.__class__.__name__
                        if class_name == 'HumanMessage':
                            role = "user"
                        elif class_name in ['AIMessage', 'ChatMessage']:
                            role = "assistant"
                    
                    message_history.append({"role": role, "content": content})
        
        return message_history
    except Exception:
        return []
import base64

def serialize_uploaded_files(files_list):
    """Convert uploaded files to JSON-serializable format with base64 encoding"""
    serialized_files = []
    for file_info in files_list:
        serialized_file = {
            'filename': file_info.get('filename', ''),
            'file_type': file_info.get('file_type', ''),
            'summary': file_info.get('summary', ''),
            'content_base64': base64.b64encode(file_info.get('content', b'')).decode('utf-8')
        }
        serialized_files.append(serialized_file)
    return serialized_files

@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):
    def get(self, request):
        """Get available threads or chat history"""
        thread_id = request.GET.get('thread_id')
        include_history = request.GET.get('include_history', 'false').lower() == 'true'
        
        if thread_id:
            # Get uploaded files for specific thread
            uploaded_files = get_uploaded_files(thread_id)
            
            # Serialize files to remove binary content
            serialized_files = serialize_uploaded_files(uploaded_files)
            
            # Get chat history to find the last message
            message_history = get_chat_history_via_api(thread_id)
            
            # Find the last AI message (reply)
            last_reply = None
            if message_history:
                # Reverse to find the last assistant message
                for msg in reversed(message_history):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        last_reply = msg.get("content")
                        break
            
            response_data = {
                'uploaded_files': serialized_files,
                'thread_id': thread_id,
                'last_reply': last_reply if last_reply else "No messages yet"
            }
            
            # Include full chat history if requested
            if include_history:
                response_data['message_history'] = message_history
            
            return JsonResponse(response_data)
        else:
            # Get all available threads
            available_threads = retrieve_all_threads()
            return JsonResponse({
                'available_threads': available_threads
            })

    def post(self, request):
        """Handle chat messages and file uploads"""
        # Check if this is a file upload
        if request.FILES:
            return self.handle_file_upload(request)
        else:
            return self.handle_chat_message(request)

    def handle_chat_message(self, request):
        """Process chat messages"""
        try:
            # Try to parse JSON data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                message = data.get('message', '')
                thread_id = data.get('thread_id', 'default')
            else:
                # Fallback to form data
                message = request.POST.get('message', '')
                thread_id = request.POST.get('thread_id', 'default')
            
            if not message:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            # Get chatbot response
            reply = get_chatbot_response(message, thread_id)
            
            return JsonResponse({
                'reply': reply,
                'thread_id': thread_id,
                'status': 'success'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    def handle_file_upload(self, request):
        """Handle file uploads"""
        try:
            uploaded_file = request.FILES.get('file')
            thread_id = request.POST.get('thread_id', 'default')
            
            if not uploaded_file:
                return JsonResponse({'error': 'No file provided'}, status=400)
            
            # Process the file
            result = handle_file_upload(
                uploaded_file.read(),
                uploaded_file.name,
                thread_id
            )
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e),
                'message': f'Error processing file: {str(e)}'
            }, status=500)

    def delete(self, request):
        """Handle file removal"""
        try:
            data = json.loads(request.body)
            filename = data.get('filename')
            thread_id = data.get('thread_id', 'default')
            
            if not filename:
                return JsonResponse({'error': 'No filename provided'}, status=400)
            
            result = remove_uploaded_file(filename, thread_id)
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)