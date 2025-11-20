"""
URL configuration for chatbot project.
"""
from django.contrib import admin
from django.urls import path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView  # ADD THIS IMPORT
from . import views

# ADD THIS FUNCTION
def root_view(request):
    return JsonResponse({
        "message": "Chatbot API is running!",
        "endpoints": {
            "chat": "/api/chat/",
            "admin": "/admin/",
            "api_test": "/api-test/"  # ADD THIS
        },
        "usage": {
            "GET": "Get available threads",
            "POST": "Send message as form data or JSON"
        }
    })

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/chat/', views.ChatAPIView.as_view(), name='chat-api'),
    path('api-test/', TemplateView.as_view(template_name='api_test.html'), name='api-test'),  # ADD THIS
    path('', root_view, name='root'),
]