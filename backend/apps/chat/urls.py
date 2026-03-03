from django.urls import path
from .views import ChatView,RAGChatView

urlpatterns = [
    path("", ChatView.as_view(), name="chat"),
    path("rag/", RAGChatView.as_view(), name="rag-chat"),
]