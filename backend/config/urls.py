from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Chat API
    path("api/chat/", include("apps.chat.urls")),
    path("api/documents/", include("apps.documents.urls")),
]