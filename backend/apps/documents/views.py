from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.rag.document_loader import DocumentLoader
from core.rag.vector_store import VectorStoreService


class UploadPDFView(APIView):

    def post(self, request):
        file = request.FILES.get("file")

        if not file:
            return Response(
                {"error": "PDF file required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_path = f"temp_{file.name}"

        with open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        documents = DocumentLoader.load_pdf(file_path)
        for i, doc in enumerate(documents):
            print(f"\n--- Page {i+1} ---")
            print(doc.page_content)

        vector_service = VectorStoreService()
        vector_service.create_store(documents)

        return Response({"message": "PDF processed successfully."})