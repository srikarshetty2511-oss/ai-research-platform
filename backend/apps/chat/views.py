from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.llm.groq_service import GroqLLMService


class ChatView(APIView):
    """
    Simple synchronous chat endpoint.
    """

    def post(self, request):
        message = request.data.get("message")

        if not message:
            return Response(
                {"error": "Message field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            llm_service = GroqLLMService()
            reply = llm_service.generate_response(message)

            return Response(
                {
                    "success": True,
                    "response": reply,
                },
                status=status.HTTP_200_OK,
            )

        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except RuntimeError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        except Exception:
            return Response(
                {"error": "Internal server error."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )