from fastapi import APIRouter
from fastapi.responses import FileResponse
from service import ConversationalRetrievalService, ApiResponse, QuestionRequest
from pathlib import Path

import time

router = APIRouter()
static_dir = Path("static_files")

api_key = "Chat_GPT_API_Key"
text_file_path = './leave_policy.txt'
service = ConversationalRetrievalService(api_key, text_file_path)


@router.post("/generate", response_model=ApiResponse)
async def ask(question_request: QuestionRequest):
    question = question_request.question
    response_text = service.get_response(question)
    return service.build_response(response_text)


@router.get("/get_initial_message")
async def get_initial_message():
    data = {
        "author": {
            "firstName": "H",
            "id": "4c2307ba-3d40-442f-b1ff-b271f63904ca",
            "lastName": "R",
            "imageUrl": "http://localhost:8000/image/pfp"
        },
        "createdAt": int(time.time() * 1000),
        "id": "c67ed376-52bf-4d4e-ba2a-7a0f8467b22a",
        "status": "seen",
        "text": "Hi, How can I assist you today",
        "type": "text",
    }
    return data


@router.get("/image/pfp")
async def get_image():
    file_path = static_dir / "pfp-1.jpg"

    # Check if the file exists
    if file_path.is_file():
        return FileResponse(file_path)
    else:
        return {"error": "File not found"}
