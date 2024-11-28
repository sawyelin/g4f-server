from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import g4f
import time
import os
from datetime import datetime

app = FastAPI(title="G4F API", version="1.0.0")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
RATE_LIMIT = 60  # requests per minute
rate_limit_dict = {}

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def check_rate_limit(client_ip: str) -> bool:
    current_time = time.time()
    if client_ip in rate_limit_dict:
        last_request_time, count = rate_limit_dict[client_ip]
        # Reset counter if minute has passed
        if current_time - last_request_time >= 60:
            rate_limit_dict[client_ip] = (current_time, 1)
            return True
        # Increment counter if within the same minute
        if count >= RATE_LIMIT:
            return False
        rate_limit_dict[client_ip] = (last_request_time, count + 1)
    else:
        rate_limit_dict[client_ip] = (current_time, 1)
    return True

@app.get("/")
async def root():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, body: ChatCompletionRequest):
    client_ip = request.client.host
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Convert messages to the format expected by g4f
        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
        
        # Get response from g4f
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_35_turbo,  # You can map different models here
            messages=messages,
            temperature=body.temperature,
        )
        
        # Format response to match OpenAI's API structure
        completion_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # G4F doesn't provide token counts
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
        
        return JSONResponse(content=completion_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    # Return a list of supported models
    return {
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "g4f",
            }
        ],
        "object": "list"
    }

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "api_error",
                "code": exc.status_code,
                "param": None,
            }
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
