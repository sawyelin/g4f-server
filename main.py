from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import g4f
from g4f import Provider
import time
import json
from datetime import datetime
import base64
from io import BytesIO

app = FastAPI(title="G4F API", version="1.0.0")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
RATE_LIMIT = 60  # requests per minute
rate_limit_dict = {}

# Available models mapping
AVAILABLE_MODELS = {
    "gpt-4": g4f.models.gpt_4,
    "gpt-3.5-turbo": g4f.models.gpt_35_turbo,
    "gpt-3.5-turbo-16k": g4f.models.gpt_35_turbo_16k,
    "claude-2": g4f.models.claude_2,
    "claude-instant-1": g4f.models.claude_instant_1,
    "palm-2": g4f.models.palm_2,
    "gemini-pro": g4f.models.gemini_pro,
    "llama-2-70b": g4f.models.llama2_70b,
    "code-llama-34b": g4f.models.codellama_34b,
    "falcon-180b": g4f.models.falcon_180b,
}

# Available providers
AVAILABLE_PROVIDERS = {
    "DeepAi": Provider.DeepAi,
    "Bard": Provider.Bard,
    "Bing": Provider.Bing,
    "OpenaiChat": Provider.OpenaiChat,
    "DeepInfra": Provider.DeepInfra,
    "You": Provider.You,
    "H2o": Provider.H2o,
    "Gemini": Provider.Gemini,
}

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[List[Function]] = None
    provider: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"
    model: Optional[str] = "stable-diffusion"
    n: Optional[int] = 1

def check_rate_limit(client_ip: str) -> bool:
    current_time = time.time()
    if client_ip in rate_limit_dict:
        last_request_time, count = rate_limit_dict[client_ip]
        if current_time - last_request_time >= 60:
            rate_limit_dict[client_ip] = (current_time, 1)
            return True
        if count >= RATE_LIMIT:
            return False
        rate_limit_dict[client_ip] = (last_request_time, count + 1)
    else:
        rate_limit_dict[client_ip] = (current_time, 1)
    return True

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/v1/models")
async def list_models():
    models_list = []
    for model_id, model in AVAILABLE_MODELS.items():
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "g4f",
            "permission": [],
            "root": "g4f",
            "parent": None,
        })
    
    return {
        "object": "list",
        "data": models_list
    }

@app.get("/v1/providers")
async def list_providers():
    return {
        "object": "list",
        "data": list(AVAILABLE_PROVIDERS.keys())
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request, body: ChatCompletionRequest):
    client_ip = request.client.host
    
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        model = AVAILABLE_MODELS.get(body.model)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {body.model} not found")
        
        provider = None
        if body.provider:
            provider = AVAILABLE_PROVIDERS.get(body.provider)
            if not provider:
                raise HTTPException(status_code=400, detail=f"Provider {body.provider} not found")
        
        messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
        
        # Handle function calling
        if body.functions:
            messages[-1]["content"] += "\n\nAvailable functions:\n" + json.dumps([fn.dict() for fn in body.functions])
        
        response = g4f.ChatCompletion.create(
            model=model,
            provider=provider,
            messages=messages,
            temperature=body.temperature,
            stream=body.stream
        )
        
        if body.stream:
            async def generate():
                try:
                    for chunk in response:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'index': 0}]})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
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
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
        
        return JSONResponse(content=completion_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations")
async def create_image(request: Request, body: ImageGenerationRequest):
    client_ip = request.client.host
    
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Use g4f's image generation capabilities
        response = g4f.images.create(
            prompt=body.prompt,
            model=body.model,
        )
        
        # Convert image data to base64
        if isinstance(response, bytes):
            image_base64 = base64.b64encode(response).decode('utf-8')
        else:
            # If response is already a URL or other format, return as is
            image_base64 = response
        
        return {
            "created": int(time.time()),
            "data": [
                {
                    "url": image_base64 if "http" in image_base64 else f"data:image/jpeg;base64,{image_base64}",
                    "b64_json": image_base64 if "http" not in image_base64 else None
                }
                for _ in range(body.n)
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: Request, body: ChatCompletionRequest):
    """Legacy endpoint for text completions"""
    return await create_chat_completion(request, body)

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
