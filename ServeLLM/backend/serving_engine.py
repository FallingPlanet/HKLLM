import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
import uvicorn

app = FastAPI()

# Initialize the vLLM engine
model_name = "your_model_name_here"  # Replace with your actual model name
engine = AsyncLLMEngine.from_pretrained(model_name)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response1: str
    response2: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Generate two responses for DPO
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        outputs = await asyncio.gather(
            engine.generate(request.prompt, sampling_params),
            engine.generate(request.prompt, sampling_params)
        )

        response1 = outputs[0].outputs[0].text
        response2 = outputs[1].outputs[0].text

        return ChatResponse(response1=response1, response2=response2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)