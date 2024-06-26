from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from backend.serving_engine import ChatRequest, ChatResponse, engine as vllm_engine
from backend.database import get_db, UserLogCreate, FeedbackCreate, UserLog, Feedback
from backend.message_queue import MessageQueue
from backend.websocket_manager import manager
from pydantic import BaseModel
import asyncio
import json

app = FastAPI()

# Initialize the message queue
mq = MessageQueue()

class CombinedChatRequest(BaseModel):
    prompt: str
    user_id: str

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await process_websocket_message(data, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def process_websocket_message(data: str, client_id: str):
    message = json.loads(data)
    if message['type'] == 'chat_request':
        await process_chat_request(message['prompt'], client_id)

async def process_chat_request(prompt: str, client_id: str):
    db = next(get_db())
    try:
        # Log the user request
        log = UserLogCreate(user_id=client_id, action="chat_request", details=prompt)
        db_log = UserLog(**log.dict())
        db.add(db_log)
        db.commit()

        # Generate responses using vLLM
        sampling_params = vllm_engine.SamplingParams(temperature=0.7, max_tokens=100)
        outputs = await asyncio.gather(
            vllm_engine.generate(prompt, sampling_params),
            vllm_engine.generate(prompt, sampling_params)
        )

        response1 = outputs[0].outputs[0].text
        response2 = outputs[1].outputs[0].text

        # Send responses back to the client
        await manager.send_message(json.dumps({
            'type': 'chat_response',
            'response1': response1,
            'response2': response2
        }), client_id)

    except Exception as e:
        await manager.send_message(json.dumps({
            'type': 'error',
            'message': str(e)
        }), client_id)

@app.post("/feedback")
async def feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    db_feedback = Feedback(**feedback.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return {"status": "success", "feedback_id": db_feedback.id}

@app.get("/logs/{user_id}")
async def get_user_logs(user_id: str, db: Session = Depends(get_db)):
    logs = db.query(UserLog).filter(UserLog.user_id == user_id).all()
    return logs

@app.get("/feedback/{user_id}")
async def get_user_feedback(user_id: str, db: Session = Depends(get_db)):
    feedback = db.query(Feedback).filter(Feedback.user_id == user_id).all()
    return feedback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)