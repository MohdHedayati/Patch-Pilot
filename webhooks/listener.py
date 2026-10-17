import os
import logging
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from webhooks.issue_handler import handle_issue_event

# Force all internal INFO logs to print to the terminal
logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI(title="Patch Pilot Webhook Server")

def process_webhook_background(payload: dict):
    try:
        print("\n[+] Starting pipeline execution...")
        gemini_key = os.getenv("GEMINI_API_KEY")
        github_token = os.getenv("GITHUB_TOKEN")
        
        result = handle_issue_event(
            payload, 
            token=github_token, 
            gemini_key=gemini_key
        )
        print(f"\n[+] Webhook Processing Complete: {result.get('message')}\n")
    except Exception as e:
        print("\n[!] PIPELINE CRASHED:")
        traceback.print_exc()

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    action = payload.get("action")
    if "issue" in payload and action in ["opened", "created"]:
        print(f"\n[+] Received triggering event '{action}' for issue #{payload['issue']['number']}")
        background_tasks.add_task(process_webhook_background, payload)
        return {"status": "processing", "issue": payload["issue"]["number"]}
    return {"status": "ignored", "reason": f"Action '{action}' is not tracked"}