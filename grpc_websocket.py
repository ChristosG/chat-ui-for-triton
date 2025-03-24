# main.py
import os
import json
import queue
import asyncio
import threading
import time  # for simulating token delays
from functools import partial

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from transformers import AutoTokenizer
from tavily import TavilyClient 
import re
import os
import logging
from typing import Optional
import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer
import json 
from pydantic import BaseModel, Extra
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

LLAMA_MODEL_NAME = 'ensemble'
TRITON_SERVER_URL = 'host.docker.internal:8000'
#TOKENIZER_PATH = '/engines/Llama-Krikri-8B-Instruct/'DeepSeek-R1-Distill-Llama-8B
TOKENIZER_PATH = '/engines/DeepSeek-R1-Distill-Qwen-14B'

class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/{LLAMA_MODEL_NAME}/generate"
    
    class Config:
        extra = 'forbid'

    @property 
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        temperature: float,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 8192,
                "temperature": temperature,
                "top_k": 50,
            }
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.llm_url, json=payload, headers=headers)
            response.raise_for_status()
            translation = response.json().get('text_output', '')
            if not translation:
                raise ValueError("No 'text_output' field in the response.")
            return translation
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM request failed: {e}")
            return ""
        except ValueError as ve:
            logging.error(f"LLM response error: {ve}")
            return ""

    @property
    def _identifying_params(self) -> dict:
        return {"llmUrl": self.llm_url}

llm = TritonLLM()

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logging.info("Tokenizer loaded successfully in llm_module.")
except Exception as e:
    logging.error(f"Failed to load tokenizer in llm_module: {e}")
    exit(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3000/ws",
        "https://zelime.duckdns.org",
        "https://zelime.duckdns.org/ws/infer",
        "https://zelime.duckdns.org/ws",
        "localhost:3000/ws",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    exit(1)


def prepare_tensor(name, input_array):
    t = grpcclient.InferInput(name, input_array.shape, np_to_triton_dtype(input_array.dtype))
    t.set_data_from_numpy(input_array)
    return t

# Extend the per-connection user data to hold conversation history.
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
        self.conversation_history = [
            # {
            #     "role": "system",
            #     "content": "Eisai o Mastoras, o prosopikos voithos tou Chris. Help him anywhere you can! if the user ask about Christos Grigoriadis, or Χρήστος Γρηγοριάδης or Χρηστος in general just respond with an `Ακραίος`"
            # }
        ]
        self.response_tokens = []

TAVILY_API_KEY = "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"

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def execute_tavily_search(query: str) -> str:
    """Execute search and format results"""
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_answer=True,
        max_results=3
    )
    
    # Format results with source credibility assessment
    if not response.get('results'):
        return "Δεν βρέθηκαν σχετικά αποτελέσματα."
    
    results = []
    for res in response['results'][:3]:  # Take top 3
        source = f"Πηγή: {res['url']} ({'Αξιόπιστη' if res['score'] > 0.7 else 'Μέτρια αξιοπιστία'})"
        content = res.get('content', 'Δεν υπάρχει περιγραφή.')
        results.append(f"{source}\n{content}")
    
    return "\n\n".join(results)

def extract_search_query(response: str) -> Optional[str]:
    """Extract search query from JSON response"""
    try:
        # Look for JSON pattern
        json_str = re.search(r'```json\s*({.+?})\s*```', response, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group(1))
            if data.get('action') == 'tavily_search':
                return data['query']
        return None
    except (json.JSONDecodeError, KeyError):
        return None
    

def tavily_search(query: str) -> str:
    conversation_history = [
    {
        "role": "system",
        "content": """Eisai o Mastoras, o prosopikos voithos tou Chris. Help him anywhere you can!
When needing fresh information, follow these steps:
1. 🤔 Analyze if the query requires web search (recent info/unknown facts)
2. 🔍 If yes, identify 1-3 key search terms in Greek/English
3. 🛠️ Output JSON with search action and query
4. 📊 Synthesize final answer from results

Example thought process:
User: "Ποιες είναι οι νέες τάσεις στην ελληνική γαστρονομία;"
Mastoras should think: "Για να απαντήσω, χρειάζομαι πρόσφατες πληροφορίες. Βασικές λέξεις-κλειδιά: 'νεότερες τάσεις ελληνική γαστρονομία 2024'"

Tool format:
```json
{"action": "tavily_search", "query": "your_keywords_here"}
```"""
    }
]
    conversation_history.append({"role": "user", "content": query})
    
    prompt = tokenizer.apply_chat_template(
        conversation_history, 
        add_generation_prompt=True, 
        tokenize=False
    )
    response = llm(prompt, temperature=0.2)
    
    search_query = extract_search_query(response)

    
    if search_query:
        search_results = execute_tavily_search(search_query)
        
        conversation_history.extend([
            {
                "role": "assistant",
                "content": f"🔍 Αναζήτηση για: {search_query}\n{response}"
            },
            {
                "role": "system",
                "content": f"Αποτελέσματα αναζήτησης:\n{search_results}"
            }
        ])
        
        final_prompt = tokenizer.apply_chat_template(
            conversation_history,
            add_generation_prompt=True,
            tokenize=False
        )
        final_response = llm(final_prompt, temperature=0.2)


        
        conversation_history.append({"role": "assistant", "content": final_response})
        return final_response
    
    # If no search needed
    conversation_history.append({"role": "assistant", "content": response})
    return response
    # response = tavily_client.search(
    #     query=query,
    #     search_depth="advanced",
    #     include_answer="advanced",
    #     max_results=5  
    # )
    # answer = response.get('answer', 'No answer provided.')
    # return answer 
        

def ws_callback(user_data, result, error):
    if error:
        user_data._completed_requests.put("Error: " + str(error))
    else:
        token = result.as_numpy('text_output')[0].decode("utf-8")
        user_data.response_tokens.append(token)
        user_data._completed_requests.put(token)

def safe_float(value, default):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default




def blocking_inference_with_timeout(payload, user_data, timeout=60):
    future = executor.submit(blocking_inference, payload, user_data)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        user_data._completed_requests.put(Exception("Inference timeout."))
        logging.error("Inference timed out.")
        
def blocking_inference(payload, user_data):
    client = grpcclient.InferenceServerClient(url="host.docker.internal:8001")

    max_tokens_val = safe_int(payload.get("max_tokens"), 8192)
    temperature_val = safe_float(payload.get("temperature"), 0.4)
    top_k_val = safe_int(payload.get("top_k"), 40)
    top_p_val = safe_float(payload.get("top_p"), 0.9)
    repetition_penalty_val = safe_float(payload.get("repetition_penalty"), 1.0)
    frequency_penalty_val = safe_float(payload.get("frequency_penalty"), 0.0)
    presence_penalty_val = safe_float(payload.get("presence_penalty"), 0.0)

    prompt = tokenizer.apply_chat_template(user_data.conversation_history, add_generation_prompt=True, tokenize=False)

    inputs = [
        prepare_tensor("text_input", np.array([[prompt]], dtype=object)),
        prepare_tensor("max_tokens", np.array([[max_tokens_val]], dtype=np.int32)),
        prepare_tensor("stream", np.array([[True]], dtype=bool)),
        prepare_tensor("beam_width", np.array([[1]], dtype=np.int32)),
        prepare_tensor("temperature", np.array([[temperature_val]], dtype=np.float32)),
        prepare_tensor("top_k", np.array([[top_k_val]], dtype=np.int32)),
        prepare_tensor("top_p", np.array([[top_p_val]], dtype=np.float32)),
        prepare_tensor("repetition_penalty", np.array([[repetition_penalty_val]], dtype=np.float32)),
        prepare_tensor("frequency_penalty", np.array([[frequency_penalty_val]], dtype=np.float32)),
        prepare_tensor("presence_penalty", np.array([[presence_penalty_val]], dtype=np.float32)),
    ]

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    client.start_stream(callback=partial(ws_callback, user_data))
    client.async_stream_infer("ensemble", inputs, outputs=outputs, request_id="")
    client.stop_stream()


@app.websocket("/ws")
async def websocket_infer(websocket: WebSocket):
    await websocket.accept()
    # Create per-connection user data that holds conversation history.
    user_data = UserData()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except Exception:
                payload = {"prompt": data}
            # Append the new user message to the conversation history.
            user_message = payload.get("prompt", data)
            if payload.get("search", False):
                user_message = tavily_search(payload["prompt"])

            user_data.conversation_history.append({"role": "user", "content": user_message})
            # Reset response tokens for the new turn.
            user_data.response_tokens = []
            
            thread = threading.Thread(target=blocking_inference_with_timeout, args=(payload, user_data, 60))
            thread.start()
            while thread.is_alive() or not user_data._completed_requests.empty():
                try:
                    token = user_data._completed_requests.get(timeout=0.1)
                    if isinstance(token, InferenceServerException) or isinstance(token, Exception):
                        await websocket.send_text("Error: " + str(token))
                    else:
                        await websocket.send_text(token)
                except queue.Empty:
                    await asyncio.sleep(0.1)
            full_response = "".join(user_data.response_tokens)
            user_data.conversation_history.append({"role": "assistant", "content": full_response})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_text("Error: " + str(e))
        await websocket.close()

@app.post("/infer")
async def infer_endpoint(data: dict):
    prompt = data.get("prompt")
    messages = [ 
        {"role": "system", "content": "Eisai o Mastoras, o prosopikos voithos tou Chris. Help him anywhere you can!"},
        {"role": "user", "content": prompt}
    ]
    data["prompt"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    user_data = UserData()
    thread = threading.Thread(target=blocking_inference, args=(data, user_data))
    thread.start()
    thread.join()
    tokens = []
    while not user_data._completed_requests.empty():
        tokens.append(user_data._completed_requests.get())
    full_text = "".join([t if not isinstance(t, Exception) else "" for t in tokens])
    return JSONResponse({"output": full_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=True)
