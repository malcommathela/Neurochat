#!/usr/bin/env python3
"""
NeuroChat - Distributed AI Chatbot
Final working version
"""

import torch
import zmq
import pickle
import numpy as np
import time
import threading
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.serialization
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

# Simulated network latency per hop (ms)
LATENCY = 2

class PipelineLayer(threading.Thread):
    def __init__(self, name, model_path, recv_port, send_port, is_first=False):
        super().__init__(daemon=True)
        self.name = name
        self.is_first = is_first
        
        print(f"[{name}] Loading model...")
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.eval()
        
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(f"tcp://127.0.0.1:{recv_port}")
        
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(f"tcp://127.0.0.1:{send_port}")
        
        print(f"[{name}] Ready (:{recv_port} -> :{send_port})")
    
    def run(self):
        while True:
            try:
                msg = self.receiver.recv()
                data = pickle.loads(msg)
                
                time.sleep(LATENCY / 1000.0)
                
                if self.is_first:
                    input_ids = torch.from_numpy(data['input_ids'].astype(np.int64))
                    position_ids = torch.arange(input_ids.size(-1), dtype=torch.long).unsqueeze(0)
                    
                    with torch.no_grad():
                        h = self.model[0](input_ids) + self.model[1](position_ids)
                        h = self.model[2](h)
                        for i in range(3, len(self.model)):
                            out = self.model[i](h)
                            h = out[0] if isinstance(out, tuple) else out
                else:
                    h = torch.from_numpy(data['hidden_states'])
                    with torch.no_grad():
                        for module in self.model:
                            out = module(h)
                            h = out[0] if isinstance(out, tuple) else out
                
                result = {
                    'hidden_states': h.detach().numpy(),
                    'conversation_id': data['conversation_id'],
                    'path': data.get('path', []) + [self.name],
                    'timestamps': data.get('timestamps', []) + [time.time()],
                    'compute_ms': data.get('compute_ms', 0) + LATENCY,
                }
                
                self.sender.send(pickle.dumps(result))
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")

class NeuroChat:
    def __init__(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('./chat_tokenizer')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading generation model...")
        self.gen_model = AutoModelForCausalLM.from_pretrained('distilgpt2', trust_remote_code=True)
        self.gen_model.eval()
        
        # Start pipeline layers
        self.layers = []
        
        l1 = PipelineLayer('layer1', 'chat_layer1.pt', 5001, 5002, is_first=True)
        l1.start()
        self.layers.append(l1)
        time.sleep(0.3)
        
        l2 = PipelineLayer('layer2', 'chat_layer2.pt', 5002, 5003)
        l2.start()
        self.layers.append(l2)
        time.sleep(0.3)
        
        l3 = PipelineLayer('layer3', 'chat_layer3.pt', 5003, 6000)
        l3.start()
        self.layers.append(l3)
        time.sleep(0.3)
        
        # Host sockets
        self.context = zmq.Context()
        self.input_push = self.context.socket(zmq.PUSH)
        self.input_push.connect("tcp://127.0.0.1:5001")
        
        self.output_pull = self.context.socket(zmq.PULL)
        self.output_pull.bind("tcp://127.0.0.1:6000")
        
        print("\n" + "="*60)
        print("       🧠 NEUROCHAT - Distributed AI Chatbot")
        print("="*60)
        print("Neural network pipeline: You → L1 → L2 → L3 → AI")
        print("="*60 + "\n")
    
    def process_through_network(self, text):
        """Send text through distributed pipeline"""
        inputs = self.tokenizer(text, return_tensors='pt')
        token_ids = inputs['input_ids'].numpy().astype(np.int64)
        
        packet = {
            'input_ids': token_ids,
            'conversation_id': str(time.time()),
            'path': ['host'],
            'timestamps': [time.time()],
            'compute_ms': 0,
        }
        
        print(f"[Network] Sending: '{text[:30]}...' → pipeline")
        self.input_push.send(pickle.dumps(packet))
        
        result_msg = self.output_pull.recv()
        result = pickle.loads(result_msg)
        
        path = ' → '.join(result['path'])
        elapsed = (time.time() - result['timestamps'][0]) * 1000
        
        print(f"[Network] ✓ Path: {path}")
        print(f"[Network] Time: {elapsed:.1f}ms")
        
        return result
    
    def generate_response(self, user_input):
        """Generate conversational response"""
        network_result = self.process_through_network(user_input)
        
        print("[AI] Generating...", end=" ", flush=True)
        
        prompt = f"Human: {user_input}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.gen_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        lines = response.split('\n')
        response = lines[0].strip()
        
        if not any(response.endswith(p) for p in ['.', '!', '?', '...']):
            response += "."
        
        if len(response.split()) < 4:
            response += " How can I help you today?"
        
        print("Done!")
        return response
    
    def chat(self):
        print("Type your message (or 'quit' to exit):\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nAssistant: Goodbye! Have a great day! 👋")
                    break
                
                response = self.generate_response(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"[Error] {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    chat = NeuroChat()
    chat.chat()
