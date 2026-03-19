#!/usr/bin/env python3
import torch
import torch.nn as nn
import zmq
import pickle
import numpy as np
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.serialization
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

LATENCY_LAYER1 = 2
LATENCY_LAYER2 = 3
LATENCY_LAYER3 = 2

class LocalLayer(threading.Thread):
    def __init__(self, name, model_path, input_port, output_port, latency, is_first=False, is_last=False):
        super().__init__()
        self.name = name
        self.model_path = model_path
        self.input_port = input_port
        self.output_port = output_port
        self.latency = latency
        self.is_first = is_first
        self.is_last = is_last
        self.daemon = True
        
        print(f"[{name}] Loading model...")
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.eval()
        print(f"[{name}] Ready on localhost:{input_port} -> localhost:{output_port}")
    
    def run(self):
        context = zmq.Context()
        receiver = context.socket(zmq.REP)
        receiver.bind(f"tcp://127.0.0.1:{self.input_port}")
        
        sender = context.socket(zmq.REQ)
        sender.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout
        sender.connect(f"tcp://127.0.0.1:{self.output_port}")
        
        while True:
            try:
                msg = receiver.recv()
                data = pickle.loads(msg)
                
                time.sleep(self.latency / 1000.0)
                
                # Process
                if self.is_first:
                    np_array = data['input_ids']
                    if np_array.dtype != np.int64:
                        np_array = np_array.astype(np.int64)
                    
                    input_ids = torch.from_numpy(np_array)
                    position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long)
                    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                    
                    print(f"[{self.name}] Processing tokens: {input_ids.shape}")
                    
                    with torch.no_grad():
                        start = time.time()
                        token_embed = self.model[0](input_ids)
                        pos_embed = self.model[1](position_ids)
                        hidden = token_embed + pos_embed
                        hidden = self.model[2](hidden)
                        
                        # Process through transformer blocks
                        for i in range(3, len(self.model)):
                            layer_output = self.model[i](hidden)
                            hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                        
                        compute_time = (time.time() - start) * 1000
                else:
                    np_array = data['hidden_states']
                    tensor_data = torch.from_numpy(np_array)
                    
                    print(f"[{self.name}] Processing hidden: {tensor_data.shape}")
                    
                    with torch.no_grad():
                        start = time.time()
                        hidden = tensor_data
                        
                        for module in self.model:
                            layer_output = module(hidden)
                            hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                        
                        compute_time = (time.time() - start) * 1000
                
                # Convert output
                output_np = hidden.detach().numpy()
                
                # Send to next
                result = {
                    'hidden_states': output_np,
                    'conversation_id': data['conversation_id'],
                    'path': data.get('path', []) + [self.name],
                    'timestamps': data.get('timestamps', []) + [time.time()],
                    'compute_time_ms': data.get('compute_time_ms', 0) + compute_time,
                }
                
                print(f"[{self.name}] Sending to next layer...")
                sender.send(pickle.dumps(result))
                
                try:
                    ack = sender.recv()
                    print(f"[{self.name}] Got ack")
                except zmq.error.Again:
                    print(f"[{self.name}] Timeout waiting for ack!")
                
                # Ack previous
                receiver.send(pickle.dumps({'status': 'ok'}))
                print(f"[{self.name}] Done!")
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    receiver.send(pickle.dumps({'status': 'error', 'message': str(e)}))
                except:
                    pass

class NeuroChatLocal:
    def __init__(self):
        print("Loading AI model...")
        self.tokenizer = AutoTokenizer.from_pretrained('./chat_tokenizer')
        self.model = AutoModelForCausalLM.from_pretrained('distilgpt2', trust_remote_code=True)
        self.model.eval()
        
        self.layers = []
        
        l1 = LocalLayer('layer1', 'chat_layer1.pt', 5001, 5002, LATENCY_LAYER1, is_first=True)
        l1.start()
        time.sleep(0.5)
        
        l2 = LocalLayer('layer2', 'chat_layer2.pt', 5002, 5003, LATENCY_LAYER2)
        l2.start()
        time.sleep(0.5)
        
        l3 = LocalLayer('layer3', 'chat_layer3.pt', 5003, 6000, LATENCY_LAYER3, is_last=True)
        l3.start()
        time.sleep(0.5)
        
        self.context = zmq.Context()
        self.to_layer1 = self.context.socket(zmq.REQ)
        self.to_layer1.setsockopt(zmq.RCVTIMEO, 30000)
        self.to_layer1.connect("tcp://127.0.0.1:5001")
        
        self.from_layer3 = self.context.socket(zmq.REP)
        self.from_layer3.setsockopt(zmq.RCVTIMEO, 30000)
        self.from_layer3.bind("tcp://127.0.0.1:6000")
        
        print("\n" + "="*60)
        print("       🧠 NEUROCHAT - Distributed AI Chatbot")
        print("="*60)
        print("Type messages (or 'quit' to exit):\n")
    
    def chat(self):
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                # Tokenize
                inputs = self.tokenizer(user_input, return_tensors='pt')
                token_ids = inputs['input_ids'].numpy().astype(np.int64)
                
                packet = {
                    'input_ids': token_ids,
                    'conversation_id': f"chat_{time.time()}",
                    'path': ['host'],
                    'timestamps': [time.time()],
                }
                
                print(f"[Network] Sending to pipeline...")
                self.to_layer1.send(pickle.dumps(packet))
                
                # Wait for layer1 ack
                try:
                    ack = self.to_layer1.recv()
                    print(f"[Network] Layer1 done")
                except zmq.error.Again:
                    print("[Network] Timeout waiting for Layer1!")
                    continue
                
                # Wait for final result from layer3
                print(f"[Network] Waiting for final result...")
                try:
                    result_msg = self.from_layer3.recv()
                    result = pickle.loads(result_msg)
                    self.from_layer3.send(pickle.dumps({'status': 'ok'}))
                    
                    path = ' → '.join(result['path'])
                    total_time = (time.time() - result['timestamps'][0]) * 1000
                    compute_time = result['compute_time_ms']
                    
                    print(f"[Network] ✓ Complete!")
                    print(f"[Network] Path: {path}")
                    print(f"[Network] Total time: {total_time:.1f}ms | Compute: {compute_time:.1f}ms")
                    
                except zmq.error.Again:
                    print("[Network] Timeout waiting for Layer3 result!")
                    continue
                except Exception as e:
                    print(f"[Network] Error: {e}")
                    continue
                
                # Generate response
                print("[Generating]...", end=" ", flush=True)
                prompt = f"Human: {user_input}\nAssistant:"
                inputs = self.tokenizer(prompt, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=25,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(prompt, "").strip()
                
                if not response.endswith(('.', '!', '?')):
                    response += "."
                if len(response.split()) < 5:
                    response += " How can I help you?"
                
                print("Done!")
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"[Error] {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    chat = NeuroChatLocal()
    chat.chat()
