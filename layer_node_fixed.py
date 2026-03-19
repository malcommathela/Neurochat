#!/usr/bin/env python3
import torch
import torch.nn as nn
import zmq
import pickle
import sys
import time
import os

# Fix for PyTorch 2.6+ weights_only default change
import torch.serialization
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

CONFIG = {
    'layer1': {
        'model_path': './chat_layer1.pt',
        'bind_addr': 'tcp://10.0.0.1:5001',
        'next_addr': 'tcp://10.0.1.2:5002',
    },
    'layer2': {
        'model_path': './chat_layer2.pt',
        'bind_addr': 'tcp://10.0.1.2:5002',
        'next_addr': 'tcp://10.0.2.3:5003',
    },
    'layer3': {
        'model_path': './chat_layer3.pt',
        'bind_addr': 'tcp://10.0.2.3:5003',
        'next_addr': 'tcp://10.0.0.3:6000',
        'is_output': True,
    }
}

class LayerNode:
    def __init__(self, layer_name):
        self.config = CONFIG[layer_name]
        self.layer_name = layer_name
        self.is_output = self.config.get('is_output', False)
        
        print(f"[{layer_name}] Loading model from {self.config['model_path']}...")
        
        if not os.path.exists(self.config['model_path']):
            print(f"[{layer_name}] ERROR: Model file not found at {os.path.abspath(self.config['model_path'])}")
            print(f"[{layer_name}] Current directory: {os.getcwd()}")
            print(f"[{layer_name}] Files here: {os.listdir('.')}")
            sys.exit(1)
        
        # Load with weights_only=False for PyTorch 2.6+ compatibility
        self.model = torch.load(self.config['model_path'], map_location='cpu', weights_only=False)
        self.model.eval()
        print(f"[{layer_name}] Model loaded successfully!")
        
        self.context = zmq.Context()
        
        self.receiver = self.context.socket(zmq.REP)
        try:
            self.receiver.bind(self.config['bind_addr'])
            print(f"[{layer_name}] Bound to {self.config['bind_addr']}")
        except zmq.ZMQError as e:
            print(f"[{layer_name}] ERROR binding to {self.config['bind_addr']}: {e}")
            sys.exit(1)
        
        self.sender = self.context.socket(zmq.REQ)
        try:
            self.sender.connect(self.config['next_addr'])
            print(f"[{layer_name}] Connected to {self.config['next_addr']}")
        except zmq.ZMQError as e:
            print(f"[{layer_name}] ERROR connecting to {self.config['next_addr']}: {e}")
            sys.exit(1)
    
    def process(self, hidden_states):
        with torch.no_grad():
            start = time.time()
            output = self.model(hidden_states)
            elapsed = (time.time() - start) * 1000
            return output, elapsed
    
    def run(self):
        print(f"[{self.layer_name}] Ready for processing! Waiting for connections...")
        
        while True:
            try:
                message = self.receiver.recv()
                data = pickle.loads(message)
                
                hidden_states = data['hidden_states']
                conv_id = data.get('conversation_id', 'unknown')
                
                print(f"[{self.layer_name}] Processing {conv_id}, input shape: {hidden_states.shape}")
                
                output, compute_time = self.process(hidden_states)
                
                next_data = {
                    'hidden_states': output,
                    'conversation_id': conv_id,
                    'path': data.get('path', []) + [self.layer_name],
                    'timestamps': data.get('timestamps', []) + [time.time()],
                    'compute_time_ms': data.get('compute_time_ms', 0) + compute_time,
                }
                
                self.sender.send(pickle.dumps(next_data))
                ack = self.sender.recv()
                
                self.receiver.send(pickle.dumps({
                    'status': 'ok',
                    'layer': self.layer_name,
                    'output_shape': list(output.shape),
                    'compute_ms': compute_time
                }))
                
            except Exception as e:
                print(f"[{self.layer_name}] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    self.receiver.send(pickle.dumps({'status': 'error', 'message': str(e)}))
                except:
                    pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 layer_node_fixed.py <layer1|layer2|layer3>")
        sys.exit(1)
    
    layer_name = sys.argv[1]
    if layer_name not in CONFIG:
        print(f"Unknown layer: {layer_name}. Use layer1, layer2, or layer3")
        sys.exit(1)
    
    node = LayerNode(layer_name)
    node.run()
