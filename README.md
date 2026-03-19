🧠 NeuroChat - Distributed AI Chatbot

A proof-of-concept demonstrating **Network-as-Neural-Network** architecture, where AI model layers are distributed across a virtual network topology instead of running centralized.

## 🎯 Concept

Traditional AI: Model runs on centralized GPU/CPU
User → [Central Server with Full Model] → Response
plain
Copy

NeuroChat: Model layers distributed across network nodes
User → [Layer 1: Embedding] → [Layer 2: Transform] → [Layer 3: Generation] → Response
↑                    ↑                    ↑
Node A               Node B               Node C
(10.0.0.1)           (10.0.1.2)           (10.0.2.3)
plain
Copy

## 🏗️ Architecture

| Component | Technology | Role |
|-----------|-----------|------|
| **Layer 1** | DistilGPT-2 Embeddings + 2 blocks | Token/Position embedding |
| **Layer 2** | 2 Transformer blocks | Hidden state processing |
| **Layer 3** | 2 Transformer blocks + LM Head | Text generation |
| **Network** | Linux Network Namespaces + veth | Virtual topology |
| **Protocol** | ZeroMQ (PUSH/PULL) | Message passing |
| **Latency** | `tc netem` | Simulated network delay |

## 🚀 Quick Start

### Prerequisites
- Linux (tested on Kali/Ubuntu in WSL2)
- Python 3.10+
- 2GB+ RAM

### Setup

# 1. Clone and enter directory
cd neurochat

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and split AI model
python3 prepare_model.py

# 5. Setup virtual network
sudo ./setup_network.sh

# 6. Run (in 4 separate terminals)
./run_layer.sh layer1
./run_layer.sh layer2
./run_layer.sh layer3
python3 neurochat.py

📁 File Structure
neurochat/
├── prepare_model.py      # Downloads & splits DistilGPT-2 into 3 layers
├── setup_network.sh      # Creates virtual network topology
├── run_layer.sh          # Launches layer in network namespace
├── layer_node_fixed.py   # Layer implementation (ZeroMQ + PyTorch)
├── neurochat.py          # Host chat interface
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file

🔬 How It Works
Model Splitting: prepare_model.py splits DistilGPT-2 into 3 pipeline stages
Network Topology: setup_network.sh creates isolated namespaces with virtual Ethernet
Message Passing: Token IDs flow as "spikes" through ZeroMQ sockets
Distributed Inference: Each layer processes and forwards hidden states
Response Generation: Final output generates conversational text

📊 Performance
  
Metric	                Value
Network Hops	          3 (Layer1→Layer2→Layer3)
Simulated Latency	      ~6ms total (2ms per hop)
Model Size	            66M parameters (DistilGPT-2)
Actual Throughput       ~1 req/sec (CPU, prototype)

Note: This is a research prototype demonstrating distributed architecture, not optimized for speed. Production systems would use GPU acceleration, RDMA, and tensor parallelism.

🎓 Educational Value
This project demonstrates:
Distributed Neural Networks: Model parallelism across network nodes
Network Topology as Architecture: Physical layout = computational graph
Message Passing as Synaptic Transmission: ZeroMQ = neural spikes
Fault Tolerance: Network can route around failed nodes
Edge AI: Intelligence distributed to network edges
🔮 Future Improvements
[ ] GPU acceleration (CUDA)
[ ] Tensor parallelism (vs current pipeline)
[ ] Async micro-batching
[ ] RDMA/NCCL for zero-copy transfer
[ ] In-network computation (P4 switches)
[ ] Fault tolerance with redundant paths
[ ] Dynamic routing based on load
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments
Hugging Face Transformers (DistilGPT-2)
ZeroMQ for messaging
PyTorch for ML framework
Made with ❤️ by malcommathela
