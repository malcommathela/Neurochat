#!/bin/bash
echo "Setting up virtual network topology..."

# Clean up existing
sudo ip netns del layer1 2>/dev/null
sudo ip netns del layer2 2>/dev/null
sudo ip netns del layer3 2>/dev/null

# Create namespaces
sudo ip netns add layer1
sudo ip netns add layer2
sudo ip netns add layer3

# Create links
sudo ip link add veth1-2 type veth peer name veth2-1
sudo ip link set veth2-1 netns layer2
sudo ip link set veth1-2 netns layer1

sudo ip link add veth2-3 type veth peer name veth3-2
sudo ip link set veth3-2 netns layer3
sudo ip link set veth2-3 netns layer2

sudo ip link add veth1-host type veth peer name veth-host-1
sudo ip link add veth3-host type veth peer name veth-host-3
sudo ip link set veth1-host netns layer1
sudo ip link set veth3-host netns layer3

# Configure IPs
sudo ip netns exec layer1 ip addr add 10.0.1.1/24 dev veth1-2
sudo ip netns exec layer1 ip addr add 10.0.0.1/24 dev veth1-host
sudo ip netns exec layer1 ip link set veth1-2 up
sudo ip netns exec layer1 ip link set veth1-host up
sudo ip netns exec layer1 ip link set lo up

sudo ip netns exec layer2 ip addr add 10.0.1.2/24 dev veth2-1
sudo ip netns exec layer2 ip addr add 10.0.2.2/24 dev veth2-3
sudo ip netns exec layer2 ip link set veth2-1 up
sudo ip netns exec layer2 ip link set veth2-3 up
sudo ip netns exec layer2 ip link set lo up

sudo ip netns exec layer3 ip addr add 10.0.2.3/24 dev veth3-2
sudo ip netns exec layer3 ip addr add 10.0.0.3/24 dev veth3-host
sudo ip netns exec layer3 ip link set veth3-2 up
sudo ip netns exec layer3 ip link set veth3-host up
sudo ip netns exec layer3 ip link set lo up

# Add latency
sudo ip netns exec layer1 tc qdisc add dev veth1-2 root netem delay 2ms
sudo ip netns exec layer2 tc qdisc add dev veth2-3 root netem delay 3ms

# Setup filesystems
sudo mkdir -p /var/netns/layer1/var/tmp
sudo mkdir -p /var/netns/layer2/var/tmp
sudo mkdir -p /var/netns/layer3/var/tmp

echo "Network ready!"
echo "Layer1: 10.0.0.1 (host), 10.0.1.1 (internal)"
echo "Layer2: 10.0.1.2, 10.0.2.2"
echo "Layer3: 10.0.2.3, 10.0.0.3 (host)"
