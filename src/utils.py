import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import turtle
import time

import src.config as config

def evaluate_model(model, data_loader):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def get_risk_score(probability):
    if probability > 0.8:
        return "High"
    elif probability > 0.5:
        return "Medium"
    else:
        return "Low"

def plot_simulation_results(results, clients):
    num_rounds = len(results['accuracy'])
    rounds = range(num_rounds)
    malicious_ids = [c.client_id for c in clients if c.is_malicious]
    benign_ids = [c.client_id for c in clients if not c.is_malicious]

    fig, axs = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('FLASH-IMMUNE Simulation Results', fontsize=16)

    axs[0, 0].plot(rounds, results['accuracy'], label='Global Accuracy')
    axs[0, 0].set_title('Global Model Accuracy')
    axs[0, 0].set_xlabel('Federated Round')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(rounds, results['loss'], label='Global Loss', color='orange')
    axs[0, 1].set_title('Global Model Loss')
    axs[0, 1].set_xlabel('Federated Round')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(rounds, results['malicious_updates_rejected'], label='Malicious Rejected', color='red')
    axs[1, 0].plot(rounds, results['benign_updates_rejected'], label='Benign Rejected', color='green')
    axs[1, 0].set_title('Update Rejection per Round')
    axs[1, 0].set_xlabel('Federated Round')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    reputations = np.array(results['reputations'])
    for i in benign_ids:
        axs[1, 1].plot(rounds, reputations[:, i], color='blue', alpha=0.3)
    for i in malicious_ids:
        axs[1, 1].plot(rounds, reputations[:, i], color='red', alpha=0.3)
    
    axs[1, 1].plot([], [], color='blue', label='Benign Clients')
    axs[1, 1].plot([], [], color='red', label='Malicious Clients')
    axs[1, 1].set_title('Client Reputation Over Time')
    axs[1, 1].set_xlabel('Federated Round')
    axs[1, 1].set_ylabel('Reputation Score')
    axs[1, 1].set_ylim(0, 1.1)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[2, 0].plot(rounds, results['privacy_budget'], label='Privacy Budget (Epsilon)', color='purple')
    axs[2, 0].set_title('Simulated Privacy Budget Consumption')
    axs[2, 0].set_xlabel('Federated Round')
    axs[2, 0].set_ylabel('Epsilon')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("flash_immune_simulation_results.png")
    print("\nResults plot saved to 'flash_immune_simulation_results.png'")

def plot_network_topology(clients, reputations):
    plt.figure(figsize=(12, 12))
    G = nx.Graph()
    
    G.add_node("Global Aggregator", layer=0, size=3000)
    
    for i in range(1, config.NUM_SIDECHAINS + 1):
        chain_node = f"Sidechain {i}"
        G.add_node(chain_node, layer=1, size=1500)
        G.add_edge("Global Aggregator", chain_node, type='ibc')

    client_map = {c.client_id: c for c in clients}

    for client_id in range(config.NUM_CLIENTS):
        client = client_map.get(client_id)
        node_id = f"Client {client_id}"
        G.add_node(node_id, layer=2, size=500)
        if client and client.data_loader:
            G.add_edge(f"Sidechain {client.sidechain_id}", node_id, type='fl')

    colors = []
    for node in G.nodes():
        if node == "Global Aggregator":
            colors.append('skyblue')
        elif "Sidechain" in node:
            colors.append('lightgray')
        else:
            client_id = int(node.split(" ")[-1])
            client = client_map.get(client_id)
            rep = reputations.get(client_id, 0)
            
            if rep < config.REPUTATION_THRESHOLD:
                colors.append('red')
            elif client and client.is_malicious:
                colors.append('orange')
            else:
                colors.append('green')

    pos = nx.multipartite_layout(G, subset_key="layer")
    sizes = [G.nodes[n]['size'] for n in G]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, font_size=8)
    
    plt.title("FLASH-IMMUNE Simulated Network Topology", size=15)
    plt.savefig("flash_network_topology.png")
    print("Network topology plot saved to 'flash_network_topology.png'")

def draw_final_splash(final_accuracy, malicious_rejected, total_malicious):
    try:
        wn = turtle.Screen()
        wn.setup(width=800, height=400)
        wn.bgcolor("black")
        wn.title("FLASH-IMMUNE :: Simulation Complete")

        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        
        t.color("#00FF00")
        t.penup()
        t.goto(-350, 150)
        t.pendown()
        t.pensize(2)
        for _ in range(2):
            t.forward(700)
            t.right(90)
            t.forward(300)
            t.right(90)

        t.penup()
        t.goto(0, 50)
        t.color("#00FFFF")
        t.write("FLASH-IMMUNE", align="center", font=("Consolas", 40, "bold"))

        t.penup()
        t.goto(0, 0)
        t.color("#00FF00")
        t.write(":: VALIDATION COMPLETE :: SYSTEM SECURE ::", align="center", font=("Consolas", 20, "normal"))

        reject_rate = (malicious_rejected / total_malicious) * 100 if total_malicious > 0 else 100
        
        t.penup()
        t.goto(-300, -80)
        t.color("white")
        t.write(f"Final Accuracy: {final_accuracy:.2f}%", align="left", font=("Consolas", 16, "normal"))
        
        t.penup()
        t.goto(-300, -110)
        t.color("red")
        t.write(f"Malicious Updates Rejected: {malicious_rejected} ({reject_rate:.1f}%)", align="left", font=("Consolas", 16, "normal"))

        t.penup()
        t.goto(270, -130)
        t.color("gray")
        t.write("Click to exit", align="center", font=("Consolas", 10, "normal"))
        
        wn.exitonclick()
        
    except Exception as e:
        print(f"\n[INFO] Could not launch turtle graphics window: {e}")
        print("This is common in restricted environments (e.g., SSH, Docker without display).")
        print("Please view the .png files for results.")