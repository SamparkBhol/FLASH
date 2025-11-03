import torch
import numpy as np
import torch.nn.functional as F

import src.config as config

def flatten_weights(update):
    vectors = []
    for key in sorted(update.keys()):
        if torch.is_tensor(update[key]):
            vectors.append(update[key].view(-1))
    return torch.cat(vectors)

class FlashImmuneValidator:
    def __init__(self, num_clients):
        self.reputation_scores = {
            i: config.INITIAL_REPUTATION for i in range(num_clients)
        }
        self.historical_contributions = {i: 0.5 for i in range(num_clients)}
        self.consensus_agreements = {i: 0.5 for i in range(num_clients)}
        self.quarantined_nodes = set()

    def simulate_zkp(self, client_id, update_delta):
        print(f"      [ZKP-SIM] Verifying ZKP stub for Client {client_id}... ", end="")
        
        is_valid = True
        
        if torch.isnan(flatten_weights(update_delta)).any():
            is_valid = False
            
        print("OK." if is_valid else "FAILED (NaN detected).")
        return is_valid

    def calculate_update_norm(self, update_delta):
        try:
            flat_vector = flatten_weights(update_delta).float()
            return torch.norm(flat_vector, p=2).item()
        except Exception:
            return float('inf')

    def calculate_consensus_score(self, update_norm, median_norm):
        if median_norm == 0:
            return 0.5

        deviation_percent = (abs(update_norm - median_norm) * 100) / (median_norm + 1e-6)
        
        if deviation_percent > config.MAX_CONSENSUS_DEVIATION:
            return 0.0
        
        score = 1.0 - (deviation_percent / config.MAX_CONSENSUS_DEVIATION)
        return max(0.0, score)

    def validate_and_update_reputations(self, update_deltas):
        validation_results = {}
        client_norms = {}
        
        for client_id, delta in update_deltas.items():
            if not delta:
                client_norms[client_id] = float('inf')
                continue
            client_norms[client_id] = self.calculate_update_norm(delta)

        valid_norms = [n for n in client_norms.values() if n != float('inf')]
        if not valid_norms:
            median_norm = 0
        else:
            median_norm = np.median(valid_norms)

        for client_id, delta in update_deltas.items():
            if not delta:
                validation_results[client_id] = {'valid': False, 'C_nt': 0.0, 'V_nt': 0.0}
                continue

            zkp_valid = self.simulate_zkp(client_id, delta)
            V_nt = 1.0 if zkp_valid else 0.0
            
            update_norm = client_norms[client_id]
            C_nt = self.calculate_consensus_score(update_norm, median_norm)

            is_valid = zkp_valid and (C_nt > 0.0)
            if not is_valid:
                V_nt = 0.0
            
            validation_results[client_id] = {'valid': is_valid, 'C_nt': C_nt, 'V_nt': V_nt}

        self.update_reputations_batch(validation_results)
        return validation_results

    def update_reputations_batch(self, validation_results):
        for client_id, result in validation_results.items():
            if client_id in self.quarantined_nodes:
                continue
                
            V_nt = result['V_nt']
            C_nt = max(0, result['C_nt'])
            
            H_nt_minus_1 = self.historical_contributions[client_id]

            R_nt = (config.REPUTATION_ALPHA * H_nt_minus_1) + \
                   (config.REPUTATION_BETA * C_nt) + \
                   (config.REPUTATION_GAMMA * V_nt)
            
            self.reputation_scores[client_id] = max(0, min(1, R_nt))
            
            self.historical_contributions[client_id] = (H_nt_minus_1 + R_nt) / 2

            if self.reputation_scores[client_id] < config.REPUTATION_THRESHOLD:
                if client_id not in self.quarantined_nodes:
                    print(f"      [IMMUNE] Client {client_id} REPUTATION ({self.reputation_scores[client_id]:.2f}) below threshold. QUARANTINED.")
                    self.quarantined_nodes.add(client_id)
            elif client_id in self.quarantined_nodes:
                 print(f"      [IMMUNE] Client {client_id} REPUTATION ({self.reputation_scores[client_id]:.2f}) recovered. Restored.")
                 self.quarantined_nodes.remove(client_id)

    def get_trusted_clients(self):
        return [cid for cid in self.reputation_scores.keys() if cid not in self.quarantined_nodes]