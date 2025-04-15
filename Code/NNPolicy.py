import torch
import torch.nn as nn
import torch.nn.functional as F
from Graph import Graph, DemandNode
from FulfillmentOptimization import Inventory, PolicyFulfillment, BalanceFulfillment
from Demand import Sequence, TemporalIndependenceGenerator, RWGenerator
import torch.optim as optim
import numpy as np
import random
from typing import List
import nevergrad as ng
import time
from concurrent.futures import ProcessPoolExecutor
np.random.seed(42)
random.seed(42)

class OnlineMatchingPolicy(nn.Module):
    """
    Neural network policy for online bipartite matching.
    Uses a feedforward network to select a supply node (or no-match) for each demand.
    """
    def __init__(self, graph: Graph, hidden_size=64, demand_embedding=False):
        super().__init__()
        self.graph = graph
        # Number of supply and demand nodes from the Graph
        self.num_supply = len(graph.supply_nodes)
        self.num_demand = len(graph.demand_nodes)
        # Determine ordering of supply nodes for consistent indexing (as used in inventory_state tuples)
        # Here we use the order of keys in graph.supply_nodes (assumed consistent with inventory_state tuple ordering)
        self.supply_order = list(graph.supply_nodes.keys())
        # Map supply node ID to index in the state vector
        self.supply_to_index = {sup_id: idx for idx, sup_id in enumerate(self.supply_order)}
        self.memorized_getitem = {}
        # If using one-hot demand encoding, input_dim will include num_demand; if using embedding, use embed vector size instead
        if demand_embedding:
            # Use an embedding layer for demand IDs to reduce dimensionality
            embed_dim = min(32, self.num_demand)  # for example, embed size (can be tuned)
            self.demand_embed = nn.Embedding(self.num_demand, embed_dim)
            demand_feat_dim = embed_dim
        else:
            self.demand_embed = None
            demand_feat_dim = self.num_demand  # one-hot length
        
        # Calculate input dimension: 
        # inventory vector (num_supply) + edge reward vector (num_supply) + demand features + 1 (time)
        self.state_dim = self.num_supply + self.num_supply + demand_feat_dim + 1
        # Define the feedforward network layers
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer: one logit per supply option plus one for "no match"
        self.out = nn.Linear(hidden_size, self.num_supply + 1)
        # Initialize weights (optional: e.g., Xavier initialization for stability)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, state_tensor):
        """Forward pass: compute action logits from state tensor."""
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        logits = self.out(x)  # shape: (batch, num_supply+1)
        return logits

    def _build_state_tensor(self, inventory_state, demand_id, t_norm):
        """
        Helper to construct the state tensor from inventory tuple, demand id, and normalized time.
        - inventory_state: tuple or list of inventory counts for each supply (in self.supply_order order)
        - demand_id: current demand node ID
        - t_norm: normalized time (e.g., t / T or t index scaled between 0 and 1)
        """
        # Inventory levels (as float tensor)
        inventory_vec = torch.tensor(inventory_state, dtype=torch.float32)
        # Edge reward vector for current demand: for each supply, reward if edge exists, else 0
        reward_vec = torch.zeros(self.num_supply, dtype=torch.float32)
        # Fill reward vector using graph edges (neighbor rewards)
        if demand_id in self.graph.demand_nodes:
            for sup_id in self.graph.demand_nodes[demand_id].neighbors:
                # Only consider reward if this supply can fulfill the demand
                if (sup_id, demand_id) in self.graph.edges:
                    idx = self.supply_to_index[sup_id]
                    reward_val = self.graph.edges[(sup_id, demand_id)].reward
                    reward_vec[idx] = reward_val
        # Demand features: either one-hot or embedding
        if self.demand_embed is None:
            # One-hot encoding for demand
            demand_one_hot = torch.zeros(self.num_demand, dtype=torch.float32)
            if demand_id < self.num_demand:
                demand_one_hot[demand_id] = 1.0
            demand_feat = demand_one_hot
        else:
            # Embedding for demand (demand_id should be an integer index 0..num_demand-1)
            demand_feat = self.demand_embed(torch.tensor(demand_id, dtype=torch.long))
            demand_feat = demand_feat.squeeze(0)  # get rid of batch dim if any
        # Time feature: use normalized time
        time_feat = torch.tensor([t_norm], dtype=torch.float32)
        # Concatenate all parts into one state vector
        state_vec = torch.cat([inventory_vec, reward_vec, demand_feat, time_feat], dim=0)
        return state_vec

    def __getitem__(self, key):
        """
        Allows the policy to be called like policy[inventory_state, t, demand_id] for integration with PolicyFulfillment.
        """
        inventory_state, t, demand_id = key
        
        if key in self.memorized_getitem:
            return self.memorized_getitem[key]
        else:
            # Normalize time if possible (if sequence length known, use t/T; otherwise use a simple scale)
            t_norm = t  # default: use raw t if no normalization
            # If the demand sequence length is known (e.g., stored in the policy or accessible via demand_id), we could normalize.
            # For simplicity, assume t is small or already a fraction (this can be adjusted based on use case).
            state_tensor = self._build_state_tensor(inventory_state, demand_id, t_norm)
            # Forward pass to get logits, then choose the best action (greedy) for inference
            logits = self.forward(state_tensor)
            # Mask out invalid actions (supply with no inventory or not neighbor)
            for idx, sup_id in enumerate(self.supply_order):
                # If no inventory or no edge from sup->demand, set logit to large negative
                if inventory_state[idx] <= 0 or (sup_id, demand_id) not in self.graph.edges:
                    logits[idx] = -1e9  # effectively zero probability after softmax
            action_index = int(torch.argmax(logits).item())  # index of highest logit
            if action_index == self.num_supply:
                # The last index corresponds to "no match"
                self.memorized_getitem[key] = -1
                return -1  # no supply chosen
            else:
                # Map index back to actual supply node ID
                supply_id = self.supply_order[action_index]
                self.memorized_getitem[key] = supply_id
                return supply_id

    def set_flat_params(self, flat_params: np.ndarray):
        """
        Sets the model's parameters from a flat numpy array.
        """
        pointer = 0
        for param in self.parameters():
            numel = param.numel()
            # Extract this slice from the flat array and reshape
            param_slice = flat_params[pointer:pointer + numel]
            param_tensor = torch.from_numpy(param_slice).view_as(param)
            # Set the parameter in-place
            with torch.no_grad():
                param.copy_(param_tensor)
            pointer += numel

def evaluate_policy_with_params(x, policy: OnlineMatchingPolicy, training_sequences, initial_inventory):
    """
    Evaluate the average total reward of the policy given a flattened parameter vector `x`.
    Used by Nevergrad for black-box optimization.
    """
    # Load the flat parameter vector into the policy
    policy.set_flat_params(x)

    policy.eval()
    with torch.no_grad():
        total_reward = 0.0
        for seq in training_sequences:
            # Reinitialize inventory for each sequence
            if hasattr(initial_inventory, "initial_inventory"):
                init_inv_dict = initial_inventory.initial_inventory
            else:
                init_inv_dict = initial_inventory
            inv_tensor = torch.tensor([init_inv_dict[sup_id] for sup_id in policy.supply_order], dtype=torch.float32)

            seq_reward = 0.0
            for t, request in enumerate(seq.requests):
                demand_id = request.demand_node
                t_norm = t / float(len(seq.requests))

                state = policy._build_state_tensor(tuple(inv_tensor.tolist()), demand_id, t_norm)
                logits = policy(state)

                # Mask out invalid actions
                for idx, sup_id in enumerate(policy.supply_order):
                    if inv_tensor[idx] <= 0 or (sup_id, demand_id) not in policy.graph.edges:
                        logits[idx] = -1e9

                action = torch.argmax(logits).item()

                if action >= len(policy.supply_order):  # "no match" dummy action
                    reward = 0.0
                else:
                    sup_id = policy.supply_order[action]
                    if (sup_id, demand_id) in policy.graph.edges:
                        reward = policy.graph.edges[(sup_id, demand_id)].reward
                        inv_tensor[action] -= 1
                        inv_tensor = inv_tensor.clamp(min=0)
                    else:
                        reward = 0.0

                seq_reward += reward
            total_reward += seq_reward

        avg_reward = total_reward / len(training_sequences)
        return -avg_reward  # Nevergrad minimizes, so we negate

def train_policy(policy: OnlineMatchingPolicy, training_sequences: List[Sequence], initial_inventory: Inventory, epochs=100, lr=0.01, tau=1.0, beta = 0.01):
    """
    Train the policy network on a set of demand sequences to maximize reward.
    :param policy: OnlineMatchingPolicy (nn.Module) to train.
    :param training_sequences: list of Demand.Sequence objects for training.
    :param initial_inventory: FulfillmentOptimization.Inventory object or dict with initial inventory counts per supply.
    :param epochs: number of training epochs.
    :param lr: learning rate for optimizer.
    :param tau: Gumbel-Softmax temperature parameter.
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    # Set seeds for reproducibility (feel free to adjust seed values)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    policy.train()  # set model to training mode
    
    # If initial_inventory is an Inventory object, extract the inventory dict
    if hasattr(initial_inventory, "initial_inventory"):
        init_inv_dict = initial_inventory.initial_inventory
    else:
        init_inv_dict = initial_inventory  # assume it's already a dict
    
    for epoch in range(1, epochs+1):
        
        total_reward_accum = 0.0  # accumulate total rewards to compute average
        for seq in training_sequences:
            # Make a fresh copy of the inventory for this sequence
            current_inventory = init_inv_dict.copy()
            # Prepare a tensor for inventory (will be updated in place as a tensor)
            inv_tensor = torch.tensor([current_inventory[sup_id] for sup_id in policy.supply_order], dtype=torch.float32)
            # Initialize total reward for this sequence
            
            rewards = []
            # Iterate through the demand requests in the sequence
            for t, request in enumerate(seq.requests):
                demand_id = request.demand_node
                # Calculate normalized time feature (if sequence length is known, use t/len; else use t or t/(len_seq))
                seq_length = seq.length if hasattr(seq, "length") else len(seq.requests)
                t_norm = t / float(seq_length)  # normalize t to [0,1)
                # Build state tensor for current step
                state = policy._build_state_tensor(tuple(inv_tensor.detach().tolist()), demand_id, t_norm)
                # Forward pass to get logits
                logits = policy.forward(state)
                # Mask out invalid actions (supply with no inventory or not neighbor)
                for idx, sup_id in enumerate(policy.supply_order):
                    # If no inventory or no edge from sup->demand, set logit to large negative
                    if inv_tensor[idx] <= 0 or (sup_id, demand_id) not in policy.graph.edges:
                        logits[idx] = -1e9  # effectively zero probability after softmax
                # Gumbel-Softmax sampling to get one-hot action (differentiable)
                action_onehot = F.gumbel_softmax(logits, tau=tau, hard=True)
                current_tau = max(tau * np.exp(-0.05 * epoch), 0.1)
                # action_probs = F.gumbel_softmax(logits, tau=current_tau, hard=False)
                # Compute immediate reward: dot product of action one-hot with reward vector
                # Build reward vector (length num_supply+1) for this demand
                reward_values = [0.0] * policy.num_supply
                if demand_id in policy.graph.demand_nodes:
                    for sup_id in policy.graph.demand_nodes[demand_id].neighbors:
                        if (sup_id, demand_id) in policy.graph.edges:
                            idx = policy.supply_to_index[sup_id]
                            reward_values[idx] = policy.graph.edges[(sup_id, demand_id)].reward
                reward_values.append(0.0)  # reward for "no match" action is 0
                reward_vec = torch.tensor(reward_values, dtype=torch.float32, device=logits.device)
                immediate_reward = torch.dot(action_onehot, reward_vec)
                # immediate_reward = torch.dot(action_probs, reward_vec)
                rewards.append(immediate_reward)
                # Update total reward for this sequence (accumulate; this is a torch tensor tracking grad)

                # Update inventory: subtract 1 from chosen supply if any (skip action has no effect)
                # The action_onehot vector has length num_supply+1; the last element corresponds to no-match
                
                inv_tensor = inv_tensor.detach()
                inv_tensor -= action_onehot[:-1]  # skip last index (no-match)
                inv_tensor = inv_tensor.clamp(min=0)
                
                
                
                # inv_tensor = inv_tensor - action_probs[:-1].detach()
                # inv_tensor.clamp_(min=0)
                # print(f'Step {t},demand {demand_id},  action: {action_onehot.tolist()}, reward {immediate_reward:.2f}, inventory: {inv_tensor}')
            
            # End of sequence: accumulate reward (as a Python float for logging) and compute loss
           
            seq_reward = torch.stack(rewards).sum()
            total_reward_accum += seq_reward.item()
            # action_probs_clipped = torch.clamp(action_probs, min=1e-8)
            # entropy = -(action_probs_clipped * action_probs_clipped.log()).sum()
            loss = -seq_reward #+ beta * entropy # negative total reward (we want to maximize reward)
            # Backpropagate for this sequence
            optimizer.zero_grad()
            loss.backward()
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item()}")
            optimizer.step()
        # Compute and print average reward for this epoch (optional for monitoring)
        avg_reward = total_reward_accum / len(training_sequences)
        
        if epoch%1==0:
            print(f"Epoch {epoch}: average total reward = {avg_reward:.3f}")
        


def create_and_train_policy_ng( graph, initial_inventory,training_sequences):
    
    policy = OnlineMatchingPolicy(graph)
    initial_params = torch.cat([p.data.flatten() for p in policy.parameters()]).numpy()

    param_dim = len(initial_params)
    instrumentation = ng.p.Array(shape=(param_dim,)).set_bounds(-6, 6)
    # optimizer = ng.optimizers.OnePlusOne(parametrization=instrumentation, budget=200)
    optimizer = ng.optimizers.NGOpt(parametrization=instrumentation, budget=400)

    print('Training')
    start = time.time()

    recommendation = optimizer.minimize(lambda x: evaluate_policy_with_params(x, policy, training_sequences, initial_inventory))

    best_params = recommendation.value
    policy.set_flat_params(best_params)  # <- APPLY the optimized params to the policy

    print(f"Training time with {len(training_sequences)} samples: {time.time() - start:.2f} seconds")
    # print("Example logits:", policy(torch.randn(policy.state_dim)))  # should not be all zeros or NaNs

    return policy

if __name__ == '__main__':
    
    supply_nodes, demand_nodes, rewards = (f'three_node_graph_supplynodes.csv', f'three_node_graph_demandnodes.csv', f'three_node_graph_rewards.csv')
    ### Graph Reading
    print('Reading graph')
    graph = Graph(supply_nodes, demand_nodes, rewards)
    T = 15
    n_train = 10
    
    n_test = 1000
    
    p = {}
    p[0] = [1/3, 1/3, 1/6, 1/6]
    p[1] = [1/3, 1/3, 1/6, 1/6]
    p[2] = [1/3, 1/3, 1/6, 1/6]
    p[3] = [1/3, 1/3, 1/6, 1/6]
    p[4] = [1/3, 1/3, 1/6, 1/6]
    p[5] = [5/12, 5/12, 1/12, 1/12]
    p[6] = [5/12, 5/12, 1/12, 1/12]
    p[7] = [5/12, 5/12, 1/12, 1/12]
    p[8] = [5/12, 5/12, 1/12, 1/12]
    p[9] = [5/12, 5/12, 1/12, 1/12]
    p[10] = [0, 0, 1/2, 1/2]
    p[11] = [0, 0, 1/2, 1/2]
    p[12] = [0, 0, 1/2, 1/2]
    p[13] = [0, 0, 1/2, 1/2]
    p[14] = [0, 0, 1/2, 1/2]
    initial_inventory = Inventory({0:5, 1:5, 2:5})
    
    
    demand_node_list = sorted([demand_node_id for demand_node_id in graph.demand_nodes])
    train_generator = TemporalIndependenceGenerator(demand_node_list,p, seed = 111)
    train_generator = RWGenerator(mean = 15, demand_nodes=demand_node_list, seed = 111)
    
    
    
    training_sequences = [train_generator.generate_sequence() for _ in range(n_train)]
    
    
    policy = create_and_train_policy_ng(graph, initial_inventory, training_sequences)
    
    # reward = evaluate_policy_with_params(best_params, policy, training_sequences, initial_inventory)
    # print(reward)
    
    test_generator = TemporalIndependenceGenerator(demand_node_list,p, seed = 121)
    test_generator = RWGenerator(mean = 15, demand_nodes=demand_node_list, seed = 121)
    test_sequences = [test_generator.generate_sequence() for _ in range(n_test)]
    
    
    policy_fulfill = PolicyFulfillment(graph)
    balance_fulfill = BalanceFulfillment(graph)
    print('Testing')
    pol = 0
    bal = 0
    # Use the trained policy to fulfill the test sequence
    for test_sequence in test_sequences:
        number_fulfillments, pol_reward, lost_sales = policy_fulfill.fulfill(test_sequence, initial_inventory, policy)
        pol += pol_reward/n_test
        number_fulfillments, bal_reward, lost_sales = balance_fulfill.fulfill(test_sequence, initial_inventory)
        bal += bal_reward/n_test
        
    print(f'policy: {pol}')
    print(f'balance: {bal}')