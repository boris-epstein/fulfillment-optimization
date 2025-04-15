import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict
from typing import List
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from Demand import Sequence, TemporalIndependenceGenerator, RWGenerator
from Graph import Graph
from FulfillmentOptimization import PolicyFulfillment, Inventory, BalanceFulfillment


def convert_sequence_to_policy_input(sequence):
    """
    Converts a Sequence object into a list of (time_step, demand_node_id)
    """
    return [(t, request.demand_node) for t, request in enumerate(sequence.requests)]

def extract_reward_matrix(graph):
    """
    Extracts the reward matrix from a Graph object as a PyTorch tensor.
    """
    supply_nodes = sorted(graph.supply_nodes.keys())  # Get sorted supply node indices
    demand_nodes = sorted(graph.demand_nodes.keys())  # Get sorted demand node indices

    reward_matrix = torch.zeros(len(supply_nodes), len(demand_nodes))

    for (supply_id, demand_id), edge in graph.edges.items():
        supply_index = supply_nodes.index(supply_id)  # Convert node ID to matrix index
        demand_index = demand_nodes.index(demand_id)
        reward_matrix[supply_index, demand_index] = edge.reward

    return reward_matrix

def train_policy(graph,sequences: List[Sequence], reward_matrix, num_epochs=300, gamma=1.0,learning_rate=0.005, T=10):
    """
    Trains the policy using Monte-Carlo policy optimization with neighbor masking.
    """
    num_offline_nodes = len(graph.supply_nodes)
    num_online_nodes = len(graph.demand_nodes)

    num_sequences = len(sequences)
    # Initialize policy model
    policy = SoftmaxMatchingPolicy(num_offline_nodes=num_offline_nodes,
                                   inventory_dim=num_offline_nodes,
                                   online_node_dim=num_online_nodes)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        exploration_weight = max(0.05, 0.5 * (1 - epoch / num_epochs))  # Anneal over time

        for sequence in sequences:
            inventory_state = torch.full((num_offline_nodes,), fill_value=5).float()
            rewards = []
            log_probs = []
            entropies = []  # Store entropy from distributions
            # sequence = Sequence([...])  # Generate a random sequence or use your dataset

            for t, request in enumerate(sequence.requests):
                demand_node_id = request.demand_node
                time_remaining = torch.tensor(T - t).float().detach()

                # Get valid neighbors from graph
                valid_neighbors = list(graph.demand_nodes[demand_node_id].neighbors)
                neighbor_mask = torch.zeros(num_offline_nodes)
                for supply_node in valid_neighbors:
                    if inventory_state[supply_node] > 0:
                        neighbor_mask[supply_node] = 1

                # Convert demand node ID to feature vector (one-hot encoding)
                online_node_features = torch.zeros(num_online_nodes)
                online_node_features[demand_node_id] = 1

                # Select action
                action, log_prob, dist = policy.select_action(inventory_state, online_node_features, time_remaining, neighbor_mask, deterministic=False)
                entropies.append(dist.entropy())  # Store entropy
                # Ensure action is valid
                action = int(action)
                if action not in valid_neighbors or inventory_state[action] == 0:
                    action = -1

                # Compute reward
                reward = 0
                if action != -1:
                    reward = graph.edges[action, demand_node_id].reward
                    inventory_state[action] -= 1

                rewards.append(reward)
                log_probs.append(log_prob)

            # Compute total collected reward
            total_reward = sum(rewards)
            # total_reward = (total_reward - torch.mean(total_reward)) / (torch.std(total_reward) + 1e-8)

            # Policy loss (negative log-prob * reward)
            entropy_bonus = -exploration_weight * sum(entropies)  # Stronger exploration bonus
            policy_loss = -sum(log_probs) * total_reward + entropy_bonus
            total_loss += policy_loss

        total_loss /= num_sequences
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

    print("Training complete!")
    return policy

def train_depletion_policy(graph, sequences: List[Sequence], reward_matrix, num_epochs=300, gamma=1.0, learning_rate=0.005, T=10):
    """
    Trains the depletion-aware policy using Monte-Carlo policy optimization.
    """
    num_offline_nodes = len(graph.supply_nodes)
    
    
    # Initialize the policy model
    policy = DepletionAwarePolicy(num_offline_nodes)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    num_sequences = len(sequences)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)

        total_loss = 0

        for sequence in sequences:
            inventory_state = torch.full((num_offline_nodes,), fill_value=5).float()
            initial_inventory = inventory_state.clone()
            rewards = []
            log_probs = []

            for t, request in enumerate(sequence.requests):
                demand_node_id = request.demand_node  # Current online node arriving
                

                # Compute valid neighbors

                valid_neighbors = list(graph.demand_nodes[demand_node_id].neighbors)
                neighbor_mask = torch.zeros(num_offline_nodes)
                for supply_node in valid_neighbors:
                    if inventory_state[supply_node] > 0:
                        neighbor_mask[supply_node] = 1

                # Get reward matrix (supply x demand)
                
                
                action, log_prob = policy.select_action(
                    reward_matrix, inventory_state, initial_inventory, neighbor_mask, demand_node_id, deterministic=True
                )

                action = int(action)
                if action not in valid_neighbors or inventory_state[action] <= 0:
                    action = -1
                    
                reward = 0
                if action != -1:
                    reward = graph.edges[action, demand_node_id].reward
                    inventory_state[action] -= 1
                    inventory_state = inventory_state.detach()
                    log_probs.append(log_prob)
                

                    
                rewards.append(reward)
                

            # Normalize rewards
            total_reward = torch.tensor(rewards, dtype=torch.float32)  # ✅ Convert list to Tensor
            if torch.std(total_reward) > 0:
                total_reward = (total_reward - torch.mean(total_reward)) / (torch.std(total_reward) + 1e-8)  # ✅ Normalize
            total_reward = total_reward.sum()  # ✅ Convert back to scalar

            # Compute policy loss (negative log-probability * reward)
            policy_loss = -sum(log_probs) * total_reward *100000
            total_loss += policy_loss

        total_loss /= num_sequences
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.5f}")
            print(f"Current theta vaues: {policy.theta.detach().numpy()}")
            
    print(f"Final theta values: {policy.theta.detach().numpy()}")
    print("Training complete!")
    return policy

def train_depletion_policy_black_box(graph, sequences, reward_matrix, T=10):
    num_offline_nodes = len(graph.supply_nodes)

    # Objective: maximize average total reward → minimize (-reward)
    def objective(theta_np):
        theta_tensor = torch.tensor(theta_np, dtype=torch.float32)

        policy = DepletionAwarePolicy(num_offline_nodes)
        with torch.no_grad():
            policy.theta.copy_(theta_tensor)

        total_reward = 0.0
        for sequence in sequences:
            inventory_state = torch.full((num_offline_nodes,), fill_value=5).float()
            initial_inventory = inventory_state.clone()

            for t, request in enumerate(sequence.requests):
                demand_node_id = request.demand_node
                valid_neighbors = list(graph.demand_nodes[demand_node_id].neighbors)
                neighbor_mask = torch.zeros(num_offline_nodes)
                for supply_node in valid_neighbors:
                    if inventory_state[supply_node] > 0:
                        neighbor_mask[supply_node] = 1

                if neighbor_mask.sum() == 0:
                    continue

                action, _ = policy.select_action(
                    reward_matrix, inventory_state, initial_inventory, neighbor_mask,
                    demand_node_id, deterministic=True
                )
                action = int(action)

                if action not in valid_neighbors or inventory_state[action] <= 0:
                    continue

                reward = graph.edges[action, demand_node_id].reward
                total_reward += reward
                inventory_state[action] -= 1

        avg_reward = total_reward / len(sequences)
        return -avg_reward  # Negative since we want to maximize

    # Initial guess: Balance heuristic (theta = 1)
    theta0 = np.ones(num_offline_nodes)

    # Run Nelder-Mead optimization
    # result = minimize(objective, theta0, method='Nelder-Mead', options={'maxiter': 200, 'disp': True})
    bounds = [(0.1, 2.0)] * num_offline_nodes
    result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=200, disp=True)
    # Build final policy with best theta
    policy = DepletionAwarePolicy(num_offline_nodes)
    with torch.no_grad():
        policy.theta.copy_(torch.tensor(result.x, dtype=torch.float32))

    print("Final optimized theta:", result.x)
    return policy


def train_depletion_nn_policy(graph, sequences, reward_matrix, num_epochs=300, lr=0.01, hidden_dim=16, T=10):
    num_offline = len(graph.supply_nodes)
    num_online = len(graph.demand_nodes)
    input_dim = 4 + num_online  # inv, SF, time, reward + one-hot demand

    policy = DepletionNNPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for sequence in sequences:
            inventory = torch.full((num_offline,), 5.0)
            initial_inventory = inventory.clone()

            log_probs = []
            rewards = []

            for t, request in enumerate(sequence.requests):
                j = request.demand_node
                mask = torch.zeros(num_offline)
                for i in graph.demand_nodes[j].neighbors:
                    if inventory[i] > 0:
                        mask[i] = 1

                if mask.sum() == 0:
                    log_probs.append(torch.tensor(0.0, requires_grad=True))
                    rewards.append(0)
                    continue

                SF = (1 - inventory / (initial_inventory + 1e-8)).detach()
                inv = inventory.detach()
                time_remaining = torch.tensor([T - t] * num_offline).float()
                reward_vector = reward_matrix[:, j].float()
                demand_one_hot = torch.zeros(num_online)
                demand_one_hot[j] = 1
                demand_feat = demand_one_hot.repeat(num_offline, 1)

                features = torch.stack([inv, SF, time_remaining, reward_vector], dim=1)
                full_features = torch.cat([features, demand_feat], dim=1)

                logits = policy(full_features, mask)
                probs = F.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)

                a = action.item()
                reward = 0
                if inventory[a] > 0 and (a, j) in graph.edges:
                    reward = graph.edges[a, j].reward
                    inventory[a] -= 1
                    inventory = inventory.detach()

                rewards.append(reward)

            total_reward = sum(rewards)
            loss = -sum(log_probs) * total_reward  # ✅ directly optimizing expected reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    return policy
def simulate_policy(graph, sequence, inventory, wrapped_policy, reward_matrix):
    """
    Evaluates the trained policy on a given sequence using PolicyFulfillment.
    """
    # Wrap trained policy to make it subscriptable


    # Use PolicyFulfillment to evaluate the policy
    policy_fulfillment = PolicyFulfillment(graph)
    fulfillments, collected_rewards, lost_sales = policy_fulfillment.fulfill(sequence, inventory, wrapped_policy, verbose=False)

    # print(f"Simulation Results:")
    # print(f" - Total Reward: {collected_rewards}")
    # print(f" - Lost Sales: {lost_sales}")
    # print(f" - Fulfilled Requests: {sum(fulfillments.values())}")

    return fulfillments, collected_rewards, lost_sales



class DepletionAwarePolicy(nn.Module):
    def __init__(self, num_offline_nodes):
        super().__init__()
        self.num_offline = num_offline_nodes

        # Ensure theta is a trainable parameter, explicitly requiring gradients
        self.theta = nn.Parameter(torch.ones(num_offline_nodes, requires_grad=True))

    def forward(self, reward_matrix, inventory_state, initial_inventory, mask, demand_node_id):
        """
        Compute logits only for the specific online node arriving.
        """
        
        
        SF = 1.0 - (inventory_state.float() / (initial_inventory.float() + 1e-8))  # Avoid division by zero
        # print(SF)
        depletion_factor = 1 - torch.exp(self.theta * (SF - 1))
        logits = reward_matrix[:, demand_node_id] * depletion_factor
        
        # Apply mask
        if mask.sum() == 0:  # ✅ No feasible actions
            logits = torch.zeros_like(logits)  # Uniform logits → uniform softmax
        else:
            logits = logits - (1 - mask) * 1e9  # Mask infeasible actions

        probs = F.softmax(logits, dim=0)
        return probs



    def select_action(self, reward_matrix, inventory_state, initial_inventory, mask, demand_node_id, deterministic=False):
        """
        Selects an action based on computed probabilities, ensuring only feasible actions are chosen.
        """
        probs = self.forward(reward_matrix, inventory_state, initial_inventory, mask, demand_node_id)

        

        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs)  # Pick the best feasible action
        else:
            action = dist.sample()  # Sample from the valid distribution

        action = action.item()  # Convert tensor to integer

        # Ensure log_prob is correctly computed
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64))

        return action, log_prob


class SoftmaxMatchingPolicy(nn.Module):
    def __init__(self, num_offline_nodes, inventory_dim, online_node_dim, hidden_dim=16):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(inventory_dim + online_node_dim + 1, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),  # Dropout to reduce overfitting
        nn.Linear(hidden_dim, num_offline_nodes)
    )

    def forward(self, inventory_state, online_node_features, time_remaining, mask):
        state = torch.cat([inventory_state, online_node_features, time_remaining.unsqueeze(0)], dim=-1)
        logits = self.model(state)

        # Apply the neighbor mask
        masked_logits = logits - (1 - mask) * 1e9  # Invalid nodes get -inf probability
        return F.softmax(masked_logits, dim=-1)
# class SoftmaxMatchingPolicy(nn.Module):
#     """
#     Online bipartite matching policy using a softmax-based action selection.
#     Now also incorporates the arriving online node and time remaining.
#     """
#     def __init__(self, num_offline_nodes: int, inventory_dim: int, online_node_dim: int, hidden_model: nn.Module = None):
#         """
#         :param num_offline_nodes: Number of offline nodes (size of action space).
#         :param inventory_dim: Number of inventory-related state features (typically num_offline_nodes).
#         :param online_node_dim: Number of features used to describe the arriving online node.
#         :param hidden_model: Optional custom model for computing logits.
#         """
#         super().__init__()
#         self.num_offline = num_offline_nodes

#         # Total input dimension = inventory features + online node features + 1 (time remaining)
#         input_dim = inventory_dim + online_node_dim + 1  

#         # Use a simple linear model by default for interpretability.
#         if hidden_model is None:
#             self.model = nn.Linear(input_dim, num_offline_nodes)
#         else:
#             self.model = hidden_model

    

#     def forward(self, inventory_state: torch.Tensor, online_node_features: torch.Tensor, time_remaining: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         Compute action probabilities given inventory state, online node features, and time remaining.
#         :param inventory_state: Inventory levels for offline nodes.
#         :param online_node_features: Features representing the arriving online node.
#         :param time_remaining: Time steps remaining.
#         :param mask: A mask where 1 = valid (neighbor with inventory), 0 = invalid.
#         :return: Softmax probabilities over offline nodes.
#         """
#         state = torch.cat([inventory_state, online_node_features, time_remaining.unsqueeze(0)], dim=-1)
        
#         logits = self.model(state)  # Compute logits for all offline nodes
        
#         # **Apply mask: Only allow neighbors with inventory**
#         masked_logits = logits - (1 - mask) * 1e9  # Invalid nodes get -inf probability

#         return F.softmax(masked_logits, dim=-1)
    
    def action_distribution(self, inventory_state: torch.Tensor, online_node_features: torch.Tensor, time_remaining: torch.Tensor, mask: torch.Tensor = None) -> Categorical:
        """
        Get a Categorical action distribution based on state and online node.
        :return: A PyTorch Categorical distribution over offline nodes.
        """
        if inventory_state.dim() == 1:
            inventory_state = inventory_state.unsqueeze(0)
        if online_node_features.dim() == 1:
            online_node_features = online_node_features.unsqueeze(0)
        if time_remaining.dim() == 0:
            time_remaining = time_remaining.unsqueeze(0).unsqueeze(0)
        elif time_remaining.dim() == 1:
            time_remaining = time_remaining.unsqueeze(-1)

        logits = self.model(torch.cat([inventory_state, online_node_features, time_remaining], dim=-1))

        if mask is None:
            mask = (inventory_state > 0).float()
        else:
            mask = mask.float()

        masked_logits = logits - (1 - mask) * 1e9  # Mask out depleted nodes
        return Categorical(logits=masked_logits)


    
    def select_action(self, inventory_state: torch.Tensor, online_node_features: torch.Tensor, time_remaining: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        """
        Select an action given the state.
        :param mask: Neighbor mask (1 = valid neighbor, 0 = non-neighbor).
        """
        dist = self.action_distribution(inventory_state, online_node_features, time_remaining, mask)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)  # Choose highest probability
        else:
            action = dist.sample()  # Sample an action
        
        return int(action), dist.log_prob(action), dist

class SubscriptableDepletionPolicyWrapper:
    def __init__(self, policy, graph, reward_matrix, initial_inventory_tuple):
        """
        Wraps the trained policy to be subscriptable so it works with PolicyFulfillment.
        """
        self.policy = policy
        self.graph = graph
        self.reward_matrix = reward_matrix
        self.initial_inventory = torch.tensor(initial_inventory_tuple)

    def __getitem__(self, key):
        """
        Allows subscript notation policy[inventory_tuple, t, demand_node_id].
        :param key: (inventory_tuple, time_step, demand_node_id)
        :return: Chosen supply node or -1 if no valid choice.
        """
        inventory_tuple, time_step, demand_node_id = key

        # Convert inventory tuple to tensor
        inventory_state = torch.tensor(inventory_tuple).float()

        # Convert demand node ID into feature vector (one-hot encoding)
        online_node_features = torch.zeros(len(self.graph.demand_nodes))
        online_node_features[demand_node_id] = 1  # Assuming one-hot encoding

        # Compute availability mask
        valid_neighbors = list(self.graph.demand_nodes[demand_node_id].neighbors)
        
        
        valid_neighbors = list(self.graph.demand_nodes[demand_node_id].neighbors)
        mask = torch.zeros(len(self.graph.supply_nodes))
        for supply_node in valid_neighbors:
            if inventory_state[supply_node] > 0:
                mask[supply_node] = 1
                
        if mask.sum() == 0:
            return -1  # No feasible match

        # **✅ FIX: Pass `demand_node_id` when calling `select_action`**
        action, _ = self.policy.select_action(
            self.reward_matrix, inventory_state, self.initial_inventory, mask, demand_node_id, deterministic=True
        )

        return action


class DepletionNNPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        logits = logits.masked_fill(mask == 0, -1e9)  # ✅ mask infeasible actions
        return logits

    def select_action(self, features: torch.Tensor, mask: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(features, mask)
            return torch.argmax(logits).item()

class NNPolicyWrapper:
    def __init__(self, policy, graph, initial_inventory, reward_matrix, T):
        self.policy = policy
        self.graph = graph
        self.initial_inventory = torch.tensor(initial_inventory).float()
        self.reward_matrix = reward_matrix
        self.T = T
        self.num_online = len(graph.demand_nodes)

    def __getitem__(self, key):
        inventory_state, t, demand_node_id = key
        inventory_state = torch.tensor(inventory_state).float()
        num_offline = len(inventory_state)

        # Mask valid offline nodes
        mask = torch.zeros(num_offline)
        for i in self.graph.demand_nodes[demand_node_id].neighbors:
            if inventory_state[i] > 0:
                mask[i] = 1

        if mask.sum() == 0:
            return -1

        with torch.no_grad():
            SF = 1.0 - (inventory_state / (self.initial_inventory + 1e-8))
            time_remaining = torch.tensor([self.T - t] * num_offline).float()
            reward_vector = self.reward_matrix[:, demand_node_id].float()

            demand_one_hot = torch.zeros(self.num_online)
            demand_one_hot[demand_node_id] = 1
            demand_feat = demand_one_hot.repeat(num_offline, 1)

            features = torch.stack([inventory_state, SF, time_remaining, reward_vector], dim=1)
            full_features = torch.cat([features, demand_feat], dim=1)

            return self.policy.select_action(full_features, mask)


if __name__ == '__main__':
    # Example Usage
    supply_nodes, demand_nodes, rewards = ('simplest_graph_109_supplynodes.csv', 'simplest_graph_109_demandnodes.csv', 'simplest_graph_109_rewards.csv')
    graph = Graph(supply_nodes, demand_nodes, rewards)
    myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
    graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
    
    balance_fulfiller = BalanceFulfillment(graph)
    T = 10
    
    initial_inventory_tuple = (5,5)
    iid_p = [1/2, 1/4, 1/4]
    p = {t: iid_p for t in range(T)}
    
    p = {}
    # p[0] = [0, 1, 0]
    # p[1] = [0, 1, 0]
    # p[2] = [0, 1, 0]
    # p[3] = [1, 0, 0]
    # p[4] = [1, 0, 0]
    # p[5] = [1,0,0]
    # p[6] = [1,0,0]
    # p[7] = [1,0,0]
    # p[8] = [0,0,1]
    # p[9] = [0,0,1]
    
    p[0] = [1/12, 5/6, 1/12]
    p[1] = [1/6, 2/3, 1/6]
    p[2] = [1/12, 5/6, 1/12]
    p[3] = [1/6, 2/3, 1/6]
    p[4] = [3/4, 1/8, 1/8]
    p[5] = [2/3,1/6,1/6]
    p[6] = [3/4,1/8,1/8]
    p[7] = [1,0,0]
    p[8] = [0,0,1]
    p[9] = [0,0,1]
    
    
    # p[0] = [0, 1, 0]
    # p[1] = [0, 1, 0]
    # p[2] = [0, 1, 0]
    # p[3] = [0, 1, 0]
    # p[4] = [0, 1, 0]
    # p[5] = [1/4, 0, 3/4]
    # p[6] = [1/4, 0, 3/4]
    # p[7] = [1/4, 0, 3/4]
    # p[8] = [1/4, 0, 3/4]
    # p[9] = [3/4, 0, 1/4]
    
    inventory = Inventory({0:5, 1:5})
    
    initial_inventory_tuple = (5,5)
    
    demand_node_list = sorted([demand_node_id for demand_node_id in graph.demand_nodes])
    train_generator = TemporalIndependenceGenerator(demand_node_list,p,seed=12)
    # train_generator = RWGenerator(mean = T, demand_nodes=demand_node_list,seed = 12,distribution='deterministic',step_size = 2)
    
    train_sequences = [train_generator.generate_sequence() for _ in range(100)]
    

    # Extract reward matrix
    reward_matrix = extract_reward_matrix(graph)

    # Initialize trained policy
    # trained_policy = train_depletion_policy(graph, train_sequences,reward_matrix, num_epochs = 251)
    # trained_policy = train_depletion_policy_black_box(graph, train_sequences, reward_matrix, T=10)
    trained_policy = train_depletion_nn_policy(graph, train_sequences, reward_matrix, num_epochs=300, lr=0.005, hidden_dim=16, T=10)
    # wrapped_policy = SubscriptableDepletionPolicyWrapper(trained_policy, graph, reward_matrix, initial_inventory_tuple)
    wrapped_policy = NNPolicyWrapper(trained_policy,graph, initial_inventory_tuple, reward_matrix, T)
    # (self, policy, graph, initial_inventory, T):
    n_test_samples = 500
    test_samples = [train_generator.generate_sequence() for _ in range(n_test_samples)] 

    trained_avg = 0
    bal_avg = 0
    # Evaluate the policy
    for test_sample in test_samples:
        fulfillments, p_rewards, lost = simulate_policy(graph, test_sample, inventory, wrapped_policy, reward_matrix )
        trained_avg += p_rewards / n_test_samples

        fulfillments, b_rewards, lost = balance_fulfiller.fulfill(sequence=test_sample, inventory=inventory, verbose=False)
        bal_avg += b_rewards / n_test_samples

    print(f"Depletion Policy Total Reward: {trained_avg}")
    print(f"Balance Total Reward: {bal_avg}")
# # Example usage:
# num_offline_nodes = 5
# inventory_dim = num_offline_nodes  # Inventory levels for each offline node
# online_node_dim = 3  # Assume each online node has 3 features

# policy = SoftmaxMatchingPolicy(num_offline_nodes, inventory_dim, online_node_dim)

# # Example input state
# inventory_state = torch.tensor([3, 2, 0, 1, 5]).float()  # Inventory levels
# online_node_features = torch.tensor([0.2, 0.5, -0.1]).float()  # Online node characteristics
# time_remaining = torch.tensor(10).float()  # Time remaining (e.g., 10 time steps left)

# # Mask to indicate availability (1 = available, 0 = depleted)
# mask = (inventory_state > 0).float()

# # Get action
# action, log_prob = policy.select_action(inventory_state, online_node_features, time_remaining, mask, deterministic=False)
# print(f"Selected action: {action}, Log-probability: {log_prob.item()}")




# # Define problem dimensions
# num_offline_nodes = 5   # Number of offline nodes (N)
# num_online_features = 3  # Number of features per online node
# T = 10                  # Number of arrivals per sequence
# num_sequences = 100      # Number of independent sequences (episodes)

# # Initialize the policy model
# policy = SoftmaxMatchingPolicy(num_offline_nodes, num_offline_nodes, num_online_features)

# # Define the optimizer
# optimizer = optim.Adam(policy.parameters(), lr=0.01)

# # Reward matrix: Each (offline, online) pair has a fixed reward (randomly initialized here)
# reward_matrix = torch.rand(num_offline_nodes, num_online_features)  # Simulated reward function

# # Function to generate a batch of online arrival sequences
# def generate_online_arrivals(batch_size, T, num_online_features):
#     """
#     Generate a batch of online arrivals with random feature vectors.
#     :return: Tensor of shape (batch_size, T, num_online_features)
#     """
#     return torch.randn(batch_size, T, num_online_features)  # Normalized features

# # Training loop
# num_epochs = 500  # Number of training iterations
# gamma = 1.0  # No discounting

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
    
#     # Generate a batch of sequences
#     online_arrivals = generate_online_arrivals(num_sequences, T, num_online_features)
    
#     total_loss = 0
#     for seq_idx in range(num_sequences):
#         # Initialize inventory levels
#         inventory_state = torch.full((num_offline_nodes,), fill_value=5).float()
        
#         log_probs = []  # Store log-probabilities for gradient update
#         rewards = []  # Store rewards
        
#         for t in range(T):
#             online_node_features = online_arrivals[seq_idx, t]
#             time_remaining = torch.tensor(T - t).float().detach()

#             # Compute availability mask
#             mask = (inventory_state > 0).float().detach()
            
#             # Select an action
#             action, log_prob = policy.select_action(inventory_state, online_node_features, time_remaining, mask, deterministic=False)
            
#             # Store log probability for training
#             log_probs.append(log_prob)

#             # Compute reward based on the chosen offline node and online node features
#             reward = torch.sum(reward_matrix[action] * online_node_features)  # Fix dot product issue
#             rewards.append(reward)
            
#             # Update inventory (decrease count of selected offline node)
#             inventory_state[action] -= 1
#             inventory_state = inventory_state.detach()  # Stop gradient on inventory updates
        
#         # Compute total collected reward
#         total_reward = sum(rewards)  # Maximize total reward
        
#         # Fix: Multiply log-probabilities by total reward to reintroduce gradient dependence
#         policy_loss = -sum(log_probs) * total_reward  # Encourage high-reward actions
        
#         total_loss += policy_loss
    
#     # Backpropagate and update policy
#     total_loss /= num_sequences
#     total_loss.backward()
#     optimizer.step()
    
#     # Logging
#     if epoch % 50 == 0:
#         print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

# print("Training complete!")


