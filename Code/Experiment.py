
from Demand import CorrelGenerator
from Fulfillment import Fulfillment, NewsvendorModel
from Graph import Graph
from collections import defaultdict
import numpy as np
import csv
from typing import List





class Experiment:
    
    def __init__(self, graph: Graph,
                 policies_to_try: List[str],
                 total_inventories: List[int],
                 n_samples : int,
                 total_inventory: int,
                 mean_demand: float,
                 st_dev_demand: float,
                 truck_fixed_cost: float,
                 truck_fixed_cost_high: float,
                 truck_variable_cost: float,
                 truck_capacity: int,
                 seed : int = 0,
                 alpha: int = 1,
                 beta:int =  1 ,
                 inventory_mode: str = 'proportional', 
                 resolving_times: list=[]
            ) -> None:
        
        self.graph = graph
        
        self.total_inventories = total_inventories
        
        
        
        self.generator = CorrelGenerator(mean_demand,st_dev_demand,graph.destination_list,graph.destination_weights,seed)
        
        self.fulfillment = Fulfillment(self.graph)
        
        self.n_samples = n_samples
    
        self.total_inventory = total_inventory
        
        self.policies_to_try = policies_to_try
        

        self.resolving_times = resolving_times
        
        self.seed = seed
        
        for policy in self.policies_to_try:
            self.fulfillment_lists_by_edge[policy] = defaultdict(lambda: np.zeros(n_samples))
        
        if inventory_mode == 'proportional':
            self.fulfillment.set_proportional_inventories(self.total_inventory)


        self.total_demands = [0 for i in range(self.n_samples)]
    
        self.fulfillment_costs = defaultdict(lambda: np.zeros(n_samples))
        self.mismatch_costs = defaultdict(lambda: np.zeros(n_samples))
        
        self.lost_sales = defaultdict(lambda: np.zeros(n_samples))
    
    
        self.newsvendor_model = NewsvendorModel(truck_fixed_cost, truck_fixed_cost_high, truck_variable_cost, truck_capacity)
        
    
    def run_simulations(self):
        
        if 'fluid' in self.policies_to_try:
            self.fulfillment.prepare_fluid_shadow_prices_fulfillment(self.generator.mean,self.graph.max_distance)
        
        print('Running the simluations')
        for i in range(self.n_samples):
            if i%10 == 0: print(i) 
            
            
            seq = self.generator.generate_sequence()
            
            self.total_demands.append(seq.length)
            
            
            if 'myopic' in self.policies_to_try:
                fuls, fulfillment_cost, lost_sales = self.fulfillment.myopic_fulfillment(seq)
                
                self.fulfillment_costs['myopic'][i] = fulfillment_cost
                self.lost_sales['myopic'][i] = lost_sales
                
                for ori, des in self.graph.edges:
                    self.fulfillment_lists_by_edge['myopic'][ori,des][i] =  fuls[ori,des] 

        
            if 'fluid' in self.policies_to_try:
        
                fuls, fulfillment_cost, lost_sales = self.fulfillment.fluid_shadow_prices_fulfillment(seq)
                
                self.fulfillment_costs['fluid'][i] = fulfillment_cost
                self.lost_sales['fluid'][i] = lost_sales
                
                for ori,des in self.graph.edges:
                    self.fulfillment_lists_by_edge['fluid'][ori,des][i] =  fuls[ori,des] 
    
            if 'fluid_resolving' in self.policies_to_try:
        
                fuls, fulfillment_cost, lost_sales = self.fulfillment.fluid_fulfillment_with_resolving(seq,self.graph.max_distance, self.resolving_times)
            
                self.fulfillment_costs['fluid_resolving'][i] = fulfillment_cost
                self.lost_sales['fluid_resolving'][i] = lost_sales
            
                for ori,des in self.graph.edges:
                    self.fulfillment_lists_by_edge['fluid_resolving'][ori,des][i] =  fuls[ori,des] 

            if 'offline' in self.policies_to_try:
                fuls, fulfillment_cost, lost_sales = self.fulfillment.offline_fulfillment(self.graph.max_distance, seq, )
                self.fulfillment_costs['offline'][i] = fulfillment_cost
                self.lost_sales['offline'][i] = lost_sales
            
                for ori,des in self.graph.edges:
                    self.fulfillment_lists_by_edge['offline'][ori,des][i] =  fuls[ori,des] 


    def compute_statistics(self):
        
        
        
        self.aggregate_edge_info = {}
        self.average_fulfillment_cost = {}
        self.stdev_fulfillment_cost = {}
        self.average_lost_sales = {}
        self.stdev_lost_sales = {}
        
        self.distance_weighted_stdev = defaultdict(float)
        self.distance_weighted_variance = defaultdict(float)
        self.distance_mean_weighted_cv = defaultdict(float)
        self.distance_weighted_cv = defaultdict(float)
        
        self.edge_ever_used = {}
        self.truck_demand_distribution = {}
        for policy in self.policies_to_try:
            self.edge_ever_used[policy] = defaultdict(lambda: False)
            self.truck_demand_distribution[policy] = {}
        
        
        
        print('Computing averages')
        for policy in self.policies_to_try:
            self.aggregate_edge_info[policy] = [None for i in range(len(self.graph.edges))]
            self.average_fulfillment_cost[policy] = np.mean(self.fulfillment_costs[policy])
            self.stdev_fulfillment_cost[policy] = np.std(self.fulfillment_costs[policy],ddof=1 )
            
            self.average_lost_sales[policy] = np.mean(self.lost_sales[policy])
            self.stdev_lost_sales[policy]  = np.std(self.lost_sales[policy], ddof=1)

        dist_times_mu_denom = defaultdict( float )


        self.excess_costs = {}
        self.optimal_newsvendor_quantity = {} #maps policy and edge to optimal order



        i=0
        for ori,des in self.graph.edges:
            
            self.excess_costs[ori,des] = self.newsvendor_model.compute_excess_cost(self.graph.edges[ori,des].cost)
            self.critical_fractile[ori,des] = self.newsvendor_model.compute_critical_fractile(self.graph.edges[ori,des].cost)
            
            
            for policy in self.policies_to_try:
                # Vanilla aggregate info
                self.aggregate_edge_info[policy][i] = ( ori, des, np.mean(self.fulfillment_lists_by_edge[policy][ori,des]) ,  np.std(self.fulfillment_lists_by_edge[policy][ori,des],ddof=1 ) )
                if self.aggregate_edge_info[policy][i][2] >0:
                    self.edge_ever_used[policy][ori,des] = True
                
                
                
                # Aggregate statistics (weighted CV and stdev s)
                self.distance_weighted_stdev[policy] += self.aggregate_edge_info[policy][i][3] * self.graph.edges[ori,des].cost
                self.distance_weighted_variance[policy] += self.aggregate_edge_info[policy][i][3]**2 * self.graph.edges[ori,des].cost
                self.distance_mean_weighted_cv[policy] += self.aggregate_edge_info[policy][i][3] * self.graph.edges[ori,des].cost
                if self.aggregate_edge_info[policy][i][2] > 0:
                    self.distance_weighted_cv[policy] += self.aggregate_edge_info[policy][i][3]/self.aggregate_edge_info[policy][i][2] * self.graph.edges[ori,des].cost
                
                dist_times_mu_denom[policy] += self.aggregate_edge_info[policy][i][2] * self.graph.edges[ori,des].cost
                
                
                
                #For newsvendor
                if self.edge_ever_used[policy][ori,des]:
                    self.truck_demand_distribution[policy][ori,des] = self.convert_flow_to_truck_distribution(policy, ori, des)
                    q_star = self.newsvendor_model.compute_optimal_newsvendor_quantity(self.critical_fractile[ori,des],self.truck_demand_distribution[policy][ori,des])
                    self.optimal_newsvendor_quantity[policy,ori,des] = q_star
                
                
                
                
            i+=1
        
        
        for i in range(self.n_samples):
            for policy in self.policies_to_try:
                for ori,des in self.graph.edges:
                    if self.edge_ever_used[policy][ori,des]:
                        # Newsvendor Stuff
                        q_star = self.optimal_newsvendor_quantity[policy,ori,des]
                        
                        demand = self.truck_demand_distribution[policy][ori,des][i]
                        mismatch_cost = self.newsvendor_model.compute_mismatch_cost(demand,self.excess_costs[ori,des],q_star)
                        
                        self.mismatch_costs[policy][i] += mismatch_cost
                        
        
        
        
        for policy in self.policies_to_try:
            self.distance_weighted_stdev[policy] = self.distance_weighted_stdev[policy]/self.graph.total_distance
            self.distance_weighted_variance[policy] = self.distance_weighted_variance[policy]/self.graph.total_distance
            self.distance_mean_weighted_cv[policy] = self.distance_mean_weighted_cv[policy]/dist_times_mu_denom[policy]
            self.distance_weighted_cv[policy] = self.distance_weighted_cv[policy]/self.graph.total_distance
        
        for policy in self.policies_to_try:
            self.aggregate_edge_info[policy].sort( key = lambda x:x[2], reverse = True )



        # Get some percentiles
        
        self.percentiles = [99.99, 99.9, 99.5, 99.0, 95.0, 90.0]
        
        percentile_indices = [ int( (1-percentile/100)*len(self.graph.edges)  )  for percentile in self.percentiles]
        
        self.percentile_output = {}
        
        for policy in self.policies_to_try:
            self.percentile_output[policy] = [ self.aggregate_edge_info[policy][index][2] for index in percentile_indices ]
            
    
    def convert_flow_to_truck_distribution(self, policy, ori, des):
        
        trucks_required = [ np.ceil(flow/self.newsvendor_model.capacity) for flow in self.fulfillment_lists_by_edge[policy][ori,des] ]
        
        return sorted(trucks_required)
        
        
        
        
        



    def save_output(self, print_edge_details: bool=True):
        
        print('Saving output files')
        if print_edge_details:
            for policy in self.policies_to_try:
                
                with open(f'{policy}_output_samples_{self.n_samples}_seed_{self.seed}.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['origin', 'destination', f'{policy}_average',f'{policy}_stdev',f'{policy}_CV'])
                    for i in range(len(self.aggregate_edge_info[policy])):
                        row = self.aggregate_edge_info[policy][i]
                        if row[2]>0:
                            spamwriter.writerow( [row[0],row[1],row[2],row[3], row[3]/row[2]] )
                        else:
                            spamwriter.writerow( [row[0],row[1],row[2],row[3], 0] )
                        
        with open(f'policy_attributes_nsamples_{self.n_samples}_seed_{self.seed}.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['policy']+ ['mean_fufillment_cost','stdev_fulfillment_cost','mean_lost_sales','stdev_lost_sales','mean_mismatch_cost','stdev_mismatch_cost' ,'distance_mean_weighted_CV','dist_weighted_CV','dist_weighted_stdev','dist_weighted_var'] +[str(percentile) for percentile in self.percentiles])
                for policy in self.policies_to_try:
                    spamwriter.writerow([policy, self.average_fulfillment_cost[policy],self.stdev_fulfillment_cost[policy], self.average_lost_sales[policy], self.stdev_lost_sales[policy] , np.mean(self.mismatch_costs[policy]), np.std(self.mismatch_costs[policy],ddof = 1) , self.distance_mean_weighted_cv[policy],self.distance_weighted_cv[policy],self.distance_weighted_stdev[policy],self.distance_weighted_variance[policy] ]+self.percentile_output[policy] )
                    
                    
                    
                    
                    
                    
                    
    def compute_mismatch_cost(self, demand, quantity_ordered, distance):
        
        mismatch_cost = 0
        
        mismatch_cost += distance * self.alpha * max(0, quantity_ordered - demand) #overage cost
        mismatch_cost += distance * self.beta * max(0, demand - quantity_ordered) #underage cost
        
            
        return mismatch_cost
    
    
    
    def compute_newsvendor_order(self, empirical_distribution: list[int]) -> int:
        
        index = int(np.floor(self.optimal_newsvendor_quantile * self.n_samples))
        
        q_star = sorted(empirical_distribution)[index]
        
        return q_star
        