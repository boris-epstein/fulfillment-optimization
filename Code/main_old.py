from Graph import Graph
from Demand import CorrelGenerator
from Fulfillment import Fulfillment
import time
import numpy as np
from collections import defaultdict
import csv
from statistics import mean, stdev



start_time = time.time()



# graph = Graph(mode = 'test_graph_1')
graph = Graph()
print(len(graph.origins))
print(len(graph.destinations))
print(len(graph.edges))
# print(len(graph.destinations['DAB4'].closest_origins))



mu = 5000
std_dev = 4000
seed = 1

z = 0.5


total_inventory = np.round( mu + z*std_dev )

fulfillment = Fulfillment(graph)
fulfillment.set_proportional_inventories(total_inventory)
# fulfillment.set_inventories_to_n(10)

print(fulfillment.initial_inventories)
print( total_inventory, sum( fulfillment.initial_inventories[origin] for origin in graph.origins) )

fulfillment.prepare_fluid_shadow_prices_fulfillment(mu,graph.max_distance)


generator = CorrelGenerator(mu,std_dev,graph.destination_list,graph.destination_weights,seed)

myopic_fulfillment_list_by_edge = defaultdict(lambda : [])
fluid_fulfillment_list_by_edge = defaultdict(lambda :[])

total_demands = []



print('Doing fulfillment')


n_samples = 500


for i in range(n_samples):
    if i%100 ==0: print(i) 
    seq = generator.generate_sequence()
    
    total_demands.append(seq.length)
    
    fuls = fulfillment.myopic_fulfillment(seq)

    
    for ori, des in graph.edges:
        myopic_fulfillment_list_by_edge[ori,des].append( fuls[ori,des] )

       
        
        
    fuls2 = fulfillment.fluid_shadow_prices_fulfillment(seq)
    
    for ori,des in graph.edges:
        fluid_fulfillment_list_by_edge[ori,des].append( fuls2[ori,des] )


# print(fulfillment.initial_inventories)

print('Computing averages')
myopic_aggregate_info = []
fluid_aggregate_info = []

for ori,des in graph.edges:
    myopic_aggregate_info.append( ( ori, des, mean(myopic_fulfillment_list_by_edge[ori,des]) , stdev(myopic_fulfillment_list_by_edge[ori,des]) ) )
    fluid_aggregate_info.append( ( ori, des, mean(fluid_fulfillment_list_by_edge[ori,des]), stdev(fluid_fulfillment_list_by_edge[ori,des]) ) )

myopic_aggregate_info.sort(key = lambda x:x[2], reverse = True)
fluid_aggregate_info.sort(key = lambda x:x[2], reverse = True)


print('Myopic top edges')
for i in range(10):
    print( myopic_aggregate_info[i] )
    
print('')

print('Fluid top edges')
for i in range(10):
    print( fluid_aggregate_info[i] )



with open(f'myopic_output_samples_{n_samples}_seed_{seed}.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['origin', 'destination', 'myopic_average','myopic_stdev','myopic_CV'])
        for i in range(len(myopic_aggregate_info)):
            row = myopic_aggregate_info[i]
            if row[2]>0:
                spamwriter.writerow( [row[0],row[1],row[2],row[3], row[3]/row[2]] )
            else:
                spamwriter.writerow( [row[0],row[1],row[2],row[3], 0] )
            
            
with open(f'fluid_output_samples_{n_samples}_seed_{seed}.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['origin', 'destination',  'fluid_average', 'fluid_stdev','fluid_CV'])
        for i in range(len(fluid_aggregate_info)):
            row = fluid_aggregate_info[i]
            if row[2]>0:
                spamwriter.writerow( [row[0],row[1],row[2],row[3], row[3]/row[2]] )
            else:
                spamwriter.writerow( [row[0],row[1],row[2],row[3], 0] )
            




print('done')
print("--- %s seconds ---" % (time.time() - start_time))




# # for ori in graph.origins:
# #     print(graph.origins[ori].region)