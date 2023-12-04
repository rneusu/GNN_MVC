import random
import time
import networkx as nx
import matplotlib.pyplot as plt

V = 0
E = 0
all = 0
cll = 0
cnt = 60
bll = 0.0
num_generations = 0
pop_size = 0
crossover_prob = 0
mutation_prob = 0
edge = []
edge_w = []
weight = []
population = []
deg = []
bests = []
best = float('inf')
start = time.time()

def GTL(state):
    class weighted_edge:
        def __init__(self, nod1, nod2, w):
            self.nod1 = nod1
            self.nod2 = nod2
            self.w = w

    class individual:
        def __init__(self, vec, fitness_value):
            self.vec = vec
            self.fitness_value = fitness_value
        

    def random_num():
        ret = random.randint(0, 2**64)
        return ret
    
    def input():
        print("GTL Algorithm running")
        global V, E, all, bll, cll, num_generations, pop_size, crossover_prob, mutation_prob, cnt, deg, weight, edge, edge_w
        vertices = state.g.nodes()
        edges = state.g.edges()
        V = len(vertices)
        E = len(edges[0])
        deg = []
        weight = []
        for i in range(V):
            deg.append(0)
            weight.append(1)
        edge = []
        edge_w = []
        for i in range(E):
            all = int(edges[0][i])  # Convert all to int
            cll = int(edges[1][i])
            edge.append((all, cll))
            deg[all] += 1
            deg[cll] += 1
            tmp = weighted_edge(all, cll, 1)
            edge_w.append(tmp)

        
    def generate_random_individual():
        tmp = individual([], 0)
        for i in range(V):
            tmp.vec.append(random_num() % 2)
        
        print("FFFF", len(tmp.vec))
        return tmp
    
    def make_vertex_cover():
        tmp = individual([], 0)
        print(state.visited)
        size = V
        for i in range(size):
            tmp.vec.append(int(state.visited[i]))
        return tmp

    def check_vertex_cover(cur):
        for i in edge:
            if cur.vec[i[0]] == 0 and cur.vec[i[1]] == 0:
                return False
        return True

    def calculate_cost(cur):
        ret = 0
        for i in edge_w:
            if cur.vec[i.nod1] == 0 and cur.vec[i.nod2] == 0:
                ret += i.w
                #i.w += 1 #Why is this here?
        return ret

    def calculate_weight(cur):
        ret = 0
        for i in range(V):
            ret += cur.vec[i] * weight[i]
        return ret

    def calculate_score(cur):
        ret = []
        current_cost = calculate_cost(cur)
        for i in range(V):
            dscore = 0
            if cur.vec[i] == 1:
                cur.vec[i] = 0
                dscore = calculate_cost(cur)
                cur.vec[i] = 1
                dscore -= current_cost
            if cur.vec[i] == 0:
                cur.vec[i] = 1
                dscore = -calculate_cost(cur)
                cur.vec[i] = 0
                dscore += current_cost
                
            if weight[i] != 0:
                ret.append(dscore / weight[i])
            else:
                print("division by zero error")
                print("weight[i] = ", weight[i])
                print(f'i: {i}')
        return ret

    def add_weight(cur):
        for i in edge_w:
            if not cur.vec[i.nod1] and not cur.vec[i.nod2]:
                i.w += 1

    def local_search():
        bests = []
        C = make_vertex_cover()
        print("debug_2")
        score_matrix = calculate_score(C)
        print("debug_3")
        start = time.time()
        array_pointer = 0
        time_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500]
        siz = len(time_array)
        ans_array = []
        print("debug_4")
        current_weight = calculate_weight(C)

        while time.time() - start < cnt:
            print("debug_5")
            bests.append(calculate_weight(C))
            time_now = time.time() - start
            previous_weight = current_weight
            while array_pointer < siz and time_array[array_pointer] <= time_now:
                ans_array.append(current_weight)
                array_pointer += 1
                print(f"{time_array[array_pointer - 1]}\t\t{ans_array[array_pointer - 1]}")
            
                        
            if(previous_weight > current_weight):
                print((time.time() - start), calculate_weight(C))

            print("debug_6")
            
            for o in range(10):
                min_score = 2**100
                min_ind = -1
                for i in range(V):
                    if C.vec[i] == 1 and min_score > score_matrix[i]:
                        min_score = score_matrix[i]
                        min_ind = i
                
                if min_ind != -1:
                    C.vec[min_ind] = 0
            
            print("debug_7")

            while not check_vertex_cover(C):
                max_score = -(2**100)
                max_deg = -(2**100)
                max_ind = -1
                for i in range(V):
                    if C.vec[i] == 0 and (max_score < score_matrix[i] or (max_score == score_matrix[i] and max_deg < deg[i])):
                        max_score = score_matrix[i]
                        max_deg = deg[i]
                        max_ind = i
                C.vec[max_ind] = 1
                add_weight(C)
        
            bests.append(calculate_weight(C))
            
            #if calculate_weight(C) > calculate_weight(previous):
            #    C = previous
            #else:   
            score_matrix = calculate_score(C)
            
            bests.append(calculate_weight(C))
            
            current_weight = calculate_weight(C)
        print(f"iteration array {bests}", file=open('output/GTL_result.txt','a'))
        return min(bests)
            

    input()
    anss = local_search()
    
    print(f"the result is: {anss}", file=open('output/GTL_result.txt','a'))
    print("the result is:", anss)
    print("it took", time.time() - start, "seconds")

    return anss