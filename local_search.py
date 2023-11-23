import random
import time

V = 0
E = 0
all = 0
cll = 0
cnt = 0
bll = 0.0
num_generations = 0
pop_size = 0
crossover_prob = 0
mutation_prob = 0
edge = []
edge_w = []
weight = []
population = []
bests = []
best = float('inf')
start = time.time()

def LOCALSEARCH(mvc):
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
        ret *= random.randint(0, 2**64)
        ret += random.randint(0, 2**64)
        ret *= random.randint(0, 2**64)
        ret = abs(ret)
        return ret
    
    def input():
        global V, E, all, bll, cll, num_generations, pop_size, crossover_prob, mutation_prob, cnt
        init_state,_ = mvc.reset()
        vertices = init_state.g.nodes()
        edges = init_state.g.edges()
        print(f'init_state: {init_state.g}')
        print(f'vertices: {vertices}')
        print(f'edges: {edges}')
        V = len(vertices)
        E = len(edges[0])
        for i in range(V):
            weight.append(1)
        
        for i in range(E):
            all = int(edges[0][i])  # Convert all to int
            cll = int(edges[1][i])
            edge.append((all, cll))
            tmp = weighted_edge(all, cll, 1)
            edge_w.append(tmp)
        
        cnt = 100

    def generate_random_individual():
        tmp = individual([], 0)
        for i in range(V):
            tmp.vec.append(random_num() % 2)
        
        print("FFFF", len(tmp.vec))
        return tmp
    
    def make_vertex_cover(cur):
        for i in edge:
            if not cur.vec[i[0]] and not cur.vec[i[1]]:
                if random_num() % 2 == 1:
                    cur.vec[i[0]] = 1
                else:
                    cur.vec[i[1]] = 1
        return cur

    def check_vertex_cover(cur):
        for i in edge:
            if not cur.vec[i[0]] and not cur.vec[i[1]]:
                return False
        return True

    def calculate_cost(cur):
        ret = 0
        for i in edge_w:
            if not cur.vec[i.nod1] and not cur.vec[i.nod2]:
                ret += i.w
                i.w += 1
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
        global bests
        C = generate_random_individual()
        print("ALIVE1", len(C.vec))
        C = make_vertex_cover(C)
        print("ALIVE2")
        score_matrix = calculate_score(C)
        start = time.time()
        while time.time() - start < cnt:
            print((time.time() - start), calculate_weight(C))
            previous = C
            min_score = float('inf')
            min_ind = 0
            for i in range(V):
                if C.vec[i] == 1 and min_score > score_matrix[i]:
                    min_score = score_matrix[i]
                    min_ind = i
            C.vec[min_ind] = 0
            while not check_vertex_cover(C):
                max_score = -1
                max_ind = 0
                for i in range(V):
                    if C.vec[i] == 0 and max_score < score_matrix[i]:
                        max_score = score_matrix[i]
                        max_ind = i
                C.vec[max_ind] = 1
                add_weight(C)
            if calculate_weight(C) > calculate_weight(previous):
                C = previous
            else:
                score_matrix = calculate_score(C)
        bests.append(calculate_weight(C))


    input()
    local_search()
    print(f"the result is: {bests[0]}", file=open('output/test_result.txt','a'))
    print("the result is:", bests[0])
    print("it took", time.time() - start, "seconds")
