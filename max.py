import numpy as np

def userCostInput():
    sp_row = int(input('Enter length row value: '))
    dm_column = int(input('Enter length column value: '))
    userCostInput.costs = [[int(input()) for x in range(dm_column)] for y in range(sp_row)]

def userSupplyAndDemand():
    supply = int(input("Enter the length of supply values: "))
    demand = int(input("Enter the length of demand values: "))
    userSupplyAndDemand.Supply = [int(input("Enter Supply values: ")) for x in range(supply)]
    userSupplyAndDemand.Demand = [int(input("Enter Demand values: ")) for x in range(demand)]

cost = None
def verify_problem(costs):
    global cost
    # making the costs matrix a numpy array
    array = np.array(costs)
    maxVal = np.max(costs)
    array = maxVal - array
    verify_problem.cost = array.tolist()
    print(f'New Cost: {verify_problem.cost}')
    new_costs = verify_problem.cost
    return new_costs

def get_balanced_transportation_problem(model_supply,model_demand, Costs):
    total_supply = sum(model_supply)
    total_demand = sum(model_demand)
    
    if total_supply < total_demand:
        new_supply = model_supply + [total_demand - total_supply]
        new_costs = Costs 
        return new_supply, model_demand, new_costs
    if total_supply > total_demand:
        new_demand = model_demand + [total_supply - total_demand]
        new_costs = Costs + [[0 for _ in model_demand]]
        return model_supply, new_demand, new_costs
    return model_supply, model_demand, Costs

def north_west_corner(model_supply, model_demand):
    supply_copy = model_supply.copy()
    demand_copy = model_demand.copy()
    i = 0
    j = 0
    bfs = []
    while len(bfs) < len(model_supply) + len(model_demand) - 1:
        s = supply_copy[i]
        d = demand_copy[j]
        v = min(s, d)
        supply_copy[i] -= v
        demand_copy[j] -= v
        bfs.append(((i, j), v))
        if supply_copy[i] == 0 and i < len(model_supply) - 1:
            i += 1
        elif demand_copy[j] == 0 and j < len(model_demand) - 1:
            j += 1
    return bfs

def get_us_and_vs(bfs, Costs):
    us = [None] * len(Costs)
    vs = [None] * len(Costs[0])
    us[0] = 0
    bfs_copy = bfs.copy()
    while len(bfs_copy) > 0:
        for index, bv in enumerate(bfs_copy):
            i, j = bv[0]
            if us[i] is None and vs[j] is None: continue
                
            cost = Costs[i][j]
            if us[i] is None:
                us[i] = cost - vs[j]
            else: 
                vs[j] = cost - us[i]
            bfs_copy.pop(index)
            break
            
    return us, vs 

def get_ws(bfs, Costs, us, vs):
    ws = []
    for i, row in enumerate(Costs):
        for j, cost in enumerate(row):
            non_basic = all([p[0] != i or p[1] != j for p, v in bfs])
            if non_basic:
                ws.append(((i, j), us[i] + vs[j] - cost))
    
    return ws

def can_be_improved(ws):
    for p, v in ws:
        if v > 0: return True
    return False

def get_entering_variable_position(ws):
    ws_copy = ws.copy()
    ws_copy.sort(key=lambda w: w[1])
    return ws_copy[-1][0]

def get_possible_next_nodes(loop, not_visited):
    last_node = loop[-1]
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    else:
        prev_node = loop[-2]
        row_move = prev_node[0] == last_node[0]
        if row_move: return nodes_in_column
        return nodes_in_row


def get_loop(bv_positions, ev_position):
    def inner(loop):
        if len(loop) > 3:
            can_be_closed = len(get_possible_next_nodes(loop, [ev_position])) == 1
            if can_be_closed: return loop
        
        not_visited = list(set(bv_positions) - set(loop))
        possible_next_nodes = get_possible_next_nodes(loop, not_visited)
        for next_node in possible_next_nodes:
            new_loop = inner(loop + [next_node])
            if new_loop: return new_loop
    
    return inner([ev_position])


def loop_pivoting(bfs, loop):
    even_cells = loop[0::2]
    odd_cells = loop[1::2]
    get_bv = lambda pos: next(v for p, v in bfs if p == pos)
    leaving_position = sorted(odd_cells, key=get_bv)[0]
    leaving_value = get_bv(leaving_position)
    
    new_bfs = []
    for p, v in [bv for bv in bfs if bv[0] != leaving_position] + [(loop[0], 0)]:
        if p in even_cells:
            v += leaving_value
        elif p in odd_cells:
            v -= leaving_value
        new_bfs.append((p, v))
        
    return new_bfs

def transportation_stepping_stone(model_supply, model_demand, Costs):
    balanced_supply, balanced_demand, balanced_costs = get_balanced_transportation_problem(
        model_supply, model_demand, Costs
    )
    def inner(bfs):
        us, vs = get_us_and_vs(bfs, balanced_costs)
        ws = get_ws(bfs, balanced_costs, us, vs)
        if can_be_improved(ws):
            ev_position = get_entering_variable_position(ws)
            loop = get_loop([p for p, v in bfs], ev_position)
            return inner(loop_pivoting(bfs, loop))
        return bfs
    
    basic_variables = inner(north_west_corner(balanced_supply, balanced_demand))
    solution = np.zeros((len(Costs), len(Costs[0])))
    for (i, j), v in basic_variables:
        solution[i][j] = v

    return solution

def get_total_cost(Costs, solution):
    total_cost = 0
    for i, row in enumerate(Costs):
        for j, cost in enumerate(row):
            total_cost += cost * solution[i][j]
    return total_cost


if __name__ == "__main__":
    userCostInput()
    costs = userCostInput.costs
    costs
    userSupplyAndDemand()
    model_supply = userSupplyAndDemand.Supply
    model_demand = userSupplyAndDemand.Demand
    print(model_supply)
    print(model_demand)
    Costs = verify_problem(costs)
    balanced_model = get_balanced_transportation_problem(model_supply, model_demand, Costs)
    balanced_model
    bfs = north_west_corner(model_supply,model_demand)
    print(bfs)
    solution = transportation_stepping_stone(model_supply, model_demand, Costs)
    print(solution)
    print('Optimal Solution: ', get_total_cost(Costs, solution))
