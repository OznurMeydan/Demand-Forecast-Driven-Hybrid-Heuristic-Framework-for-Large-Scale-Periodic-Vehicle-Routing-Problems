import pandas as pd
import numpy as np
import math
import copy
import random
import time

# ==========================================
# 1. SETTINGS & PARAMETERS
# ==========================================
VEHICLE_CAPACITY = 16.35    # m3 (Truck Capacity)
MAX_ITERATIONS = 1000       # Increased for deeper search
TABU_TENURE = 20            # Prevents cycling for longer
NEIGHBORS_TO_CHECK = 100    # Checks more moves per step

# File Names
DISTANCE_FILE = 'Mesafe_Matris_123_Bayi.xlsx'
FORECAST_FILE = 'GELECEK_51_HAFTA_TAHMINLERI.xlsx'

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_and_process_data():
    print("Loading data...")
    try:
        # Forecast
        forecast_df = pd.read_excel(FORECAST_FILE, engine='openpyxl')
        forecast_df['TARIH'] = pd.to_datetime(forecast_df['TARIH'], format='mixed', errors='coerce')
        forecast_df = forecast_df.dropna(subset=['TARIH'])
        
        pivoted_forecast = forecast_df.pivot_table(index='BAYI_KODU', columns='TARIH', values='TAHMIN_MIKTAR', aggfunc='sum').fillna(0)
        pivoted_forecast = pivoted_forecast.reindex(sorted(pivoted_forecast.columns), axis=1)
        
        # Matrix
        dist_df = pd.read_excel(DISTANCE_FILE, index_col=0, engine='openpyxl')
        dist_df.index = dist_df.index.astype(str)
        dist_df.columns = dist_df.columns.astype(str)
        
        if 'MERKEZ_DEPO' in dist_df.index: depot_id = 'MERKEZ_DEPO'
        else: depot_id = dist_df.index[0]

        print(f"Data Loaded. Weeks: {len(pivoted_forecast.columns)}")
        return pivoted_forecast, dist_df, depot_id
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================
def calculate_frequency_and_load(weekly_volume, capacity):
    if weekly_volume <= 0.001: return 0, 0
    freq = math.ceil(weekly_volume / capacity)
    if freq == 0: freq = 1
    daily_load = weekly_volume / freq
    if daily_load > capacity:
        freq += 1
        daily_load = weekly_volume / freq
    return freq, daily_load

def assign_visit_days(freq):
    if freq == 0: return []
    if freq == 1: return [2]             # Wed
    if freq == 2: return [0, 3]          # Mon, Thu
    if freq == 3: return [0, 2, 4]       # Mon, Wed, Fri
    if freq == 4: return [0, 1, 3, 4]    # Tue empty
    if freq >= 5: return [0, 1, 2, 3, 4] # Mon-Fri
    return [0]

def calculate_route_cost(route, matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += matrix[route[i]][route[i+1]]
    return cost

def check_feasibility(route, node_loads, capacity):
    total_vol = 0
    for i in range(1, len(route) - 1):
        total_vol += node_loads.get(route[i], 0)
    if total_vol > capacity + 0.01: return False
    return True

# ==========================================
# 4. OPTIMIZATION ALGORITHMS (2-OPT + TABU)
# ==========================================

def two_opt_optimize_route(route, matrix):
    """
    Intra-Route Optimization: Reorders nodes within a single route to minimize distance.
    """
    best_route = route[:]
    best_cost = calculate_route_cost(best_route, matrix)
    improved = True
    
    while improved:
        improved = False
        # Try reversing every possible segment
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1: continue
                
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = calculate_route_cost(new_route, matrix)
                
                if new_cost < best_cost - 0.001: # Small tolerance
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    route = best_route # Update current reference
        if improved:
            pass 
    return best_route

def clarke_wright_initial(nodes_to_visit, node_loads, matrix):
    routes = [[0, n, 0] for n in nodes_to_visit]
    savings = []
    for i in nodes_to_visit:
        for j in nodes_to_visit:
            if i != j:
                s = matrix[0][i] + matrix[0][j] - matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)
    
    for sav, i, j in savings:
        r_i_idx, r_j_idx = -1, -1
        for idx, r in enumerate(routes):
            if i in r and r.index(i) == len(r) - 2: r_i_idx = idx
            if j in r and r.index(j) == 1: r_j_idx = idx
            
        if r_i_idx != -1 and r_j_idx != -1 and r_i_idx != r_j_idx:
            new_route = routes[r_i_idx][:-1] + routes[r_j_idx][1:]
            if check_feasibility(new_route, node_loads, VEHICLE_CAPACITY):
                routes[r_i_idx] = new_route
                routes.pop(r_j_idx)
    return routes

def tabu_search(initial_routes, node_loads, matrix):
    # Step 1: Optimize Initial Routes with 2-Opt
    current_routes = [two_opt_optimize_route(r, matrix) for r in initial_routes]
    
    best_sol = copy.deepcopy(current_routes)
    curr_sol = copy.deepcopy(current_routes)
    best_cost = sum([calculate_route_cost(r, matrix) for r in best_sol])
    
    tabu_list = [] 
    
    for iter_num in range(MAX_ITERATIONS):
        candidates = []
        
        # Neighborhood Search
        for _ in range(NEIGHBORS_TO_CHECK):
            temp_sol = copy.deepcopy(curr_sol)
            if len(temp_sol) < 1: break
            
            move_type = random.choice(['Relocate', 'Swap'])
            
            # --- RELOCATE ---
            if move_type == 'Relocate':
                if len(temp_sol) < 2: continue # Need 2 routes
                r_src = random.randint(0, len(temp_sol)-1)
                if len(temp_sol[r_src]) <= 3: continue 
                
                n_idx = random.randint(1, len(temp_sol[r_src]) - 2)
                node = temp_sol[r_src][n_idx]
                r_dst = random.randint(0, len(temp_sol)-1)
                
                temp_sol[r_src].pop(n_idx)
                ins = random.randint(1, len(temp_sol[r_dst]) - 1)
                temp_sol[r_dst].insert(ins, node)
                
                if check_feasibility(temp_sol[r_dst], node_loads, VEHICLE_CAPACITY):
                    if len(temp_sol[r_src]) == 2: temp_sol.pop(r_src)
                    cost = sum([calculate_route_cost(r, matrix) for r in temp_sol])
                    candidates.append((cost, temp_sol, node))

            # --- SWAP ---
            elif move_type == 'Swap':
                if len(temp_sol) < 2: continue
                r1 = random.randint(0, len(temp_sol)-1)
                r2 = random.randint(0, len(temp_sol)-1)
                if r1 == r2: continue
                if len(temp_sol[r1]) <= 2 or len(temp_sol[r2]) <= 2: continue
                
                n1_idx = random.randint(1, len(temp_sol[r1]) - 2)
                n2_idx = random.randint(1, len(temp_sol[r2]) - 2)
                val1 = temp_sol[r1][n1_idx]
                val2 = temp_sol[r2][n2_idx]
                
                temp_sol[r1][n1_idx] = val2
                temp_sol[r2][n2_idx] = val1
                
                if (check_feasibility(temp_sol[r1], node_loads, VEHICLE_CAPACITY) and 
                    check_feasibility(temp_sol[r2], node_loads, VEHICLE_CAPACITY)):
                    cost = sum([calculate_route_cost(r, matrix) for r in temp_sol])
                    # Store tuple for swap tabu check
                    candidates.append((cost, temp_sol, (val1, val2)))

        candidates.sort(key=lambda x: x[0])
        
        # Move Selection
        for cost, sol, node_info in candidates:
            # Check Tabu
            is_tabu = False
            nodes = node_info if isinstance(node_info, tuple) else (node_info,)
            for t_node, t_exp in tabu_list:
                if t_node in nodes and iter_num < t_exp:
                    is_tabu = True
                    break
            
            # Aspiration
            if (not is_tabu) or (cost < best_cost):
                curr_sol = sol
                # Update Tabu List
                for n in nodes: tabu_list.append((n, iter_num + TABU_TENURE))
                
                if cost < best_cost:
                    best_sol = copy.deepcopy(curr_sol)
                    best_cost = cost
                break
                
    # Final Polish: Run 2-Opt one last time on the best solution
    final_routes = [two_opt_optimize_route(r, matrix) for r in best_sol]
    return final_routes

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    start_time = time.time()
    pivoted_forecast, dist_df, depot_id = load_and_process_data()
    if pivoted_forecast is None: return

    valid_ids = [str(d) for d in pivoted_forecast.index if str(d) in dist_df.index]
    ordered_nodes = [depot_id] + valid_ids
    final_matrix = dist_df.loc[ordered_nodes, ordered_nodes].values
    final_matrix = np.nan_to_num(final_matrix, nan=10000.0)
    
    idx_to_real_id = {i: node_id for i, node_id in enumerate(ordered_nodes)}
    real_id_to_idx = {node_id: i for i, node_id in enumerate(ordered_nodes)}
    
    summary_results = []
    detailed_routes = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    print(f"\nOptimization Started (Advanced Mode)...")
    
    for week_date in pivoted_forecast.columns:
        week_str = str(week_date.date())
        print(f" -> Processing: {week_str}...", end="\r")
        
        node_daily_loads = {}
        schedule = {0:[], 1:[], 2:[], 3:[], 4:[]}
        
        for d_id_str in valid_ids:
            try: vol = pivoted_forecast.loc[int(d_id_str), week_date]
            except: vol = 0
            if vol > 0.001:
                mat_idx = real_id_to_idx[d_id_str]
                freq, load = calculate_frequency_and_load(vol, VEHICLE_CAPACITY)
                days = assign_visit_days(freq)
                for d in days:
                    schedule[d].append(mat_idx)
                    node_daily_loads[mat_idx] = load
        
        weekly_dist = 0
        weekly_trucks = 0
        
        for day_idx in range(5):
            nodes_today = schedule[day_idx]
            if not nodes_today: continue
            
            # --- HYBRID ALGORITHM ---
            # 1. Construction
            init_routes = clarke_wright_initial(nodes_today, node_daily_loads, final_matrix)
            # 2. Improvement (Metaheuristic)
            opt_routes = tabu_search(init_routes, node_daily_loads, final_matrix)
            
            day_dist = sum([calculate_route_cost(r, final_matrix) for r in opt_routes])
            weekly_dist += day_dist
            weekly_trucks += len(opt_routes)
            
            for truck_idx, route in enumerate(opt_routes):
                real_route_path = [str(idx_to_real_id[n]) for n in route]
                route_str = " -> ".join(real_route_path)
                route_load = sum([node_daily_loads.get(n, 0) for n in route if n != 0])
                route_km = calculate_route_cost(route, final_matrix)
                
                detailed_routes.append({
                    "Week": week_str,
                    "Day": day_names[day_idx],
                    "Truck_No": truck_idx + 1,
                    "Total_Load_m3": round(route_load, 2),
                    "Usage_%": round((route_load / VEHICLE_CAPACITY) * 100, 1),
                    "Distance_km": round(route_km, 2),
                    "Stops": len(route) - 2,
                    "Route_Path": route_str
                })
        
        summary_results.append({
            "Week": week_str,
            "Total_Distance_km": round(weekly_dist, 2),
            "Total_Trucks": weekly_trucks
        })
        
    output_filename = 'PVRP_Optimization_Final_Report.xlsx'
    print(f"\n\nSaving to {output_filename}...")
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        pd.DataFrame(summary_results).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(detailed_routes).to_excel(writer, sheet_name='Details', index=False)
        
    print(f"✅ DONE! Execution Time: {time.time() - start_time:.1f} sec")

if __name__ == "__main__":
    main()

