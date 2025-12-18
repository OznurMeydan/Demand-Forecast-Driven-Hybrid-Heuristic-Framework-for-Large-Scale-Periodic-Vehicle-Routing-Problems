import pandas as pd
import numpy as np
import random
import os
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt


Q = 16.35
LAMBDA = 0.5
DAYS = [0,1,2,3,4]   # Mon–Fri
DEPOT = 0
POP_SIZE = 40
MAX_ITER = 150
RHO = 0.3
TOURNAMENT_SIZE = 2


df = pd.read_excel("GELECEK_51_HAFTA_TAHMINLERI.xlsx")
dist_df = pd.read_excel("Mesafe_Matris_123_Bayi.xlsx", index_col=0)


dist_df.index = dist_df.index.map(lambda x: 0 if x=="MERKEZ_DEPO" else int(x))
dist_df.columns = dist_df.columns.map(lambda x: 0 if x=="MERKEZ_DEPO" else int(x))
dist_df = dist_df.applymap(lambda x: float(str(x).replace(",", ".")))
D = dist_df.to_dict()


results = []
for bayi, grp in df.groupby("BAYI_KODU"):
    weekly = grp["TAHMIN_MIKTAR"].values
    mean_d = weekly.mean()
    std_d = weekly.std(ddof=0)
    robust_d = mean_d + LAMBDA * std_d
    freq = max(1, min(5, int(np.ceil(robust_d / Q))))
    results.append({"BAYI_KODU": int(bayi), "ROBUST_WEEKLY": robust_d, "VISIT_FREQ": freq})


freq_df = pd.DataFrame(results)
visit_freq = dict(zip(freq_df.BAYI_KODU, freq_df.VISIT_FREQ))
robust_daily_demand = {r.BAYI_KODU: r.ROBUST_WEEKLY / r.VISIT_FREQ for _, r in freq_df.iterrows()}


def split_routes(customers, demand):
    routes, load, current = [], 0, []
    for c in customers:
        d_val = demand.get(c, 0)
        if load + d_val <= Q:
            current.append(c)
            load += d_val
        else:
            if current: routes.append(current)
            current = [c]
            load = d_val
    if current: routes.append(current)
    return routes


def route_distance(route):
    if not route: return 0
    d = D[DEPOT][route[0]]
    for i in range(len(route)-1):
        d += D[route[i]][route[i+1]]
    return d + D[route[-1]][DEPOT]


def two_opt(route):
    best = route[:]
    improved = True
    while improved:
        improved = False
        best_dist = route_distance(best)
        for i in range(len(best)-1):
            for j in range(i+1, len(best)):
                new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new) < best_dist:
                    best = new
                    improved = True
    return best


def build_solution(assign, demand):
    total = 0
    for d in DAYS:
        routes = split_routes(assign[d], demand)
        for r in routes:
            r_opt = two_opt(r)
            total += route_distance(r_opt)
    return total




def random_assignment():
    A = {d: [] for d in DAYS}
    for c, f in visit_freq.items():
        for d in random.sample(DAYS, f): A[d].append(c)
    return A


def cw_seed_assignment():
    A = {d: [] for d in DAYS}; assigned = Counter(); customers = list(visit_freq.keys())
    savings = []
    for i in customers:
        for j in customers:
            if i < j: savings.append((D[DEPOT][i] + D[DEPOT][j] - D[i][j], i, j))
    savings.sort(reverse=True)
    for _, i, j in savings:
        if assigned[i] < visit_freq[i] and assigned[j] < visit_freq[j]:
            d = random.choice(DAYS)
            if i not in A[d] and j not in A[d]:
                A[d] += [i, j]; assigned[i] += 1; assigned[j] += 1
    for c in customers:
        while assigned[c] < visit_freq[c]:
            d = random.choice(DAYS)
            if c not in A[d]: A[d].append(c); assigned[c] += 1
    return A


def repair(sol):
    count = Counter()
    for d in DAYS:
        for c in sol[d]: count[c] += 1
    for c in visit_freq:
        while count[c] > visit_freq[c]:
            d = random.choice([x for x in DAYS if c in sol[x]])
            sol[d].remove(c); count[c] -= 1
        while count[c] < visit_freq[c]:
            d = random.choice(DAYS)
            if c not in sol[d]: sol[d].append(c); count[c] += 1
    return sol




P = [cw_seed_assignment() if i < int(RHO*POP_SIZE) else random_assignment() for i in range(POP_SIZE)]
fitness_vals = [build_solution(p, robust_daily_demand) for p in P]
best_hist, avg_hist = [], []


print("GA Başlıyor...")
for it in range(MAX_ITER):
    best_hist.append(min(fitness_vals))
    avg_hist.append(np.mean(fitness_vals))
   
    idxs = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    p1 = deepcopy(P[min(idxs, key=lambda i: fitness_vals[i])])
    idxs = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    p2 = deepcopy(P[min(idxs, key=lambda i: fitness_vals[i])])
   
    child = repair({d: (p1 if random.random() < 0.5 else p2)[d][:] for d in DAYS})
    child_fit = build_solution(child, robust_daily_demand)
   
    worst = np.argmax(fitness_vals)
    if child_fit < fitness_vals[worst]:
        P[worst] = child
        fitness_vals[worst] = child_fit


best_assignment = deepcopy(P[np.argmin(fitness_vals)])
print("Optimizasyon Tamamlandı.")


MUTATION_RATE = 0.05


print("GA Başlıyor...")
for it in range(MAX_ITER):
    best_hist.append(min(fitness_vals))
    avg_hist.append(np.mean(fitness_vals))
   
    idxs = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    p1 = deepcopy(P[min(idxs, key=lambda i: fitness_vals[i])])
    idxs = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
    p2 = deepcopy(P[min(idxs, key=lambda i: fitness_vals[i])])
   
    child = repair({d: (p1 if random.random() < 0.5 else p2)[d][:] for d in DAYS})
   
    if random.random() < MUTATION_RATE:
       
        all_dealers = list(visit_freq.keys())
        target_dealer = random.choice(all_dealers)
       
        for d in DAYS:
            if target_dealer in child[d]:
                child[d].remove(target_dealer)
       
        new_days = random.sample(DAYS, visit_freq[target_dealer])
        for d in new_days:
            child[d].append(target_dealer)
       
        child = repair(child)


    child_fit = build_solution(child, robust_daily_demand)
   
    worst = np.argmax(fitness_vals)
    if child_fit < fitness_vals[worst]:
        P[worst] = child
        fitness_vals[worst] = child_fit


weekly_summary, route_rows = [], []


for week, week_df in df.groupby("TARIH"):
    weekly_demand = week_df.set_index("BAYI_KODU")["TAHMIN_MIKTAR"].to_dict()
    daily_demand = {c: weekly_demand.get(c, 0) / visit_freq[c] for c in visit_freq}
   
    week_cost = build_solution(best_assignment, daily_demand)
    week_trips = 0
   
    for d in DAYS:
        routes = split_routes(best_assignment[d], daily_demand)
        week_trips += len(routes)
        for r_id, r in enumerate(routes, 1):
            route_rows.append({
                "Hafta": week, "Gun": ["Mon","Tue","Wed","Thu","Fri"][d],
                "Rota_ID": r_id, "Rota": "0 → " + " → ".join(map(str,r)) + " → 0",
                "Yuk": round(sum(daily_demand.get(c, 0) for c in r), 2),
                "Mesafe_km": round(route_distance(two_opt(r)), 2)
            })
           
    weekly_summary.append({"Hafta": week, "Toplam_Mesafe_km": round(week_cost, 2), "Toplam_Kamyon_Seferi": week_trips})


ga_conv_df = pd.DataFrame({"Iteration": range(1, len(best_hist)+1), "Best_Fitness": best_hist, "Average_Fitness": avg_hist})
best_assign_df = pd.DataFrame([{"Day": ["Mon","Tue","Wed","Thu","Fri"][d], "Dealer": c} for d in DAYS for c in best_assignment[d]])
weekly_summary_df = pd.DataFrame(weekly_summary)
weekly_routes_df = pd.DataFrame(route_rows)


output_path = "PVRP_GA_RESULTS_FINAL_0.1.2.xlsx"


try:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        ga_conv_df.to_excel(writer, sheet_name="GA_Convergence", index=False)
        best_assign_df.to_excel(writer, sheet_name="Best_Assignment", index=False)
        weekly_summary_df.to_excel(writer, sheet_name="Weekly_Summary", index=False)
        weekly_routes_df.to_excel(writer, sheet_name="Weekly_Routes", index=False)
    print(f">>> Excel başarıyla oluşturuldu: {os.path.abspath(output_path)}")
except Exception as e:
    print(f"!!! HATA: Excel oluşturulamadı. Nedeni: {e}")


plt.plot(best_hist, label="Best"); plt.plot(avg_hist, label="Average"); plt.legend(); plt.show()




