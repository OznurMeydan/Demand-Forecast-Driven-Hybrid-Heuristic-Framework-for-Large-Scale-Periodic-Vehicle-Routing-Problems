# Demand-Forecast-Driven Hybrid Heuristic Framework for Large-Scale Periodic Vehicle Routing Problems

This repository contains the **demand forecasting and model selection layer** of a hybrid framework that feeds a Periodic Vehicle Routing Problem (P-VRP) with statistically robust and operationally safe weekly demand estimates for each dealer.

The methodology is designed for a **long-tail spare parts logistics network** with intermittent demand, heterogeneous dealer profiles, and strict service-level constraints. :contentReference[oaicite:0]{index=0}  

> In short:  
> - We segment dealers by **volume** and **volatility**,  
> - benchmark all candidate models using **MASE / WMAPE / BIAS**,  
> - derive **segment-specific performance targets** via internal benchmarking,  
> - and finally select, per dealer, the model that is both **accurate (low WMAPE)** and **operationally safe (positive BIAS preferred)**.

---

## 1. Methodology Overview

The forecasting layer follows the methodological criteria described in the accompanying report: :contentReference[oaicite:1]{index=1}  

### 1.1 Data eligibility and horizon

- Only dealers with at least **~2 years of history (≥ 722 days)** are included, to allow reliable estimation of yearly seasonality and to avoid overfitting.
- Training period: **01 July 2023 – 27 June 2025** (≈ 24 months, two full annual cycles).
- Test period: **28 June 2025 – 31 October 2025** (≈ 4 months / 18 weeks), aligned with a tactical planning horizon.

### 1.2 Temporal aggregation

- Raw daily shipments are aggregated to **weekly** time series (Monday-based), following ADIDA-style aggregation for intermittent demand.  
- This reduces noise and stabilizes the error metrics without losing the signal relevant for weekly route planning.

### 1.3 Two-dimensional segmentation

To reflect the “Long Tail” structure of the dealer network, a **2-D segmentation** is applied:

1. **Volume-based segmentation (ABCD)**
   - A (VIP): top 10% dealers by total volume (≥ 90th percentile)  
   - B (High): 70–90th percentile  
   - C (Standard): 40–70th percentile  
   - D (Micro): bottom 40%

2. **Difficulty segmentation (volatility)**
   - CV (coefficient of variation) is computed on the **active period** of each dealer.
   - Dealers with CV below the **median CV** are labelled **“Stable (Easy)”**; those above are **“Volatile (Hard)”**.

Each dealer is therefore assigned to a unique **(Volume_Segment, Difficulty)** pair, e.g. `C-Volatile`.

### 1.4 Error metrics

Instead of R², the following metrics are used, as recommended in intermittent demand and logistics literature:

- **MASE** – Model quality vs. naive benchmark  
- **WMAPE** – Main accuracy metric (Weighted MAPE, volume-weighted)  
- **BIAS** – Directional risk indicator (under- vs over-forecasting)

### 1.5 Internal benchmarking and target setting

Performance targets are not imposed externally but derived from the **best achievable performance inside each segment**:

- For each dealer, across all candidate models, we store the **minimum WMAPE** (“Best_WMAPE”).
- For each (Volume_Segment, Difficulty) group, we compute:
  - **Q1 (25th percentile)** of Best_WMAPE → `EXCELLENT_TARGET (Q1)`  
  - **Q3 (75th percentile)** of Best_WMAPE → upper limit of **“normal range”** and entry to **“risk zone”**  
- Target bands:
  - **EXCELLENT**: WMAPE < Q1  
  - **NORMAL**: Q1 ≤ WMAPE ≤ Q3  
  - **RISKY**: WMAPE > Q3  

This yields **segment-specific realistic targets** (e.g. A-Stable may have 15%, while D-Volatile may realistically be 45–50%).

### 1.6 Hybrid model selection logic (MASE → WMAPE → BIAS)

For each dealer and each candidate model, we compute MASE / WMAPE / BIAS, then apply a **cost-sensitive selection algorithm**:

1. **Quality filter (MASE)**
   - Models with **MASE > 1.2** are considered low-quality and removed.
   - If no model survives, all models are kept but flagged as “Low Quality (MASE>1.2)”.

2. **Accuracy ranking (WMAPE)**
   - Remaining models are sorted by **ascending WMAPE**.
   - The lowest-WMAPE model becomes the **initial champion**.

3. **Risk management (BIAS-based swap)**
   - If the champion has **negative BIAS** (systematic under-forecasting)  
   - and the second-best model has **positive BIAS** (slight over-forecasting)  
   - and the WMAPE gap between them is **< 5 percentage points**,  
   → then the second-best model is selected instead.  

This implements an **asymmetric loss function**: under-forecasting (stock-out, lost sales, load infeasibility) is penalised more heavily than mild over-forecasting (extra inventory, spare capacity).

The final weekly demand forecasts are then fed into the **P-VRP heuristic layer** as capacity requirements for each dealer and time period.

---

## 2. Repository Structure (forecasting layer)

A suggested structure for the forecasting component is:

```text
forecasting/
  ├── mature_dealers_segmentation_targets.py   # script 1: segmentation + target table
  ├── final_model_selection.py                 # script 2: model selection & performance status
  └── ...
data/
  ├── raw/                                     # XLSX files from Renault MAIS (not in repo)
  └── processed/
results/
  ├── mature_dealers/
  └── final_selection/
## 6. Hybrid Metaheuristic Optimization Layer (Tabu Search & P-HGA)

While the forecasting layer provides statistically robust weekly demand estimates per dealer, the optimization layer is responsible for transforming these forecasts into **operationally feasible periodic routes** under vehicle capacity and visit-frequency constraints.  

Two complementary hybrid metaheuristics are implemented:

- a **Hybrid Tabu Search (HTS)** for intensive local improvement of daily routes, and  
- a **Periodic Hybrid Genetic Algorithm (P-HGA)** for global exploration of periodic day assignments and route structures.

Both algorithms use the **forecast-driven weekly volumes** as input and operate on a **Periodic Vehicle Routing Problem (PVRP)** formulation over a multi-week planning horizon.

---

### 6.1 Problem Setting and Decomposition

- Planning horizon: multi-week, with operational decisions made at the **day level** (Mon–Fri).
- Vehicle capacity: **Q = 16.35 m³**, which is treated as a hard volumetric constraint.
- For each dealer *i* and week *w*:
  - A target **visit frequency** `f_i` is computed based on the forecasted weekly volume (`Q_week_i`) and vehicle capacity.
  - Dealers are assigned to specific weekdays (e.g., single-frequency dealers → Wednesdays) to balance workload.
- The periodic problem is decomposed into:
  1. **Day assignment decisions** (which days each dealer is visited), and  
  2. **Daily routing decisions** (customer sequencing per day).

The forecasting layer thus feeds the optimization layer with **(dealer, week, volume, frequency)** tuples, which are then converted into concrete route plans.

---

### 6.2 Hybrid Tabu Search (HTS)

The Hybrid Tabu Search is designed to **intensify** and improve daily routes derived from a constructive heuristic, rather than exploring the full solution space from scratch.

#### 6.2.1 High-Level Pipeline

The HTS framework follows a two-stage structure (Algorithm 01 & 02 in the report):

1. **Weekly and daily decomposition**
   - For each week *w*:
     - For each dealer *i*:
       - Compute `(freq_i, load_i)` from weekly volumes and capacity.
       - Assign visit days `Days_i` according to a rule-based schedule (e.g., 1-visit dealers → Wednesday).
   - For each day *d* in `{Monday, …, Friday}`:
     - Construct the daily node set `N_d` from dealers assigned to *d*.

2. **Constructive initialization (Clarke–Wright Savings)**
   - Run a **Clarke–Wright Savings Heuristic** on each `N_d` to obtain an initial set of feasible routes:
     - Start from one route per dealer (`0 → i → 0`).
     - Compute savings for all pairs and merge routes greedily, subject to capacity feasibility.

3. **Pre-processing with 2-opt**
   - Apply **2-opt** to each initial route set to remove edge crossings and quickly move toward a local optimum.

4. **Tabu Search improvement**
   - For each day *d*, run Tabu Search for a fixed number of iterations (e.g., 1,000):
     - Generate neighboring solutions using:
       - **Relocate** moves (move a dealer to a new position/route),
       - **Swap** moves (exchange dealers between routes).
     - Check **feasibility** (capacity constraint) for each candidate.
     - Maintain a **Tabu List** to prevent cycling back to recently applied moves.
     - Use an **aspiration criterion**: a tabu move is allowed if it improves the best solution found so far.

5. **Final refinement**
   - After Tabu iterations, apply a final 2-opt pass to `S_best` to ensure locally optimized routes.

The resulting `S_best` represents the **optimized weekly schedule** composed of the best daily routes across the entire planning horizon.

#### 6.2.2 Implementation Highlights and Parameters

- **Move operators**: Relocate and Swap (for both intra-route and inter-route improvements).
- **Local search**: 2-opt applied:
  - once before Tabu Search (to clean up initial routes), and  
  - once after (to polish the final solution).
- **Tabu mechanics**:
  - Tabu list stores recently applied moves.
  - **Tabu tenure** (e.g. 20 iterations) controls how long a move remains tabu.
  - Aspiration allows overriding tabu status for globally improving solutions.
- **Neighborhood sampling**:
  - In each iteration, a limited number of neighbors (e.g. up to 100) are evaluated for efficiency.
- **Hard feasibility checks**:
  - Every candidate route set is checked against capacity `Q = 16.35 m³`; infeasible candidates are discarded.

#### 6.2.3 Observed Behavior under Volume-Constrained Conditions

During the 51-week analysis, many daily solutions returned by HTS were **identical or very close** to the initial Clarke–Wright solutions. Rather than indicating poor search behavior, this is a direct consequence of the **Full Truckload (FTL)** environment:

- Vehicles frequently operate at **>90% volume utilization**, leaving almost no residual capacity for Relocate or Swap moves without violating capacity.
- A large share of routes are **single-stop routes** (one dealer nearly fills the entire truck). For these cases, the optimal solution is trivially a depot–dealer–depot shuttle, and no further improvement is possible.
- For multi-stop routes, Clarke–Wright already produces strong geographic clusters; HTS confirms that many of these are near-optimal given capacity saturation.

This convergence between constructive and improvement phases is therefore interpreted as evidence that:

- The **distribution network is operating near a high-efficiency plateau**, and  
- The hybrid approach successfully validates and preserves this efficient equilibrium.

---

### 6.3 Periodic Hybrid Genetic Algorithm (P-HGA)

The Periodic Hybrid Genetic Algorithm complements HTS by providing a **population-based global search** mechanism that explores different day assignments and route structures, while still leveraging domain-specific heuristics and local search.

#### 6.3.1 Design Rationale

- Genetic Algorithms (GA) are well-suited for **multimodal** and **constraint-rich** search spaces, such as PVRP.
- Pure GA, however, faces:
  - slow convergence when initialized randomly, and  
  - difficulty maintaining feasibility under capacity + periodic visit constraints.
- The proposed **P-HGA** addresses these issues by:
  - initializing part of the population using **Clarke–Wright**,  
  - enforcing visit-frequency and capacity constraints via a **repair mechanism**, and  
  - applying **2-opt local search** to refine each offspring.

The approach is inspired by classical hybrid GA frameworks from the PVRP/MDPVRP literature (e.g., Ochelska-Mierzejewska et al., Vidal, Taillard).

#### 6.3.2 Algorithmic Structure (Algorithms 0, 1, 2)

**Algorithm 0 – Population Initialization**

- Inputs:
  - Node set `N`, distance matrix `D`, vehicle capacity `Q = 16.35`.
- Mixed initialization strategy:
  - A fraction `ρ = 0.3` of the population is built using the **Clarke–Wright Savings Heuristic**:
    - produces structured, geographically coherent routes.
  - The remaining `1 − ρ` individuals are generated **randomly**:
    - ensures diversity and broader coverage of the search space.
- Output:
  - Initial population `P`, which seeds the evolutionary process.

**Algorithm 1 – Clarke & Wright Savings Heuristic**

- Starts from one route per customer (`0 → i → 0`).
- Computes savings for all pairs (i, j) and sorts them in descending order.
- Iteratively merges routes if the merge:
  - yields positive savings, and  
  - does not violate capacity.
- In this project, the classical C&W is extended via a `cw_seed_assignment` function that:
  - incorporates the **periodic structure** by assigning dealers to service days,  
  - while preserving the high-quality spatial clusters produced by C&W.

**Algorithm 2 – Periodic Hybrid Genetic Algorithm**

- Inputs:
  - Distance matrix, demand forecasts, visit frequencies `f_i`, vehicle capacity, planning horizon (Mon–Fri).
- Main steps:
  1. **InitializePopulation** using Algorithm 0 (hybrid CW + random).
  2. Evaluate fitness of each individual:
     - `Fitness(s) = TotalDistance(s)`.
  3. Keep track of the **BestFoundSolution**.
  4. While stopping criteria not met (e.g. max 150 iterations):
     - Select parents using **Tournament Selection** (T = 2).
     - Apply **crossover** to recombine day assignments and routing structures.
     - Apply **mutation** with a controlled probability.
     - Perform `SplitAndRouteConstruction`:
       - enforce visit frequencies (`f_i` days per week),
       - build daily capacity-feasible routes.
     - Apply **2-opt Local Search** to each day’s routes.
     - Repair individuals that violate capacity or visit-frequency constraints.
     - Re-evaluate fitness and update the population (elitism).

#### 6.3.3 Parameter Settings and Sensitivity

Key parameter settings used in the implementation:

- Population size: **40**
- Maximum iterations: **150**
- CW-based initialization share: **ρ = 0.3** (30%)
- Selection: **Tournament Selection** with T = 2
- Mutation rate:
  - Tested values: 0.10 and 0.05
  - Final choice: **0.05**, as higher mutation (0.10) degraded solution quality.
- Local search: **2-opt** applied to every individual after construction.
- Robust demand modeling:
  - Effective load per dealer is computed using `mean + 0.5 * σ` to protect against demand uncertainty.
- Capacity constraint: `Q = 16.35` m³ strictly enforced.

Sensitivity analysis across population sizes and mutation rates shows that:

- Population sizes of **40** and **100** outperform smaller size (20),  
- but **Pop = 40, Mut = 5%** provides the best trade-off between solution quality and runtime, achieving costs around **6.83 × 10⁴ km** in the experiments.
- Excessive mutation disrupts good genetic structures and harms the exploration–exploitation balance.

The resulting cost curves are nearly flat from the first generation, indicating that:

- CW-based initialization provides **high-quality warm starts**, and  
- 2-opt local search quickly pushes individuals to strong local optima, after which the GA primarily maintains and slightly refines these elite solutions.

---

### 6.4 Summary of the Hybrid Heuristic Layer

- **Hybrid Tabu Search**:
  - Focuses on **daily local intensification**, confirming and slightly improving routes generated by Clarke–Wright under tight capacity constraints.
- **Periodic Hybrid Genetic Algorithm**:
  - Provides **global exploration** over day assignments and route structures, guided by CW seeding and 2-opt local refinement.
- Together with the **demand-forecast-driven input**, these methods form a coherent, industry-ready framework capable of solving a fully periodic, capacity-constrained VRP with real-world spare parts data.

In combination, the forecasting and hybrid metaheuristic layers support a **data-driven decision-support system** for Renault MAIS that can be extended or reused for other long-tail logistics networks with similar characteristics.
