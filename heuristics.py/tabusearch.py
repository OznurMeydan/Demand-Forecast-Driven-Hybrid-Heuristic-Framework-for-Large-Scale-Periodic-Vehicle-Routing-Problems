Input: N (Nodes), D (Distance Matrix), Q_week (Weekly Volumes), C (Vehicle Capacity)
Output: S_best (Optimized Weekly Schedule)

1:  For each week w do:
2:      For each node i in N do:
3:          freq, load = CalculateFrequencyAndLoad(Q_week_i, C)
4:          Days_i = AssignVisitDays(freq)
5:      End For
6:
7:      For each day d in {Monday, ..., Friday} do:
8:          N_d = {nodes assigned to day d}
9:          S_initial = ClarkeWrightSavings(N_d, D, C)
10:         S_day = TabuSearchImprovement(S_initial, D, C)
11:         S_best = S_best + S_day
12:     End For
13: End For
14: Return S_best

Algorithm 2: TabuSearchImprovement(S, D, C)

1:  S_curr = Apply2OptToAllRoutes(S, D)
2:  S_best = S_curr
3:  TabuList = Empty
4:
5:  For k = 1 to MaxIterations do:
6:      Candidates = Empty
7:      For n = 1 to NeighborsToCheck do:
8:          move = RandomlyPick(Relocate, Swap)
9:          S_cand = ApplyMove(S_curr, move)
10:
11:         If IsFeasible(S_cand, C) is True then:
12:             Candidates = Candidates + {(S_cand, move)}
13:         End If
14:     End For
15:
16:     Sort Candidates by Cost(S_cand) ascending
17:     For each (S_cand, move) in Candidates do:
18:         IsTabu = (move is in TabuList)
19:         IsAspiration = (Cost(S_cand) < Cost(S_best))
20:
21:         If (Not IsTabu) OR IsAspiration then:
22:             S_curr = S_cand
23:             UpdateTabuList(move, Tenure)
24:             If Cost(S_curr) < Cost(S_best) then:
25:                 S_best = S_curr
26:             End If
27:             Break // Best admissible move found, exit candidate loop
28:         End If
29:     End For
30: End For
31:
32: Return Apply2OptToAllRoutes(S_best, D)
