import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, Integers, minimize, value, Reals, Binary
import pandas as pd
import matplotlib.pyplot as plt
import cplex
print(cplex.__version__)
# Initial arrays and constants
t = np.zeros(48)
wind = np.zeros(48)
wpow = np.zeros(48)
vci = 3
vr = 7
vco = 12
pwr = 15000

# Wind power calculation
for i in range(len(wind)):
    t[i] = i + 1
    wind[i] = 0  # Simplified - original data reading commented out
    if wind[i] <= vci:
        wpow[i] = 0
    elif wind[i] <= vr:
        wpow[i] = pwr * ((wind[i]**3 - vci**3)/(vr**3 - vci**3))
    elif wind[i] <= vco:
        wpow[i] = pwr
    else:
        wpow[i] = 0

avgwind = np.mean(wind)

# Constants
tl = len(t)
il = 3
rho = [60, 0.8, 2.12]
mmin = [9.99, 0, 750]
mmax = [33.3, 2250, 1100]
nemax = 15
msmin = [4035, 22601, 0]
msmax = [807197, 4.52e6, 0]
mnh3tar = 1000
rmin = 100
rmax = 40
rampt = 4
damax = 1
ceb = np.zeros(48)
ces = np.zeros(48)
gridemm = np.zeros(48)

# Load data (modify path as needed)
df1 = pd.read_csv("G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/US-NE-ISNE_2024_hourly.csv")
# ceb_total = df1["power price"].values[392:536]
# gridemm_total = df1["CI kgCO2/kwh"].values[392:536]
ceb_total = df1["power price"].values[320:540+47]
gridemm_total = df1["CI kgCO2/kwh"].values[360:540+47]
movhor_tl = len(ceb_total)-47

# More variables
mnh3 = np.zeros(movhor_tl)
mh2 = np.zeros(movhor_tl)
mn2 = np.zeros(movhor_tl)
mh2b = np.zeros(movhor_tl)
totalcost = np.zeros(movhor_tl)
totalemms = np.zeros(movhor_tl)
cnh3 = 1.46
con = 50
cel = 0.42
cpsa = 0.10
csh2 = 5.50
csn2 = 0.14
cbh2 = 2.30
h2perh = 100
nh3_tot_min = 45000
ces=ceb-0.03
h2fossil = 9.3
el_start_risk = 5
el_op_risk = 1
reac_op_risk = 1
reac_ramp_risk = 2.5
stor_n2_risk = 1
stor_h2_risk = 1.5

# Initial decision variables
da_ini = np.zeros(tl + 3)
de_ini = np.zeros(tl)
ms_ini = np.zeros((il, tl + 1))
mp_ini = np.zeros((il, tl + 1))
mnh3dev_ini = np.zeros(tl)
ne_ini = np.zeros(tl + 1)
peb_ini = np.zeros(tl)
pei_ini = np.zeros((il, tl))
pes_ini = np.zeros(tl)
u_ini = np.zeros(il)
bh2_ini = np.zeros(tl)

mi0 = msmin.copy()
ne0 = 0
mnh30 = mnh3tar
da0 = [0, 0, 0]
times = np.concatenate(([0], t))

loop =movhor_tl
# Main optimization loop
for i in range(loop):
    ceb = ceb_total[i:i+48]
    gridemm = gridemm_total[i:i+48]
    ces = ceb - 0.03

    # Define optimization model
    nh3w = ConcreteModel()

    # Index sets
    nh3w.T = range(1, tl+1)
    nh3w.T0 = range(0, tl+1)
    nh3w.IL = range(1, il+1)
    nh3w.DA = range(-2, tl+1)

    # Variables
    nh3w.da = Var(nh3w.DA, domain=Binary, initialize=lambda m, j: da_ini[j+2])
    nh3w.de = Var(nh3w.T, domain=Integers, bounds=(0, None), initialize=lambda m, j: de_ini[j-1])
    nh3w.ms = Var(nh3w.IL, nh3w.T0, domain=NonNegativeReals, initialize=lambda m, i, t: ms_ini[i-1,t])
    nh3w.mp = Var(nh3w.IL, nh3w.T0, domain=NonNegativeReals, initialize=lambda m, i, t: mp_ini[i-1,t])
    nh3w.mnh3dev = Var(nh3w.T, within=Reals, initialize=lambda m, j: mnh3dev_ini[j-1])
    nh3w.ne = Var(nh3w.T0, domain=Integers, initialize=lambda m, j: ne_ini[j])
    nh3w.peb = Var(nh3w.T, domain=NonNegativeReals, initialize=lambda m, j: peb_ini[j-1])
    nh3w.pei = Var(nh3w.IL, nh3w.T, domain=NonNegativeReals, initialize=lambda m, i, j: pei_ini[i-1,j-1])
    nh3w.pes = Var(nh3w.T, domain=NonNegativeReals, initialize=lambda m, j: pes_ini[j-1])
    nh3w.u = Var(nh3w.IL, domain=NonNegativeReals, initialize=lambda m, i: u_ini[i-1])
    nh3w.bh2 = Var(nh3w.T, domain=NonNegativeReals, initialize=lambda m, j: bh2_ini[j-1])

    # Constraints
    def energybal_rule(m, j):
        return wpow[j-1] + m.peb[j] - m.pes[j] - m.pei[1,j] - m.pei[2,j] - m.pei[3,j] == 0
    nh3w.energybal = Constraint(nh3w.T, rule=energybal_rule)

    def energyunt_rule(m, i, j):
        return m.pei[i,j] == rho[i-1] * m.mp[i,j]
    nh3w.energyunt = Constraint(nh3w.IL, nh3w.T, rule=energyunt_rule)

    def minflow_rule(m, i, j):
        return m.mp[i,j] >= mmin[i-1] if i > 1 else Constraint.Skip
    nh3w.minflow = Constraint(nh3w.IL, nh3w.T, rule=minflow_rule)

    def maxflow_rule(m, i, j):
        return m.mp[i,j] <= mmax[i-1] if i > 1 else Constraint.Skip
    nh3w.maxflow = Constraint(nh3w.IL, nh3w.T, rule=maxflow_rule)

    nh3w.minelec = Constraint(nh3w.T, rule=lambda m, j: m.ne[j] >= 0)
    nh3w.maxelec = Constraint(nh3w.T, rule=lambda m, j: m.ne[j] <= nemax)
    nh3w.minh2pr = Constraint(nh3w.T, rule=lambda m, j: mmin[0]*m.ne[j] <= m.mp[1,j])
    nh3w.maxh2pr = Constraint(nh3w.T, rule=lambda m, j: m.mp[1,j] <= mmax[0]*m.ne[j])
    nh3w.minstor = Constraint(nh3w.IL, nh3w.T, rule=lambda m, i, j: m.ms[i,j] >= msmin[i-1])
    nh3w.maxstor = Constraint(nh3w.IL, nh3w.T, rule=lambda m, i, j: m.ms[i,j] <= msmax[i-1])
    nh3w.deviation = Constraint(nh3w.T, rule=lambda m, j: m.mnh3dev[j] == mnh3tar - m.mp[3,j])
    nh3w.maxbh2 = Constraint(nh3w.T, rule=lambda m, j: m.bh2[j] <= h2perh)
    nh3w.nh3min = Constraint(rule=lambda m: sum(m.mp[3,j] for j in m.T) >= nh3_tot_min)

    def h2stor_rule(m, j):
        return m.ms[1,j] == m.ms[1,j-1] + m.bh2[j] + m.mp[1,j] - (3/17)*m.mp[3,j]
    nh3w.h2stor = Constraint(nh3w.T, rule=h2stor_rule)

    def n2stor_rule(m, j):
        return m.ms[2,j] == m.ms[2,j-1] + m.mp[2,j] - (14/17)*m.mp[3,j]
    nh3w.n2stor = Constraint(nh3w.T, rule=n2stor_rule)

    nh3w.totalstor = Constraint(nh3w.IL, rule=lambda m, i: m.u[i] == m.ms[i,0] - m.ms[i,tl])
    nh3w.elecon = Constraint(nh3w.T, rule=lambda m, j: m.de[j] >= m.ne[j] - m.ne[j-1])
    nh3w.rampmin = Constraint(nh3w.T, rule=lambda m, j: -m.da[j]*rmin <= m.mp[3,j] - m.mp[3,j-1])
    nh3w.rampmax = Constraint(nh3w.T, rule=lambda m, j: m.mp[3,j] - m.mp[3,j-1] <= m.da[j]*rmax)
    
    def ramplim_rule(m, j):
        return sum(m.da[k] for k in range(max(-2, j+1-rampt), j+1)) <= damax
    nh3w.ramplim = Constraint(nh3w.T, rule=ramplim_rule)

    nh3w.ms_t0 = Constraint(nh3w.IL, rule=lambda m, i: m.ms[i,0] == mi0[i-1])
    nh3w.ne_t0 = Constraint(rule=lambda m: m.ne[0] == ne0)
    nh3w.mnh3_t0 = Constraint(rule=lambda m: m.mp[3,0] == mnh30)
    nh3w.da_t = Constraint(range(-2, 1), rule=lambda m, j: m.da[j] == da0[j+2])

    # Objective
    # def costs_rule(m):
    #     return sum(ceb[j-1]*m.peb[j] - ces[j-1] * m.pes[j] + \
    #                cnh3*m.mnh3dev[j] + con*m.de[j] + cel*m.mp[1,j]+cbh2*m.bh2[j] +\
    #                 cpsa*m.mp[2,j]  for j in m.T) +csh2*m.u[1] + csn2*m.u[2]
    def costs_rule(m):
        return sum(sum(ceb[j-1]*m.mp[1+i,j]*rho[i] for i in range(3)) + \
                    cel*m.mp[1,j]+cbh2*m.bh2[j] + cpsa*m.mp[2,j]  for j in m.T)
    def emms_rule(m):
        return sum(rho[j-1]*m.mp[j,i]*gridemm[i-1] for j in m.IL for i in m.T) + \
           sum(h2fossil*m.bh2[i] for i in m.T)
    def wateruse_rule(m):
        return sum(m.mp[1,j]*9.0 + m.bh2[j]*4.5 for j in m.T)
    def safety_rule(m):
        return sum(el_op_risk*m.mp[1,i]/mmax[0] for i in m.T)
    def Pareto_rule(m):
        return 0.55*costs_rule(m)+0.45*(emms_rule(m)+wateruse_rule(m)+safety_rule(m)) ### make them more balanced
        # nh3w.obj = Objective(rule=emms_rule, sense=minimize)
    nh3w.obj = Objective(rule=emms_rule, sense=minimize)
    # nh3w.obj = Objective(rule=costs_rule, sense=minimize)
    # Solve (using GLPK - install with 'conda install -c conda-forge glpk' or use another solver)
    from pyomo.opt import SolverFactory
    solver = SolverFactory('cplex')
    results = solver.solve(nh3w)

    # Calculate additional expressions for output
    costss = value(costs_rule(nh3w))
    
    emms = sum(rho[j-1]*value(nh3w.mp[j,i])*gridemm[i-1] for j in nh3w.IL for i in nh3w.T) + \
           sum(h2fossil*value(nh3w.bh2[i]) for i in nh3w.T)
    watuse = sum(value(nh3w.mp[1,j])*9.0 + value(nh3w.bh2[j])*4.5 for j in nh3w.T)
    safetyelectro = sum(el_op_risk*value(nh3w.mp[1,i])/mmax[0] for i in nh3w.T)
    
    # objvals = [value(nh3w.costs), emms, watuse, safetyelectro]
    objvals = [costss, emms, watuse, safetyelectro]
    print(objvals)

    # moving horizon modification:
    for k in range(il):  # il is 3 in your context, so range(il) is 0 to 2
        mi0[k] = value(nh3w.ms[k + 1, 1])  # i + 1 because Pyomo indices start at 1 for nh3w.ms
    # Update the number of electrons consumed in the system for the next iteration
    ne0 = value(nh3w.ne[1])
    # Update the ammonia produced for the next iteration, used to track the flow of ammonia produced
    mnh30 = value(nh3w.mp[3, 1])
    # Update the ramping decision variables for the next iteration, used to track ramping in the system
    for k in range(3):  # 1:3 in Julia becomes 0:2 in Python
        da0[k] = value(nh3w.da[k - 2])  # Adjust index: i-3 in Julia (1-3=-2) becomes i-2 in Python (0-2=-2)


    # # Store results
    # if i != loop-1:
    #     mnh3[i] = value(nh3w.mp[3,1])
    #     mh2[i] = value(nh3w.mp[1,1])
    #     mn2[i] = value(nh3w.mp[2,1])
    #     mh2b[i] = value(nh3w.bh2[1])
    # else:
    #     for j in range(1,tl+1):  # Ensure we capture the final time step results for all variables
    #         mnh3[i+j-1] = value(nh3w.mp[3,j])
    #         mh2[i+j-1] = value(nh3w.mp[1,j])
    #         mn2[i+j-1] = value(nh3w.mp[2,j])
    #         mh2b[i+j-1] = value(nh3w.bh2[j])
        
    #     print("Final Time Step Results:",mnh3[i:], mh2[i:], mn2[i:], mh2b[i:])
    mnh3[i] = value(nh3w.mp[3,1])
    mh2[i] = value(nh3w.mp[1,1])
    mn2[i] = value(nh3w.mp[2,1])
    mh2b[i] = value(nh3w.bh2[1])
    #######materials storage#######
    # mnh3[i] = value(nh3w.ms[3,1])
    # mh2[i] = value(nh3w.ms[1,1])
    # mn2[i] = value(nh3w.ms[2,1])
    # mh2b[i] = value(nh3w.bh2[1])
    totalcost[i] = sum(ceb[0]*value(nh3w.mp[1+i,1])*rho[i] for i in range(3)) + \
                    cel*value(nh3w.mp[1,1])+cbh2*value(nh3w.bh2[1]) + cpsa*value(nh3w.mp[2,1]) 
    totalemms[i] = sum(rho[j-1]*value(nh3w.mp[j,1])*gridemm[0] for j in nh3w.IL)+ \
    h2fossil*value(nh3w.bh2[1])

    # Update initial values


# Plotting
plt.figure(figsize=(16, 5))
x = range(1, movhor_tl+1)
plt.plot(x, mnh3, label="Ammonia Produced")
plt.plot(x, mh2, label="Hydrogen Produced")
plt.plot(x, mn2, label="Nitrogen Produced")
plt.plot(x, mh2b, label="Hydrogen Bought")
plt.xlabel("Time, h")
plt.ylabel("Material Flow, kg/h")
plt.legend()
plt.show()

import matplotlib.gridspec as gridspec

# Create a figure with two subplots: one larger and one smaller
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 ratio for the two subplots

# Main plot (top subplot)
ax1 = plt.subplot(gs[0])
x = range(1, movhor_tl+1)
ax1.plot(x, ObjectiveNumber[:,0], label="Costs", color="blue")
ax1.plot(x, ObjectiveNumber[:,1], label="Emissions", color="green")
ax1.plot(x, ObjectiveNumber[:,2], label="Water Use", color="orange")
ax1.plot(x, ObjectiveNumber[:,3], label="Safety", color="purple")
ax1.axvspan(0, 121, color=groupingcolor, alpha=groupingalpha)
ax1.axvspan(144, 178, color=groupingcolor, alpha=groupingalpha)
ax1.set_xlabel("Time, h")
ax1.set_ylabel("Objective Metrics")
ax1.legend(loc="upper left")

# Secondary y-axis for totalcost and totalemms
ax2 = ax1.twinx()
ax2.plot(x, totalcost, label="Cost per Step", color="red", linestyle="--")
ax2.plot(x, totalemms, label="Emms per Step", color="brown", linestyle="--")
ax2.set_ylabel("Cost and Emissions")
ax2.legend(loc="upper right")

ax1.set_title("Objective Metrics and Cost/Emissions per Step")

# Smaller subplot (bottom)
ax3 = plt.subplot(gs[1])
time_range = range(1, movhor_tl+1)
ax3.plot(time_range, df1["power price"].values[320:540], label="Power Price", color="blue")
ax3.plot(time_range, df1["CI kgCO2/kwh"].values[320:540], label="CI kgCO2/kWh", color="green")
ax3.axvspan(0, 121, color=groupingcolor, alpha=groupingalpha)
ax3.axvspan(144, 178, color=groupingcolor, alpha=groupingalpha)
ax3.set_xlabel("Time, h")
ax3.set_ylabel("Values")
ax3.legend(loc="upper left")
ax3.set_title("Power Price and Carbon Intensity")

# Adjust layout
plt.tight_layout()
plt.show()


totalnh3 = sum(mnh3)
print(f"{totalnh3} ammonia produced")
print(f"{sum(totalcost)} total cost in the scheduling horizon")
print(f"{sum(totalemms)} total emissions in the scheduling horizon")
print(f"{sum(totalcost)/totalnh3} cost per ammonia in the scheduling horizon")
print(f"{sum(totalemms)/totalnh3} total emissions in the scheduling horizon")
# print("total emission grid: ", totalemms)