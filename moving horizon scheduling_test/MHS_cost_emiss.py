import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, Integers, minimize, value, Reals, Binary
)
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.opt import SolverFactory
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import cplex

print(cplex.__version__)

solver = SolverFactory('cplex')

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
    wind[i] = 0
    if wind[i] <= vci:
        wpow[i] = 0
    elif wind[i] <= vr:
        wpow[i] = pwr * ((wind[i]**3 - vci**3) / (vr**3 - vci**3))
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

a = 200
b = 500

# Load data
df1 = pd.read_csv("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/2024_NewEngland_DA/US-NE-ISNE_2024_hourly.csv")

# df2 = pd.read_csv("G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/combining last nine months of New England 2024/Tuned Para Random_forest_last_9_2024_NewEngland_2_classification.xlsx")
df2 = pd.read_excel("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/2024_NewEngland_DA/combining last nine months of New England 2024/Tuned Para Random_forest_last_9_2024_NewEngland_2_classification.xlsx")


# df1 = pd.read_csv(
#     "G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/US-NE-ISNE_2024_hourly.csv"
# )

# df2 = pd.read_csv(
#     "G:/My Drive/AA gourp/machine learning/2024_NewEngland_DA/combining last nine months of New England 2024/lstm_200_two_class_predicted_vs_true_results.csv"
# )

ceb_total = df1["power price"].values[a:b+47]
gridemm_total = df1["CI kgCO2/kwh"].values[a:b+47]

predicted_labels = df2["Predicted_Label"].values[a:b]
true_labels = df2["True_Label"].values[a:b]

movhor_tl = len(ceb_total) - 47

# Result arrays
mnh3 = np.zeros(movhor_tl)
mh2 = np.zeros(movhor_tl)
mn2 = np.zeros(movhor_tl)
mh2b = np.zeros(movhor_tl)

totalcost = np.zeros(movhor_tl)
totalemms = np.zeros(movhor_tl)

ObjectiveNumber = np.zeros((movhor_tl, 4))

# Economic and environmental parameters
cnh3 = 1.46
con = 50
cel = 0.42
cpsa = 0.10
csh2 = 5.50
csn2 = 0.14
cbh2 = 2.30

h2perh = 100
nh3_tot_min = 45000

ces = ceb - 0.03

h2fossil = 9.3

el_start_risk = 5
el_op_risk = 1
reac_op_risk = 1
reac_ramp_risk = 2.5
stor_n2_risk = 1
stor_h2_risk = 1.5

# Initial decision variable values
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

loop = movhor_tl

# Regions where predicted label is competing
competing_region = []

# ==========================================================
# NEW: storage for tradeoff-length bar plot
# ==========================================================
tradeoff_indices = []
tradeoff_lengths = []
tradeoff_signed_cost_diffs = []
tradeoff_true_labels = []
tradeoff_predicted_labels = []
tradeoff_cost_02 = []
tradeoff_cost_08 = []
tradeoff_emms_02 = []
tradeoff_emms_08 = []

# Main optimization loop
for i in range(loop):

    ceb = ceb_total[i:i+48]
    gridemm = gridemm_total[i:i+48]
    ces = ceb - 0.03

    # Define optimization model
    nh3w = ConcreteModel()

    # Index sets
    nh3w.T = range(1, tl + 1)
    nh3w.T0 = range(0, tl + 1)
    nh3w.IL = range(1, il + 1)
    nh3w.DA = range(-2, tl + 1)

    # Variables
    nh3w.da = Var(
        nh3w.DA,
        domain=Binary,
        initialize=lambda m, j: da_ini[j + 2]
    )

    nh3w.de = Var(
        nh3w.T,
        domain=Integers,
        bounds=(0, None),
        initialize=lambda m, j: de_ini[j - 1]
    )

    nh3w.ms = Var(
        nh3w.IL,
        nh3w.T0,
        domain=NonNegativeReals,
        initialize=lambda m, i, t: ms_ini[i - 1, t]
    )

    nh3w.mp = Var(
        nh3w.IL,
        nh3w.T0,
        domain=NonNegativeReals,
        initialize=lambda m, i, t: mp_ini[i - 1, t]
    )

    nh3w.mnh3dev = Var(
        nh3w.T,
        within=Reals,
        initialize=lambda m, j: mnh3dev_ini[j - 1]
    )

    nh3w.ne = Var(
        nh3w.T0,
        domain=Integers,
        initialize=lambda m, j: ne_ini[j]
    )

    nh3w.peb = Var(
        nh3w.T,
        domain=NonNegativeReals,
        initialize=lambda m, j: peb_ini[j - 1]
    )

    nh3w.pei = Var(
        nh3w.IL,
        nh3w.T,
        domain=NonNegativeReals,
        initialize=lambda m, i, j: pei_ini[i - 1, j - 1]
    )

    nh3w.pes = Var(
        nh3w.T,
        domain=NonNegativeReals,
        initialize=lambda m, j: pes_ini[j - 1]
    )

    nh3w.u = Var(
        nh3w.IL,
        domain=NonNegativeReals,
        initialize=lambda m, i: u_ini[i - 1]
    )

    nh3w.bh2 = Var(
        nh3w.T,
        domain=NonNegativeReals,
        initialize=lambda m, j: bh2_ini[j - 1]
    )

    # Constraints
    def energybal_rule(m, j):
        return (
            wpow[j - 1]
            + m.peb[j]
            - m.pes[j]
            - m.pei[1, j]
            - m.pei[2, j]
            - m.pei[3, j]
            == 0
        )

    nh3w.energybal = Constraint(nh3w.T, rule=energybal_rule)

    def energyunt_rule(m, i, j):
        return m.pei[i, j] == rho[i - 1] * m.mp[i, j]

    nh3w.energyunt = Constraint(nh3w.IL, nh3w.T, rule=energyunt_rule)

    def minflow_rule(m, i, j):
        if i > 1:
            return m.mp[i, j] >= mmin[i - 1]
        return Constraint.Skip

    nh3w.minflow = Constraint(nh3w.IL, nh3w.T, rule=minflow_rule)

    def maxflow_rule(m, i, j):
        if i > 1:
            return m.mp[i, j] <= mmax[i - 1]
        return Constraint.Skip

    nh3w.maxflow = Constraint(nh3w.IL, nh3w.T, rule=maxflow_rule)

    nh3w.minelec = Constraint(
        nh3w.T,
        rule=lambda m, j: m.ne[j] >= 0
    )

    nh3w.maxelec = Constraint(
        nh3w.T,
        rule=lambda m, j: m.ne[j] <= nemax
    )

    nh3w.minh2pr = Constraint(
        nh3w.T,
        rule=lambda m, j: mmin[0] * m.ne[j] <= m.mp[1, j]
    )

    nh3w.maxh2pr = Constraint(
        nh3w.T,
        rule=lambda m, j: m.mp[1, j] <= mmax[0] * m.ne[j]
    )

    nh3w.minstor = Constraint(
        nh3w.IL,
        nh3w.T,
        rule=lambda m, i, j: m.ms[i, j] >= msmin[i - 1]
    )

    nh3w.maxstor = Constraint(
        nh3w.IL,
        nh3w.T,
        rule=lambda m, i, j: m.ms[i, j] <= msmax[i - 1]
    )

    nh3w.deviation = Constraint(
        nh3w.T,
        rule=lambda m, j: m.mnh3dev[j] == mnh3tar - m.mp[3, j]
    )

    nh3w.maxbh2 = Constraint(
        nh3w.T,
        rule=lambda m, j: m.bh2[j] <= h2perh
    )

    nh3w.nh3min = Constraint(
        rule=lambda m: sum(m.mp[3, j] for j in m.T) >= nh3_tot_min
    )

    def h2stor_rule(m, j):
        return (
            m.ms[1, j]
            == m.ms[1, j - 1]
            + m.bh2[j]
            + m.mp[1, j]
            - (3 / 17) * m.mp[3, j]
        )

    nh3w.h2stor = Constraint(nh3w.T, rule=h2stor_rule)

    def n2stor_rule(m, j):
        return (
            m.ms[2, j]
            == m.ms[2, j - 1]
            + m.mp[2, j]
            - (14 / 17) * m.mp[3, j]
        )

    nh3w.n2stor = Constraint(nh3w.T, rule=n2stor_rule)

    nh3w.totalstor = Constraint(
        nh3w.IL,
        rule=lambda m, i: m.u[i] == m.ms[i, 0] - m.ms[i, tl]
    )

    nh3w.elecon = Constraint(
        nh3w.T,
        rule=lambda m, j: m.de[j] >= m.ne[j] - m.ne[j - 1]
    )

    nh3w.rampmin = Constraint(
        nh3w.T,
        rule=lambda m, j: -m.da[j] * rmin <= m.mp[3, j] - m.mp[3, j - 1]
    )

    nh3w.rampmax = Constraint(
        nh3w.T,
        rule=lambda m, j: m.mp[3, j] - m.mp[3, j - 1] <= m.da[j] * rmax
    )

    def ramplim_rule(m, j):
        return sum(
            m.da[k] for k in range(max(-2, j + 1 - rampt), j + 1)
        ) <= damax

    nh3w.ramplim = Constraint(nh3w.T, rule=ramplim_rule)

    nh3w.ms_t0 = Constraint(
        nh3w.IL,
        rule=lambda m, i: m.ms[i, 0] == mi0[i - 1]
    )

    nh3w.ne_t0 = Constraint(
        rule=lambda m: m.ne[0] == ne0
    )

    nh3w.mnh3_t0 = Constraint(
        rule=lambda m: m.mp[3, 0] == mnh30
    )

    nh3w.da_t = Constraint(
        range(-2, 1),
        rule=lambda m, j: m.da[j] == da0[j + 2]
    )

    # Objective component definitions
    def costs_rule(m):
        return sum(
            sum(
                ceb[j - 1] * m.mp[1 + ii, j] * rho[ii]
                for ii in range(3)
            )
            + cel * m.mp[1, j]
            + cbh2 * m.bh2[j]
            + cpsa * m.mp[2, j]
            for j in m.T
        )

    def emms_rule(m):
        return (
            sum(
                rho[j - 1] * m.mp[j, i] * gridemm[i - 1]
                for j in m.IL
                for i in m.T
            )
            + sum(h2fossil * m.bh2[i] for i in m.T)
        )

    def wateruse_rule(m):
        return sum(
            m.mp[1, j] * 9.0 + m.bh2[j] * 4.5
            for j in m.T
        )

    def safety_rule(m):
        return sum(
            el_op_risk * m.mp[1, i] / mmax[0]
            for i in m.T
        )

    def Pareto_rule(m):
        return (
            0.45 * 2 * costs_rule(m)
            + 0.55 * (emms_rule(m) + wateruse_rule(m) + safety_rule(m))
        )

    def average_rule(m):
        return (
            0.5 * 3 * costs_rule(m)
            + 0.5 * emms_rule(m)
        )

    # ==========================================================
    # Competing region: calculate Pareto sweep and tradeoff length
    # ==========================================================
    if False:
        tradeoff_true_labels.append(true_labels[i])

        competing_region.append(i)

        weight_range = np.arange(0.2, 0.81, 0.01)
        pareto_results = []

        for w in weight_range:

            def weighted_pareto_rule(m):
                return w * 1.5 * costs_rule(m) + (1 - w) * emms_rule(m)

            if hasattr(nh3w, 'obj'):
                nh3w.del_component('obj')

            nh3w.obj = Objective(rule=weighted_pareto_rule, sense=minimize)

            result = solver.solve(nh3w)

            cost_val = value(costs_rule(nh3w))
            emms_val = value(emms_rule(nh3w))

            pareto_results.append((w, cost_val, emms_val))

        if hasattr(nh3w, 'obj'):
            nh3w.del_component('obj')

        pareto_array = np.array(pareto_results)

        # ======================================================
        # NEW: calculate tradeoff length from cost at w=0.8 and w=0.2
        # ======================================================
        w_idx_02 = np.where(np.isclose(pareto_array[:, 0], 0.2))[0][0]
        w_idx_08 = np.where(np.isclose(pareto_array[:, 0], 0.8))[0][0]

        cost_02 = pareto_array[w_idx_02, 1]
        cost_08 = pareto_array[w_idx_08, 1]

        emms_02 = pareto_array[w_idx_02, 2]
        emms_08 = pareto_array[w_idx_08, 2]

        cost_08_minus_02 = cost_08 - cost_02
        tradeoff_length = abs(cost_08_minus_02)

        tradeoff_indices.append(i)
        tradeoff_lengths.append(tradeoff_length)
        tradeoff_signed_cost_diffs.append(cost_08_minus_02)
        tradeoff_true_labels.append(true_labels[i])
        tradeoff_predicted_labels.append(predicted_labels[i])
        tradeoff_cost_02.append(cost_02)
        tradeoff_cost_08.append(cost_08)
        tradeoff_emms_02.append(emms_02)
        tradeoff_emms_08.append(emms_08)

        print(
            f"Time {i}: "
            f"cost@0.2 = {cost_02:.4f}, "
            f"cost@0.8 = {cost_08:.4f}, "
            f"cost@0.8 - cost@0.2 = {cost_08_minus_02:.4f}, "
            f"length = {tradeoff_length:.4f}, "
            f"true label = {true_labels[i]}"
        )

        # Identify knee points
        costs = pareto_array[:, 1]
        emms = pareto_array[:, 2]
        tol = 1e-2

        knee_ranges = []
        start = 0

        for j in range(1, len(costs)):
            if (
                abs(costs[j] - costs[start]) > tol
                or abs(emms[j] - emms[start]) > tol
            ):
                if j - start > 1:
                    knee_ranges.append((start, j - 1))
                start = j

        if len(costs) - start > 1:
            knee_ranges.append((start, len(costs) - 1))

        if knee_ranges:
            longest = max(knee_ranges, key=lambda x: x[1] - x[0])
            w_start, w_end = longest
            w_mid = pareto_array[(w_start + w_end) // 2, 0]

            print(
                f"Selected knee point at weight {w_mid} "
                f"with cost {pareto_array[w_start, 1]} "
                f"and emms {pareto_array[w_start, 2]}"
            )
        else:
            w_mid = 0.5
            print("No knee point found, using default weight 0.5.")

        def selected_pareto_rule(m):
            return (
                w_mid * costs_rule(m)
                + (1 - w_mid) * (
                    emms_rule(m)
                    + wateruse_rule(m)
                    + safety_rule(m)
                )
            )

        nh3w.obj = Objective(rule=selected_pareto_rule, sense=minimize)

    else:
        nh3w.obj = Objective(rule=costs_rule, sense=minimize)
        # nh3w.obj = Objective(rule=emms_rule, sense=minimize)
        tradeoff_true_labels.append(true_labels[i])


    # Solve selected moving-horizon optimization
    results = solver.solve(nh3w)

    # Calculate output metrics
    costss = value(costs_rule(nh3w))
    emms = value(emms_rule(nh3w))

    watuse = sum(
        value(nh3w.mp[1, j]) * 9.0
        + value(nh3w.bh2[j]) * 4.5
        for j in nh3w.T
    )

    safetyelectro = sum(
        el_op_risk * value(nh3w.mp[1, j]) / mmax[0]
        for j in nh3w.T
    )

    objvals = [costss, emms, watuse, safetyelectro]
    ObjectiveNumber[i, :] = objvals

    # Moving horizon state update
    for k in range(il):
        mi0[k] = value(nh3w.ms[k + 1, 1])

    ne0 = value(nh3w.ne[1])
    mnh30 = value(nh3w.mp[3, 1])

    for k in range(3):
        da0[k] = value(nh3w.da[k - 2])

    # Store first-step decisions
    mnh3[i] = value(nh3w.mp[3, 1])
    mh2[i] = value(nh3w.mp[1, 1])
    mn2[i] = value(nh3w.mp[2, 1])
    mh2b[i] = value(nh3w.bh2[1])

    totalcost[i] = (
        sum(
            ceb[0] * value(nh3w.mp[1 + ii, 1]) * rho[ii]
            for ii in range(3)
        )
        + cel * value(nh3w.mp[1, 1])
        + cbh2 * value(nh3w.bh2[1])
        + cpsa * value(nh3w.mp[2, 1])
    )

    totalemms[i] = (
        sum(
            rho[j - 1] * value(nh3w.mp[j, 1]) * gridemm[0]
            for j in nh3w.IL
        )
        + h2fossil * value(nh3w.bh2[1])
    )


# ==========================================================
# Export source data for both figures into one Excel file
# ==========================================================
x_hours = np.arange(1, movhor_tl + 1)
in_competing_region = np.isin(x_hours - 1, competing_region).astype(int)

# Figure 1 source data: material flows (NH3, H2, N2, H2 bought)
fig1_df = pd.DataFrame({
    "Time_h":                     x_hours,
    "Ammonia_Produced_kg_per_h":  mnh3,
    "Hydrogen_Produced_kg_per_h": mh2,
    "Nitrogen_Produced_kg_per_h": mn2,
    "Hydrogen_Bought_kg_per_h":   mh2b,
    "Competing_Region_Flag":      in_competing_region,
})

# Figure 2 source data: objective metrics, per-step cost/emms,
# and the power-price / carbon-intensity profiles in the bottom subplot
fig2_df = pd.DataFrame({
    "Time_h":                 x_hours,
    "Cost_Objective":         ObjectiveNumber[:, 0],
    "Emissions_Objective":    ObjectiveNumber[:, 1],
    "Water_Use_Objective":    ObjectiveNumber[:, 2],
    "Safety_Objective":       ObjectiveNumber[:, 3],
    "Cost_per_Step":          totalcost,
    "Emissions_per_Step":     totalemms,
    "Power_Price":            df1["power price"].values[a:b],
    "CI_kgCO2_per_kWh":       df1["CI kgCO2/kwh"].values[a:b],
    "Competing_Region_Flag":  in_competing_region,
})

# Per-step labels (the predicted label drives the grey-shaded competing regions)
labels_df = pd.DataFrame({
    "Time_h":          x_hours,
    "True_Label":      true_labels,
    "Predicted_Label": predicted_labels,
})

# Tradeoff sweep results (one row per competing hour, used for any tradeoff-length plot)
if len(tradeoff_lengths) > 0:
    tradeoff_df = pd.DataFrame({
        "Time_Index":               tradeoff_indices,
        "True_Label":               tradeoff_true_labels,
        "Predicted_Label":          tradeoff_predicted_labels,
        "Cost_at_w_0_2":            tradeoff_cost_02,
        "Cost_at_w_0_8":            tradeoff_cost_08,
        "Cost_0_8_minus_Cost_0_2":  tradeoff_signed_cost_diffs,
        "Tradeoff_Length_Abs":      tradeoff_lengths,
        "Emissions_at_w_0_2":       tradeoff_emms_02,
        "Emissions_at_w_0_8":       tradeoff_emms_08,
    })
else:
    tradeoff_df = pd.DataFrame(columns=[
        "Time_Index", "True_Label", "Predicted_Label",
        "Cost_at_w_0_2", "Cost_at_w_0_8", "Cost_0_8_minus_Cost_0_2",
        "Tradeoff_Length_Abs", "Emissions_at_w_0_2", "Emissions_at_w_0_8",
    ])

# Summary scalars
totalnh3_export = float(np.sum(mnh3))
summary_df = pd.DataFrame({
    "Metric": [
        "Total ammonia produced (kg)",
        "Total cost in scheduling horizon",
        "Total cost objective",
        "Total emissions in scheduling horizon",
        "Total emissions objective",
        "Cost per ammonia",
        "Emissions per ammonia",
        "a (start index)",
        "b (end index)",
        "Moving-horizon length",
    ],
    "Value": [
        totalnh3_export,
        float(np.sum(totalcost)),
        float(np.sum(ObjectiveNumber[:, 0])),
        float(np.sum(totalemms)),
        float(np.sum(ObjectiveNumber[:, 1])),
        float(np.sum(totalcost) / max(totalnh3_export, 1e-12)),
        float(np.sum(totalemms) / max(totalnh3_export, 1e-12)),
        a,
        b,
        movhor_tl,
    ],
})

excel_out = "MHS_cost_emiss_source_data.xlsx"
with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    fig1_df.to_excel(writer,     sheet_name="Fig1_Material_Flows",       index=False)
    fig2_df.to_excel(writer,     sheet_name="Fig2_Objectives_and_Price", index=False)
    labels_df.to_excel(writer,   sheet_name="Labels",                    index=False)
    tradeoff_df.to_excel(writer, sheet_name="Tradeoff_Sweep",            index=False)
    summary_df.to_excel(writer,  sheet_name="Summary",                   index=False)
print(f"Saved figure source data to: {excel_out}")


# ==========================================================
# Plot 1: material flows
# ==========================================================
groupingcolor = 'grey'
groupingalpha = 0.1

plt.figure(figsize=(28, 2.5))

x = range(1, movhor_tl + 1)

plt.plot(x, mnh3, label="Ammonia Produced")
plt.plot(x, mh2, label="Hydrogen Produced")
plt.plot(x, mn2, label="Nitrogen Produced")
plt.plot(x, mh2b, label="Hydrogen Bought")

for idx in competing_region:
    plt.axvspan(idx, idx + 1, color=groupingcolor, alpha=groupingalpha)

plt.xlabel("Time, h",fontsize=16)
plt.ylabel("Material Flow, kg/h", fontsize=16)
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    frameon=False,
)
plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.show()


# ==========================================================
# Plot 2: objective metrics and price/emission profiles
# ==========================================================
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax1 = plt.subplot(gs[0])

x = range(1, movhor_tl + 1)

ax1.plot(x, ObjectiveNumber[:, 0], label="Costs", color="blue")
ax1.plot(x, ObjectiveNumber[:, 1], label="Emissions", color="green")
ax1.plot(x, ObjectiveNumber[:, 2], label="Water Use", color="orange")
ax1.plot(x, ObjectiveNumber[:, 3], label="Safety", color="purple")

for idx in competing_region:
    ax1.axvspan(idx, idx + 1, color=groupingcolor, alpha=groupingalpha)

ax1.set_xlabel("Time, h")
ax1.set_ylabel("Objective Metrics")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()

ax2.plot(x, totalcost, label="Cost per Step", color="red", linestyle="--")
ax2.plot(x, totalemms, label="Emms per Step", color="brown", linestyle="--")

ax2.set_ylabel("Cost and Emissions")
ax2.legend(loc="upper right")

ax1.set_title("Objective Metrics and Cost/Emissions per Step")

ax3 = plt.subplot(gs[1])

time_range = range(1, movhor_tl + 1)

ax3.plot(
    time_range,
    df1["power price"].values[a:b],
    label="Power Price",
    color="blue"
)

ax3.plot(
    time_range,
    df1["CI kgCO2/kwh"].values[a:b],
    label="CI kgCO2/kWh",
    color="green"
)

for idx in competing_region:
    ax3.axvspan(idx, idx + 1, color=groupingcolor, alpha=groupingalpha)

ax3.set_xlabel("Time, h")
ax3.set_ylabel("Values")
ax3.legend(loc="upper left")
ax3.set_title("Power Price and Carbon Intensity")

plt.tight_layout()
plt.show()


# ==========================================================
# NEW Plot 3: tradeoff-length bar plot color-coded by true labelss
# ==========================================================
# if len(tradeoff_lengths) > 0
    # Optional second plot: signed value exactly as cost@0.8 - cost@0.2
    
# ==========================================================
# Final summary
# ==========================================================
totalnh3 = sum(mnh3)

print(f"{totalnh3} ammonia produced")
print(f"{sum(totalcost)} total cost in the scheduling horizon")
print(f"{sum(ObjectiveNumber[:, 0])} total cost objective in the scheduling horizon")
print(f"{sum(totalemms)} total emissions in the scheduling horizon")
print(f"{sum(ObjectiveNumber[:, 1])} total emissions objective in the scheduling horizon")
print(f"{sum(totalcost) / totalnh3} cost per ammonia in the scheduling horizon")
print(f"{sum(totalemms) / totalnh3} total emissions per ammonia in the scheduling horizon")