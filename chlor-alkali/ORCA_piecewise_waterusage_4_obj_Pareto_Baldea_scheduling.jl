using JuMP, CPLEX
using CSV, DataFrames, ORCA
df1 = CSV.read("UK_price and emission.csv", DataFrame)
numbegin = 60
T = 48
price = zeros(T)  # example: 3x3 matrix over timegrid

gridCI = zeros(T)  # example: 3x3 matrix over timegrid

price .= Float64.(df1[!, "price(dollar/kwh)"][numbegin:numbegin+T-1])  # note ".=" for broadcasting
gridCI .= Float64.(df1[!, "Actual Carbon Intensity (kgCO2/kWh)"][numbegin:numbegin+T-1])  # note ".=" for broadcasting
println("Price (first 5): ", price)
println("Grid CI (first 5): ", gridCI)

"""
Solve simplified chlor-alkali scheduling with 4 objectives.

Returns a NamedTuple with:
  ok::Bool, Jc::Float64, Je::Float64, Jw::Float64, Js::Float64

Weights:
  w_pair controls the tradeoff between two chosen objectives (objA vs objB).

Water objective has three physically-grounded terms:
  1. Reaction water: 36 kg per kmol Cl2 (stoichiometric, ∝ R[t])
  2. Electro-osmotic drag: d_w kg per A·h — water dragged through the
     membrane by Na+ ions (n_drag ≈ 4 mol H2O per mol Na+)
  3. Boiler make-up water: b_w kg per kmol NaOH surplus — water consumed
     by the steam boiler when concentrating NaOH from 32% to 50%.
     This term penalizes overproduction (R[t] - Dgas[t]) and creates a
     tradeoff with cost/emissions, which prefer overproducing during
     cheap/clean hours and storing the surplus.
"""
I_nom = 5e6 
k_surge = 0.8  
Nc = 160.0
ηF = 0.95
F = 96485.0
U_nom = 3.2
γ = 0.99
M0 = 0.0
Mmax = 5e4
Imin = 0.0
Imax = 1e7
n_drag = 4.0
c_op = 1.0
c_ramp = 1.0
@assert length(price) == T
@assert length(gridCI) == T
@assert length(Dgas) == T
@assert length(Dliq) == T

# Electro-osmotic drag coefficient: kg water per A per hour
# Each Na+ crossing the membrane drags n_drag molecules of H2O.
# Current I (A) = I/F mol e⁻/s; for Na+, 1 mol e⁻ transfers 1 mol Na+.
# Water dragged = Nc * (I/F) * n_drag mol/s = Nc * n_drag * I / F mol/s
# Convert to kg/h: × 18e-3 kg/mol × 3600 s/h
d_w = Nc * n_drag * 18.0e-3 * 3600.0 / F   # ≈ 0.43 kg water/(A·h) for n_drag=4
w_transient = 0.05
# A -> kmol/h (Cl2)
α = ηF * (Nc/(2F)) * 3600.0 * 1e-3
# kW per A
β = (Nc * U_nom) / 1000.0

model = Model(CPLEX.Optimizer)
set_silent(model)

# ---- variables ----
@variable(model, Imin <= I[1:T] <= Imax)     # A              # kmol/h (Cl2)
@variable(model, Sin[1:T] >= 0)              # kmol/h into storage
@variable(model, Sout[1:T] >= 0)             # kmol/h out of storage
@variable(model, 0 <= M[1:T] <= Mmax)        # kmol in storage
# ---- surge helper (linearized max) ----
@variable(model, Isurge[1:T] >= 0)
# abs ramp helper for safety
@variable(model, dI[2:T] >= 0)
@variable(model, dI_up[2:T] >= 0)

# ---- mappings ----
@expression(model, R[t=1:T], α * I[t])
@expression(model, P[t=1:T], β * I[t])

# ---- demand / storage constraints ----
@constraint(model, [t=1:T], R[t] >= Dgas[t])

@constraint(model, [t=1:T], Dliq[t] == γ*(R[t] - Dgas[t]) + (Sout[t] - Sin[t]))
@constraint(model, [t=1:T], Sin[t] <= γ*(R[t] - Dgas[t]))

@constraint(model, M[1] == M0 + Sin[1] - Sout[1])
@constraint(model, [t=2:T], M[t] == M[t-1] + Sin[t] - Sout[t])
@constraint(model, M[T] >= M0)

# ---- ramp abs constraints ----
@constraint(model, [t=2:T], dI[t] >=  I[t] - I[t-1])
@constraint(model, [t=2:T], dI[t] >= -(I[t] - I[t-1]))

# ramp-up only: dI_up[t] = max(I[t] - I[t-1], 0)
@constraint(model, [t=2:T], dI_up[t] >= I[t] - I[t-1])

# Isurge[t] = max(I[t] - I_nom, 0)  (only penalize when I exceeds I_nom)
@constraint(model, [t=1:T], Isurge[t] >= I[t] - I_nom)
# @expression(model, Isurge[t=1:T], max(I[t] - I_nom, 0))
# ---- objectives (expressions) ----
@expression(model, Jcost,  sum(price[t]  * P[t] for t in 1:T))   # $
@expression(model, Jem,    sum(gridCI[t] * P[t] for t in 1:T))   # kgCO2 (if CI is kg/kWh)
# Water = reaction water + electro-osmotic drag + boiler make-up for NaOH evaporation
#   (1) 36 kg/kmol Cl2 — stoichiometric reaction water (∝ R[t])
#   (2) d_w * I[t] — water dragged through membrane by Na+ ions (∝ I[t])
#   (3) b_w * γ*(R[t]-Dgas[t]) — boiler water for concentrating NaOH (∝ surplus production)
#       This term penalizes overproduction and creates tradeoff with cost/emissions.
# @expression(model, Jwater, sum(36.0 * R[t] + k_surge * Isurge[t] for t in 1:T))  # kg water
@expression(model, Jwater, sum(36.0 * R[t] for t in 1:T)
+sum(w_transient * dI_up[t] for t in 2:T)
)  # kg water
# @expression(model, Jwater, sum(36.0 * R[t] + d_w * I[t] + b_w * γ * (R[t] - Dgas[t]) + k_surge * Isurge[t] for t in 1:T))  # kg water
# Safety proxy: operating current + ramp magnitude
@expression(model, Jsafe,  c_op*sum(I[t] for t in 1:T) + c_ramp*sum(dI[t] for t in 2:T))

# ---- optional eps constraints ----

# helper to select objective expression by symbol

# @objective(model, Min, w_pair*A + (1.0-w_pair)*B)

# optimize!(model)


if termination_status(model) != MOI.OPTIMAL
    return (ok=false, Jc=NaN, Je=NaN, Jw=NaN, Js=NaN)
end

res = ORCA.main(model, [Jcost, Jem, Jwater, Jsafe], 2)
println("ORCA results:", res.groups)

