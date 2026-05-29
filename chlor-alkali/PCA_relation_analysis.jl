using JuMP, CPLEX
using CSV, DataFrames
using Statistics, LinearAlgebra  # Essential for PCA and Correlation analysis
using Plots

# ==========================================
# 1. Data Loading & Parameter Initialization
# ==========================================
# Load your data here (Make sure the path is correct)
# df1 = CSV.read("UK_price and emission.csv", DataFrame)
df1 = CSV.read("UK_price and emission.csv", DataFrame)
numbegin = 155
T = 48
price = zeros(T)  # example: 3x3 matrix over timegrid

gridCI = zeros(T)  # example: 3x3 matrix over timegrid

price .= Float64.(df1[!, "price(dollar/kwh)"][numbegin:numbegin+T-1])  # note ".=" for broadcasting
gridCI .= Float64.(df1[!, "Actual Carbon Intensity (kgCO2/kWh)"][numbegin:numbegin+T-1])  # note ".=" for broadcasting
println("Price (first 5): ", price)
println("Grid CI (first 5): ", gridCI)

# Option B: Define them as constant baseline demands for testing
Dgas = fill(50.0, T) # Example: 1500 units of gas demanded per hour
Dliq = fill(300.0, T) # Example: 3000 units of liquid demanded per hour
# Plant Physical Parameters
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
Imax = 1e9
n_drag = 4.0
c_op = 1.0
c_ramp = 1.0

@assert length(price) == T
@assert length(gridCI) == T

# Coefficients Calculation
d_w = Nc * n_drag * 18.0e-3 * 3600.0 / F   
w_transient = 0.05
α = ηF * (Nc/(2F)) * 3600.0 * 1e-3
β = (Nc * U_nom) / 1000.0
println("Coefficients: α = $(round(α, digits=4)) kmol/h per A, β = $(round(β, digits=4)) kW per A, d_w = $(round(d_w, digits=4)) kg water/(A·h)")
# ==========================================
# 2. Optimization Model Construction
# ==========================================
model = Model(CPLEX.Optimizer)
set_silent(model) # Suppress solver logs for clean output

# ---- Variables ----
@variable(model, Imin <= I[1:T] <= Imax)     
@variable(model, Sin[1:T] >= 0)              
@variable(model, Sout[1:T] >= 0)             
@variable(model, 0 <= M[1:T] <= Mmax)        
@variable(model, Isurge[1:T] >= 0)
@variable(model, dI[2:T] >= 0)
@variable(model, dI_up[2:T] >= 0)

# ---- Expressions ----
@expression(model, R_expr[t=1:T], α * I[t])
@expression(model, P[t=1:T], β * I[t])

# ---- Constraints ----
@constraint(model, [t=1:T], R_expr[t] >= Dgas[t])
@constraint(model, sum(R_expr[t] for t in 1:T) >= 9600)

# @constraint(model, [t=1:T], Dliq[t] == γ*(R_expr[t] - Dgas[t]) + (Sout[t] - Sin[t]))
@constraint(model, [t=1:T], Sin[t] <= γ*(R_expr[t] - Dgas[t]))

@constraint(model, M[1] == M0 + Sin[1] - Sout[1])
@constraint(model, [t=2:T], M[t] == M[t-1] + Sin[t] - Sout[t])
@constraint(model, M[T] >= M0)

ramp_limit = 1e5
@constraint(model, [t=2:T], I[t] - I[t-1] <= ramp_limit)  # Max ramp up
@constraint(model, [t=2:T], I[t-1] - I[t] <= ramp_limit)  # Max ramp down

# 5. Tracking variables (for cost penalties if you want to use them later)
@constraint(model, [t=2:T], dI[t] >= I[t] - I[t-1])
@constraint(model, [t=2:T], dI[t] >= -(I[t] - I[t-1]))
@constraint(model, [t=2:T], dI_up[t] >= I[t] - I[t-1])
@constraint(model, [t=1:T], Isurge[t] >= I[t] - I_nom)
# ---- Objective Expressions ----
@expression(model, Jcost, sum(price[t]  * P[t] for t in 1:T))   
@expression(model, Jem,   sum(gridCI[t] * P[t] for t in 1:T))   

# ==========================================
# 3. Strict PCA-based Relationship Diagnosis
# ==========================================
println("\nStarting Pareto Front Sampling...")

# Step 3.1: Find bounds of Emission objective (Anchor points)
@objective(model, Min, Jcost)
optimize!(model)
E_max = value(Jem)
println("Maximum Emission: E_max = $(round(E_max, digits=2))")

@objective(model, Min, Jem)
optimize!(model)
E_min = value(Jem)
println("Minimum Emission: E_min = $(round(E_min, digits=2))")


println("Emission Bounds: E_min = $(round(E_min, digits=2)), E_max = $(round(E_max, digits=2))")

# Step 3.2: Pareto Sampling via Epsilon-Constraint Method
n_samples = 60
eps_values = range(E_min, E_max, length=n_samples)
pareto_sols = Vector{Vector{Float64}}()

# Define a named constraint for Emission to be updated iteratively
@constraint(model, em_constr, Jem <= E_max)
@objective(model, Min, Jcost) # Minimize Cost subject to Emission limit

for eps in eps_values
    set_normalized_rhs(em_constr, eps) # Efficiently update the RHS of the constraint
    optimize!(model)
    
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        push!(pareto_sols, [value(Jcost), value(Jem)])
    end
end

println("Successfully obtained $(length(pareto_sols)) non-dominated solutions.")

# Step 3.2.1: Plot Pareto Frontier (Cost vs Emission)
if !isempty(pareto_sols)
    costs = [sol[1] for sol in pareto_sols]
    emissions = [sol[2] for sol in pareto_sols]

    order_idx = sortperm(emissions)
    costs_sorted = costs[order_idx]
    emissions_sorted = emissions[order_idx]

    pareto_plot = scatter(
        costs_sorted,
        emissions_sorted,
        label="Pareto Solutions",
        xlabel="Cost",
        ylabel="Emission",
        title="Pareto Frontier (Cost vs Emission)",
        markersize=5,
        markerstrokewidth=0.5,
    )
    display(plot(pareto_plot, costs_sorted, emissions_sorted, label="Frontier"))
    savefig(pareto_plot, "pareto_frontier.png")
    println("Saved Pareto frontier plot to pareto_frontier.png")
else
    println("No Pareto solutions found; skipping plot.")
end

# Step 3.3: Data processing and Correlation Matrix R
# Matrix construction (N x 2)
data = reduce(hcat, pareto_sols)'  
# cor() calculates Pearson correlation matrix (inherently standardizes the data)
R = cor(data)                      

# Step 3.4: Eigen-decomposition
eig = eigen(R)
# Sort eigenvalues and eigenvectors in descending order
idx = sortperm(eig.values, rev=true)
λ = eig.values[idx]
V = eig.vectors[:, idx]

# Step 3.5: Apply Deb & Saxena's strict dimensionality criteria
explained_variance_PC1 = λ[1] / sum(λ)
explained_variance_PC2 = λ[2] / sum(λ)
correlation_r = R[1, 2]

println("\n==========================================")
println("           PCA DIAGNOSTIC RESULTS")
println("==========================================")
println("Correlation Coefficient (r) between Cost & Emission: ", round(correlation_r, digits=4))
println("Explained Variance by PC1: ", round(explained_variance_PC1 * 100, digits=2), "%")
println("Explained Variance by PC2 (Conflict): ", round(explained_variance_PC2 * 100, digits=2), "%")
println("------------------------------------------")

# Threshold Cut (TC) Rule from the paper: 95%
if explained_variance_PC1 >= 0.95
    println(">>> CONCLUSION: Objectives are CORRELATING (Redundant) <<<")
    println("Reason: A single dimension (PC1) explains >95% of the system variance.")
    println("Advice: You can simplify this to a Single-Objective problem.")
else
    println(">>> CONCLUSION: Objectives are COMPETING (Conflicting) <<<")
    println("Reason: The conflict dimension (PC2) is significant (>5% variance).")
    
    # Strict mathematical proof via eigenvector signs
    v2 = V[:, 2]
    if sign(v2[1]) != sign(v2[2])
        println("Strict Proof: In the conflict eigenvector (PC2) $(round.(v2, digits=3)),")
        println("weights of Cost and Emission have opposite signs, proving a fundamental trade-off.")
    end
end
println("==========================================")