using JuMP, CPLEX
using CSV, DataFrames
using Plots
using XLSX
using Statistics
using ORCA

# ==========================================
# Configuration
# ==========================================
const T       = 48
const Imin    = 0.0
const Imax    = 1e7
const ramp_limit = 1.5e5
const Nc      = 160.0
const ηF      = 0.95
const F       = 96485.0
const U_nom   = 3.2
const α       = ηF * (Nc / (2F)) * 3600.0 * 1e-3
const β       = (Nc * U_nom) / 1000.0
const Dgas    = fill(150.0, T)

# ==========================================
# 1. Load Data
# ==========================================
df_chlor  = CSV.read("datasets/Tokyo_first_8000_correlation_strength_comparison.csv", DataFrame)
df_ammonia = CSV.read("datasets/total_factor_grouping_2022_Tokyo_DA.csv", DataFrame)
df_prices  = DataFrame(XLSX.readtable("datasets/Tokyo_price and emission intensity.xlsx", 1))

# Align to shared length (8000 rows)
n = min(nrow(df_chlor), nrow(df_ammonia), 8000)
df_chlor   = df_chlor[1:n, :]
df_ammonia = df_ammonia[1:n, :]

new_strength = Float64.(df_chlor[!, :New_Strength])
old_strength = Float64.(df_ammonia[!, :Column2])

# ==========================================
# 2. Find Candidate Points
# ==========================================
# Score = new_strength - old_strength  (high = Case A: chlor high / ammonia low)
score = new_strength .- old_strength

# Case A: chlor-alkali NEW strength HIGH, ammonia OLD strength LOW
idx_A = argmax(score)

# Case B: chlor-alkali NEW strength LOW, ammonia OLD strength HIGH
idx_B = argmin(score)

println("=" ^ 55)
println("Case A (Chlor HIGH / Ammonia LOW):  window index = $idx_A")
println("  New Strength (chlor-alkali) = $(round(new_strength[idx_A], digits=6))")
println("  Old Strength (ammonia)      = $(round(old_strength[idx_A], digits=6))")
println()
println("Case B (Chlor LOW / Ammonia HIGH):  window index = $idx_B")
println("  New Strength (chlor-alkali) = $(round(new_strength[idx_B], digits=6))")
println("  Old Strength (ammonia)      = $(round(old_strength[idx_B], digits=6))")
println("=" ^ 55)

# ==========================================
# 3. Helper: extract window & run ORCA
# ==========================================
function get_window(df_prices, numbegin)
    last_index = numbegin + T - 1
    price  = Float64.(df_prices[!, "price(dollars/kwh)"][numbegin:last_index])
    gridCI = Float64.(df_prices[!, "CO2 kg/kWh (LCA)"][numbegin:last_index])
    return price, gridCI
end

function solve_orca(price, gridCI)
    Tlocal = length(price)
    model  = Model(CPLEX.Optimizer)
    set_silent(model)

    @variable(model, Imin <= I[1:Tlocal] <= Imax)
    @expression(model, R_expr[t=1:Tlocal], α * I[t])
    @expression(model, P[t=1:Tlocal],      β * I[t])

    @constraint(model, [t=1:Tlocal],    R_expr[t] >= Dgas[t])
    @constraint(model, sum(R_expr[t] for t in 1:Tlocal) >= 9600)
    @constraint(model, [t=2:Tlocal],    I[t] - I[t-1] <= ramp_limit)
    @constraint(model, [t=2:Tlocal],    I[t-1] - I[t] <= ramp_limit)

    @expression(model, Jcost, sum(price[t]  * P[t] for t in 1:Tlocal))
    @expression(model, Jem,   sum(gridCI[t] * P[t] for t in 1:Tlocal))

    res = ORCA.main(model, [Jcost, Jem], 2)
    return res
end

# Generate Pareto frontier via weighted-sum scalarization (n_points evenly spaced weights)
function pareto_weighted_sum(price, gridCI; n_points=30)
    Tlocal = length(price)
    pareto_cost = Float64[]
    pareto_em   = Float64[]
    I_profiles  = Matrix{Float64}(undef, Tlocal, n_points)

    for (k, w) in enumerate(range(0.0, 1.0, length=n_points))
        model = Model(CPLEX.Optimizer)
        set_silent(model)

        @variable(model, Imin <= I[1:Tlocal] <= Imax)
        @expression(model, P[t=1:Tlocal], β * I[t])
        @expression(model, R_expr[t=1:Tlocal], α * I[t])

        @constraint(model, [t=1:Tlocal],    R_expr[t] >= Dgas[t])
        @constraint(model, sum(R_expr[t] for t in 1:Tlocal) >= 9600)
        @constraint(model, [t=2:Tlocal],    I[t] - I[t-1] <= ramp_limit)
        @constraint(model, [t=2:Tlocal],    I[t-1] - I[t] <= ramp_limit)

        Jcost = sum(price[t]  * P[t] for t in 1:Tlocal)
        Jem   = sum(gridCI[t] * P[t] for t in 1:Tlocal)

        @objective(model, Min, w * Jcost + (1 - w) * Jem)
        optimize!(model)

        push!(pareto_cost, value(Jcost))
        push!(pareto_em,   value(Jem))
        I_profiles[:, k] = value.(I)
    end

    # Sort by cost for a clean frontier curve
    ord = sortperm(pareto_cost)
    return pareto_cost[ord], pareto_em[ord], I_profiles[:, ord]
end

# ==========================================
# 4. Solve both cases
# ==========================================
price_A, gridCI_A = get_window(df_prices, idx_A)
price_B, gridCI_B = get_window(df_prices, idx_B)

println("Running ORCA for Case A (window $idx_A)...")
res_A = solve_orca(price_A, gridCI_A)

println("Running ORCA for Case B (window $idx_B)...")
res_B = solve_orca(price_B, gridCI_B)

println("Computing Pareto frontier for Case A...")
pareto_cost_A, pareto_em_A, I_profiles_A = pareto_weighted_sum(price_A, gridCI_A)

println("Computing Pareto frontier for Case B...")
pareto_cost_B, pareto_em_B, I_profiles_B = pareto_weighted_sum(price_B, gridCI_B)

time_axis = 1:T

# ==========================================
# 5. Plotting
# ==========================================

# ---------- Case A ----------
p_price_A = plot(time_axis, price_A,
    xlabel="Time Step (h)", ylabel="Price (\$/kWh)",
    title="Case A (idx=$idx_A): Power Price",
    label="Price", color=:steelblue, linewidth=2)

p_em_A = plot(time_axis, gridCI_A,
    xlabel="Time Step (h)", ylabel="Emission Intensity (kg CO₂/kWh)",
    title="Case A (idx=$idx_A): Emission Intensity",
    label="Emission", color=:firebrick, linewidth=2)

p_pareto_A = scatter(pareto_cost_A, pareto_em_A,
    xlabel="Cost Objective", ylabel="Emission Objective",
    title="Case A (idx=$idx_A): Pareto Frontier",
    label="Pareto points", color=:purple, markersize=5)

# Current profiles: plot each Pareto solution as a semi-transparent line
p_current_A = plot(title="Case A (idx=$idx_A): Current Profiles",
    xlabel="Time Step (h)", ylabel="Current I (A)", legend=false)
for j in 1:size(I_profiles_A, 2)
    plot!(p_current_A, time_axis, I_profiles_A[:, j],
          alpha=0.3, color=:darkorange, linewidth=1)
end

fig_A = plot(p_price_A, p_em_A, p_pareto_A, p_current_A,
    layout=(2, 2), size=(1100, 800),
    plot_title="Case A — Chlor-Alkali HIGH / Ammonia LOW  (window $idx_A)")
savefig(fig_A, "case_A_analysis.png")
println("Saved case_A_analysis.png")

# ---------- Case B ----------
p_price_B = plot(time_axis, price_B,
    xlabel="Time Step (h)", ylabel="Price (\$/kWh)",
    title="Case B (idx=$idx_B): Power Price",
    label="Price", color=:steelblue, linewidth=2)

p_em_B = plot(time_axis, gridCI_B,
    xlabel="Time Step (h)", ylabel="Emission Intensity (kg CO₂/kWh)",
    title="Case B (idx=$idx_B): Emission Intensity",
    label="Emission", color=:firebrick, linewidth=2)

p_pareto_B = scatter(pareto_cost_B, pareto_em_B,
    xlabel="Cost Objective", ylabel="Emission Objective",
    title="Case B (idx=$idx_B): Pareto Frontier",
    label="Pareto points", color=:purple, markersize=5)

p_current_B = plot(title="Case B (idx=$idx_B): Current Profiles",
    xlabel="Time Step (h)", ylabel="Current I (A)", legend=false)
for j in 1:size(I_profiles_B, 2)
    plot!(p_current_B, time_axis, I_profiles_B[:, j],
          alpha=0.3, color=:darkorange, linewidth=1)
end

fig_B = plot(p_price_B, p_em_B, p_pareto_B, p_current_B,
    layout=(2, 2), size=(1100, 800),
    plot_title="Case B — Chlor-Alkali LOW / Ammonia HIGH  (window $idx_B)")
savefig(fig_B, "case_B_analysis.png")
println("Saved case_B_analysis.png")

# ---------- Overlay: mark chosen points on full strength scatter ----------
p_overview = scatter(new_strength, old_strength,
    xlabel="New Strength (Chlor-Alkali)",
    ylabel="Old Strength (Ammonia)",
    title="All 8000 Windows — Selected Cases Highlighted",
    label="All windows", color=:lightgray, markersize=3,
    markerstrokewidth=0, alpha=0.6)

scatter!(p_overview, [new_strength[idx_A]], [old_strength[idx_A]],
    label="Case A (Chlor↑ Ammonia↓)", color=:blue,
    markersize=10, markershape=:star5)

scatter!(p_overview, [new_strength[idx_B]], [old_strength[idx_B]],
    label="Case B (Chlor↓ Ammonia↑)", color=:red,
    markersize=10, markershape=:star5)

savefig(p_overview, "selected_points_overview.png")
println("Saved selected_points_overview.png")

# ==========================================
# 6. Output CSV
# ==========================================
rows = []
for (case_label, idx, price_vec, gridCI_vec) in [
        ("A_ChlHigh_AmmLow", idx_A, price_A, gridCI_A),
        ("B_ChlLow_AmmHigh",  idx_B, price_B, gridCI_B)]

    for t in 1:T
        push!(rows, (
            case        = case_label,
            window_idx  = idx,
            time_step   = t,
            price       = price_vec[t],
            emission    = gridCI_vec[t],
            new_strength_chlor  = new_strength[idx],
            old_strength_ammonia = old_strength[idx],
            note        = t == 1 ? "★ SELECTED WINDOW" : ""
        ))
    end
end

out_df = DataFrame(rows)
CSV.write("selected_cases_price_emission.csv", out_df)
println("Saved selected_cases_price_emission.csv  ($(nrow(out_df)) rows)")
println("\nDone. Outputs:")
println("  case_A_analysis.png")
println("  case_B_analysis.png")
println("  selected_points_overview.png")
println("  selected_cases_price_emission.csv")
