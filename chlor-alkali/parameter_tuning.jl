using JuMP, CPLEX
using CSV, DataFrames
using Plots
using XLSX
using Statistics
using ORCA

# ==========================================
# Fixed Plant Parameters
# ==========================================
const T    = 48
const Nc   = 160.0
const ηF   = 0.95
const F    = 96485.0
const U_nom = 3.2
const α    = ηF * (Nc / (2F)) * 3600.0 * 1e-3   # chlorine production coefficient
const β    = (Nc * U_nom) / 1000.0                # linear power coefficient
# Cell resistance (Ω): quadratic power term P(I) = β*I + β2*I²
# U_cell = U_nom + r_cell*I  →  P = (Nc/1000)*(U_nom*I + r_cell*I²)
# r_cell ≈ 1e-7 Ω gives ~0.5V overpotential at I_nom=5MA (physically realistic)
const r_cell = 1e-7
const β2   = (Nc * r_cell) / 1000.0               # coefficient of I² term
const Dgas = fill(150.0, T)
const I_nom = 5e6       # nominal current (A) — used as reference

# ==========================================
# Window to Study
# ==========================================
const WINDOW = 2348

# ==========================================
# Load Price / Emission Data
# ==========================================
df_prices = DataFrame(XLSX.readtable(
    "datasets/Tokyo_price and emission intensity.xlsx", 1))

function get_window(numbegin)
    last_index = numbegin + T - 1
    price  = Float64.(df_prices[!, "price(dollars/kwh)"][numbegin:last_index])
    gridCI = Float64.(df_prices[!, "CO2 kg/kWh (LCA)"][numbegin:last_index])
    return price, gridCI
end

price_ref, gridCI_ref = get_window(WINDOW)

println("Window $WINDOW  |  price range: [$(round(minimum(price_ref),digits=5)), $(round(maximum(price_ref),digits=5))]  |  gridCI range: [$(round(minimum(gridCI_ref),digits=4)), $(round(maximum(gridCI_ref),digits=4))]")

# ==========================================
# Core Solver: successive linearization + ORCA + weighted-sum Pareto
# ==========================================

# Step 1: solve single-objective LP (minimize cost) to get operating point I*
function solve_reference_point(price, gridCI, Imax_val, ramp_val)
    Tlocal   = length(price)
    Imin_val = 0.0
    m = Model(CPLEX.Optimizer); set_silent(m)
    @variable(m, I[1:Tlocal] >= Imin_val)
    @constraint(m, [t=1:Tlocal], I[t] <= Imax_val)   # explicit upper bound
    @expression(m, R[t=1:Tlocal], α * I[t])
    @constraint(m, [t=1:Tlocal],  R[t] >= Dgas[t])
    @constraint(m, sum(R[t] for t in 1:Tlocal) >= 9600)
    @constraint(m, [t=2:Tlocal],  I[t] - I[t-1] <= ramp_val)
    @constraint(m, [t=2:Tlocal],  I[t-1] - I[t] <= ramp_val)
    # Minimize cost using full quadratic power
    @objective(m, Min, sum(price[t] * (β * I[t] + β2 * I[t]^2) for t in 1:Tlocal))
    optimize!(m)
    return value.(I)   # I*_t for each time step
end

# Step 2: linearize P(I) = β*I + β2*I² at I* → P(I) ≈ (β + 2β2*I*_t)*I_t + const
# The linear coefficient at time t becomes: c_t = β + 2*β2*I*_t
# (constants drop out of gradient, so ORCA sees: Jobj row = [c_t * price_t] or [c_t * gridCI_t])
function linearized_coeffs(I_star)
    return [β + 2.0 * β2 * I_star[t] for t in 1:length(I_star)]
end

function run_case(price, gridCI, Imax_val, ramp_val;
                  n_pareto=30, label="")

    Tlocal   = length(price)
    Imin_val = 0.0

    # --- Step 1: get operating point I* for this (Imax, ramp) combination ---
    I_star = solve_reference_point(price, gridCI, Imax_val, ramp_val)
    c_lin  = linearized_coeffs(I_star)   # locally-linearized power coefficients

    println("    I* range: [$(round(minimum(I_star)/I_nom,digits=3)), $(round(maximum(I_star)/I_nom,digits=3))] × Inom")

    # --- Step 2: ORCA on the linearized model ---
    # Jobj rows now = [c_lin[t]*price[t]] and [c_lin[t]*gridCI[t]]
    # which genuinely differ across (Imax,ramp) because I* does.
    m_orca = Model(CPLEX.Optimizer); set_silent(m_orca)
    # Declare variables as free (or with a safe lower bound of 0) so that
    # all operative bounds become explicit constraint rows that ORCA's
    # linear_jac extractor can see as Jacobian rows.
    @variable(m_orca, I_o[1:Tlocal] >= 0.0)
    @constraint(m_orca, ub[t=1:Tlocal], I_o[t] <= Imax_val)   # Imax bound → constraint row
    @expression(m_orca, R_o[t=1:Tlocal], α * I_o[t])
    @constraint(m_orca, [t=1:Tlocal],    R_o[t] >= Dgas[t])
    @constraint(m_orca, sum(R_o[t] for t in 1:Tlocal) >= 9600)
    @constraint(m_orca, [t=2:Tlocal],    I_o[t] - I_o[t-1] <= ramp_val)
    @constraint(m_orca, [t=2:Tlocal],    I_o[t-1] - I_o[t] <= ramp_val)
    # Linearized objectives: coefficient at t = c_lin[t]
    @expression(m_orca, Jcost_lin, sum(price[t]  * c_lin[t] * I_o[t] for t in 1:Tlocal))
    @expression(m_orca, Jem_lin,   sum(gridCI[t] * c_lin[t] * I_o[t] for t in 1:Tlocal))
    res_orca      = ORCA.main(m_orca, [Jcost_lin, Jem_lin], 2)
    corr_strength = res_orca.adjMatrix[1, 2]

    # --- Step 3: weighted-sum Pareto frontier using full quadratic power ---
    pareto_cost = Float64[]
    pareto_em   = Float64[]
    I_profiles  = Matrix{Float64}(undef, Tlocal, n_pareto)

    for (k, w) in enumerate(range(0.0, 1.0, length=n_pareto))
        m = Model(CPLEX.Optimizer); set_silent(m)
        @variable(m, I[1:Tlocal] >= 0.0)
        @constraint(m, [t=1:Tlocal], I[t] <= Imax_val)   # explicit upper bound
        @expression(m, R[t=1:Tlocal], α * I[t])
        @constraint(m, [t=1:Tlocal],    R[t] >= Dgas[t])
        @constraint(m, sum(R[t] for t in 1:Tlocal) >= 9600)
        @constraint(m, [t=2:Tlocal],    I[t] - I[t-1] <= ramp_val)
        @constraint(m, [t=2:Tlocal],    I[t-1] - I[t] <= ramp_val)
        Jc = sum(price[t]  * (β * I[t] + β2 * I[t]^2) for t in 1:Tlocal)
        Je = sum(gridCI[t] * (β * I[t] + β2 * I[t]^2) for t in 1:Tlocal)
        @objective(m, Min, w * Jc + (1 - w) * Je)
        optimize!(m)
        push!(pareto_cost, value(Jc))
        push!(pareto_em,   value(Je))
        I_profiles[:, k] = value.(I)
    end

    ord = sortperm(pareto_cost)
    return (
        corr       = corr_strength,
        pcost      = pareto_cost[ord],
        pem        = pareto_em[ord],
        I_profiles = I_profiles[:, ord],
        I_star     = I_star,
        Imax       = Imax_val,
        ramp       = ramp_val,
        label      = label
    )
end

# ==========================================
# Parameter Grid
# ==========================================
# Imax options: multiples of I_nom (5e6)
#   Baseline: 2.0×  (1e7) — already too loose
#   Realistic: 1.0× to 1.4× I_nom
Imax_options = [
    (0.5 * I_nom, "Imax=0.5×Inom (5.0MA)"),
    (0.8 * I_nom, "Imax=0.8×Inom (6.0MA)"),
    (1.4 * I_nom, "Imax=1.4×Inom (7.0MA)"),
    (0.1 * I_nom, "Imax=0.1×Inom (10MA) [baseline]"),
]

# Ramp options: fraction of I_nom per time step
#   Baseline: 1.5e5 A/step ≈ 3% of I_nom per step
#   Tight: 1–5% of I_nom per step is common in real electrolyzers
ramp_options = [
    (0.005 * I_nom, "ramp=0.5%·Inom (50kA/step)"),
    (0.02 * I_nom, "ramp=2%·Inom (100kA/step)"),
    (0.03 * I_nom, "ramp=3%·Inom (150kA/step) [baseline]"),
    (0.05 * I_nom, "ramp=5%·Inom (250kA/step)"),
]

println("\nRunning parameter sweep on window $WINDOW ...")
println("$(length(Imax_options)) Imax × $(length(ramp_options)) ramp = $(length(Imax_options)*length(ramp_options)) cases\n")

results = []
for (Imax_val, Imax_lbl) in Imax_options
    for (ramp_val, ramp_lbl) in ramp_options
        lbl = "$(Imax_lbl) | $(ramp_lbl)"
        print("  Running: $lbl ... ")
        r = run_case(price_ref, gridCI_ref, Imax_val, ramp_val; label=lbl)
        println("corr = $(round(r.corr, digits=6))")
        push!(results, r)
    end
end

# ==========================================
# Summary Table
# ==========================================
println("\n" * "=" ^ 80)
println("PARAMETER SWEEP SUMMARY  (window $WINDOW)")
println("=" ^ 80)
println(rpad("Imax", 12), rpad("Ramp (A/step)", 22), rpad("Corr Strength", 16),
        rpad("Cost range", 20), "Emission range")
println("-" ^ 80)

summary_rows = []
for r in results
    Imax_frac = round(r.Imax / I_nom, digits=2)
    ramp_frac = round(r.ramp / I_nom * 100, digits=1)
    cost_spread  = round(maximum(r.pcost) - minimum(r.pcost), digits=2)
    em_spread    = round(maximum(r.pem)   - minimum(r.pem),   digits=2)
    println(rpad("$(Imax_frac)×Inom", 12),
            rpad("$(ramp_frac)%·Inom", 22),
            rpad(round(r.corr, digits=6), 16),
            rpad(cost_spread, 20),
            em_spread)
    push!(summary_rows, (
        Imax_fraction = Imax_frac,
        ramp_fraction_pct = ramp_frac,
        corr_strength = r.corr,
        cost_spread   = cost_spread,
        em_spread     = em_spread,
        label         = r.label
    ))
end
println("=" ^ 80)

CSV.write("parameter_tuning_summary.csv", DataFrame(summary_rows))
println("\nSaved parameter_tuning_summary.csv")

# ==========================================
# Plots
# ==========================================
time_axis = 1:T

# ---------- 1. Pareto frontiers — vary Imax, fix ramp at baseline (3%) ----------
baseline_ramp = 0.03 * I_nom
ramp_subset = filter(r -> r.ramp ≈ baseline_ramp, results)

p_pareto_Imax = plot(
    title="Pareto Frontiers (ramp=3%·Inom fixed, vary Imax)\nWindow $WINDOW",
    xlabel="Cost Objective (normalised units)",
    ylabel="Emission Objective",
    legend=:topright)
colors_Imax = [:black, :steelblue, :darkorange, :firebrick]
for (r, c) in zip(ramp_subset, colors_Imax)
    Imax_lbl = "$(round(r.Imax/I_nom, digits=1))×Inom"
    plot!(p_pareto_Imax, r.pcost, r.pem,
        label="$Imax_lbl  (corr=$(round(r.corr,digits=4)))",
        color=c, linewidth=2, marker=:circle, markersize=4)
end
savefig(p_pareto_Imax, "tuning_pareto_vary_Imax.png")
println("Saved tuning_pareto_vary_Imax.png")

# ---------- 2. Pareto frontiers — vary ramp, fix Imax at median value ----------
Imax_sorted = sort(unique([r.Imax for r in results]))
target_Imax = Imax_sorted[max(1, length(Imax_sorted) ÷ 2)]  # pick middle Imax
Imax_subset = filter(r -> r.Imax ≈ target_Imax, results)

p_pareto_ramp = plot(
    title="Pareto Frontiers (Imax=1.2×Inom fixed, vary ramp)\nWindow $WINDOW",
    xlabel="Cost Objective",
    ylabel="Emission Objective",
    legend=:topright)
colors_ramp = [:black, :steelblue, :darkorange, :firebrick]
for (r, c) in zip(Imax_subset, colors_ramp)
    ramp_lbl = "$(round(r.ramp/I_nom*100, digits=1))%·Inom"
    plot!(p_pareto_ramp, r.pcost, r.pem,
        label="$ramp_lbl  (corr=$(round(r.corr,digits=4)))",
        color=c, linewidth=2, marker=:circle, markersize=4)
end
savefig(p_pareto_ramp, "tuning_pareto_vary_ramp.png")
println("Saved tuning_pareto_vary_ramp.png")

# ---------- 3. Current profiles for baseline vs recommended ----------
# Baseline: largest Imax + loosest ramp (most unconstrained)
# Recommended: pick case with lowest corr strength (most competing — most interesting)
r_baseline = results[argmax([r.Imax + r.ramp for r in results])]
r_rec      = results[argmin([r.corr for r in results])]

function current_profile_plot(r, title_str)
    p = plot(title=title_str,
             xlabel="Time Step (h)", ylabel="Current I (A)",
             legend=false)
    for j in 1:size(r.I_profiles, 2)
        plot!(p, time_axis, r.I_profiles[:, j],
              alpha=0.35, color=:darkorange, linewidth=1)
    end
    # Overlay Imax reference line
    hline!(p, [r.Imax], color=:red, linestyle=:dash, linewidth=1.5,
           label="Imax")
    hline!(p, [I_nom],  color=:gray, linestyle=:dot, linewidth=1.5,
           label="Inom")
    return p
end

p_curr_base = current_profile_plot(r_baseline,
    "Baseline (Imax=$(round(r_baseline.Imax/I_nom,digits=2))×Inom, ramp=$(round(r_baseline.ramp/I_nom*100,digits=1))%)  corr=$(round(r_baseline.corr,digits=4))")
p_curr_rec  = current_profile_plot(r_rec,
    "Most Competing (Imax=$(round(r_rec.Imax/I_nom,digits=2))×Inom, ramp=$(round(r_rec.ramp/I_nom*100,digits=1))%)  corr=$(round(r_rec.corr,digits=4))")

# Also plot price & emission for context
p_price = plot(time_axis, price_ref,
    xlabel="Time Step (h)", ylabel="Price (\$/kWh)",
    title="Window $WINDOW: Power Price",
    label="Price", color=:steelblue, linewidth=2)

p_em = plot(time_axis, gridCI_ref,
    xlabel="Time Step (h)", ylabel="Emission (kg CO₂/kWh)",
    title="Window $WINDOW: Emission Intensity",
    label="Emission", color=:firebrick, linewidth=2)

fig_compare = plot(p_price, p_em, p_curr_base, p_curr_rec,
    layout=(2,2), size=(1200, 850),
    plot_title="Baseline vs Recommended Parameters — Window $WINDOW")
savefig(fig_compare, "tuning_baseline_vs_recommended.png")
println("Saved tuning_baseline_vs_recommended.png")

# ---------- 4. Heatmap of correlation strength over parameter grid ----------
Imax_vals  = sort(unique([r.Imax  for r in results]))
ramp_vals  = sort(unique([r.ramp  for r in results]))
corr_matrix = [
    filter(r -> r.Imax ≈ im && r.ramp ≈ ra, results)[1].corr
    for im in Imax_vals, ra in ramp_vals
]

Imax_labels = ["$(round(v/I_nom,digits=1))×Inom" for v in Imax_vals]
ramp_labels = ["$(round(v/I_nom*100,digits=1))%" for v in ramp_vals]

p_heatmap = heatmap(ramp_labels, Imax_labels, corr_matrix,
    xlabel="Ramp Rate (% of Inom per step)",
    ylabel="Imax (× Inom)",
    title="Correlation Strength Heatmap\n(Cost vs Emission, window $WINDOW)",
    color=:viridis, clims=(minimum(corr_matrix)*0.995, 1.0),
    annotate_kw=(fontsize=9,))

# Annotate each cell
for (i, im) in enumerate(Imax_vals)
    for (j, ra) in enumerate(ramp_vals)
        v = corr_matrix[i, j]
        annotate!(p_heatmap, j, i, text("$(round(v,digits=4))", 8, :white))
    end
end

savefig(p_heatmap, "tuning_corr_heatmap.png")
println("Saved tuning_corr_heatmap.png")

println("\n✓ All done. Outputs:")
println("  parameter_tuning_summary.csv")
println("  tuning_pareto_vary_Imax.png")
println("  tuning_pareto_vary_ramp.png")
println("  tuning_baseline_vs_recommended.png")
println("  tuning_corr_heatmap.png")
