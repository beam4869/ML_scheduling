using JuMP, CPLEX
using CSV, DataFrames
using Plots, ORCA
using XLSX
using Statistics
import MathOptInterface as MOI
import JSON

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
const T = 48
const START_RANGE = 1:8000

# Load external data sources
df_prices = DataFrame(XLSX.readtable("Tokyo_price and emission intensity.xlsx", 1))
df_metrics = CSV.read("Tokyopareto_metric_comparison_1_to_672.csv", DataFrame)
# df_areas = CSV.read("Tokyo_pareto_area_comparison_1_to_672.csv", DataFrame)

# Storage for results
new_correlation_strengths = Float64[]
group_categories = String[]
price_gridCI_pearson = Float64[]

# Plant physical parameters (Global)
const I_nom = 5e6
const k_surge = 0.8
const Nc = 160.0
const ηF = 0.95
const F = 96485.0
const U_nom = 3.2
const γ = 0.99
const M0 = 0.0
const Mmax = 5e4
const Imin = 0.0
const Imax = 1e7
const n_drag = 4.0
const c_op = 1.0
const c_ramp = 1.0
const ramp_limit = 1.5e5
const α = ηF * (Nc / (2F)) * 3600.0 * 1e-3
const β = (Nc * U_nom) / 1000.0
const Dgas = fill(150.0, T)

# ==========================================
# 2. Main Loop (1 to 672)
# ==========================================
println("Starting optimization loop...")

for numbegin in START_RANGE
    last_index = numbegin + T - 1
    
    # Data windowing
    price = Float64.(df_prices[!, "price(dollars/kwh)"][numbegin:last_index])
    gridCI = Float64.(df_prices[!, "CO2 kg/kWh (LCA)"][numbegin:last_index])
    Tlocal = length(price)

    push!(price_gridCI_pearson, cor(price, gridCI))

    # Model Setup
    model = Model(CPLEX.Optimizer)
    set_silent(model)

    @variable(model, Imin <= I[1:Tlocal] <= Imax)
    @expression(model, R_expr[t=1:Tlocal], α * I[t])
    @expression(model, P[t=1:Tlocal], β * I[t])

    @constraint(model, [t=1:Tlocal], R_expr[t] >= Dgas[t])
    @constraint(model, sum(R_expr[t] for t in 1:Tlocal) >= 9600)
    @constraint(model, [t=2:Tlocal], I[t] - I[t-1] <= ramp_limit)
    @constraint(model, [t=2:Tlocal], I[t-1] - I[t] <= ramp_limit)

    @expression(model, Jcost, sum(price[t]  * P[t] for t in 1:Tlocal))
    @expression(model, Jem,   sum(gridCI[t] * P[t] for t in 1:Tlocal))

    # ORCA Run
    res = ORCA.main(model, [Jcost, Jem], 2)
    
    # Record Correlation Strength (adjMatrix)
    push!(new_correlation_strengths, res.adjMatrix[1,2])
    
    # Logic for Color Coding from Column 1
    # Example Column1: "[,]"
    col1_str = df_metrics[numbegin, :Column1]
    groups = JSON.parse(col1_str)
    
    # Check if 1 (Cost) and 2 (Emission) are in the same inner group
    is_correlating = false
    for g in groups
        if 1 in g && 2 in g
            is_correlating = true
            break
        end
    end
    
    push!(group_categories, is_correlating ? "Correlating" : "Competing")
    
    if numbegin % 50 == 0
        println("Processed $numbegin / 672...")
    end
end

# ==========================================
# 3. Data Visualization
# ==========================================
println("Generating plot...")

# Combine data for plotting
p_df = DataFrame(
    New_Strength = new_correlation_strengths,
    Old_Strength = df_metrics[!, :Column2],
    area_metric = df_metrics[!, :area_score],
    Price_GridCI_Pearson = price_gridCI_pearson,
    Status = group_categories
)

CSV.write("Tokyo_first_8000_correlation_strength_comparison.csv", p_df)
println("Saved table to 'Tokyo_first_8000_correlation_strength_comparison.csv'")

# Pearson correlation coefficients
corr_input = dropmissing(select(p_df, [:New_Strength, :Old_Strength, :area_metric]))
corr_input_r2 = dropmissing(select(p_df, [:New_Strength, :Old_Strength, :Price_GridCI_Pearson]))

if nrow(corr_input) >= 2
    corr_matrix = cor(Matrix(corr_input))

    println("\nPearson correlation matrix (New_Strength, Old_Strength, area_metric):")
    println(corr_matrix)

    r_new_old = cor(corr_input.New_Strength, corr_input.Old_Strength)
    r_new_area = cor(corr_input.New_Strength, corr_input.area_metric)
    r_old_area = cor(corr_input.Old_Strength, corr_input.area_metric)

    println("\nPairwise Pearson correlation coefficients:")
    println("r(New_Strength, Old_Strength) = $(round(r_new_old, digits=6))")
    println("r(New_Strength, area_metric)  = $(round(r_new_area, digits=6))")
    println("r(Old_Strength, area_metric)  = $(round(r_old_area, digits=6))")
else
    @warn "Not enough non-missing rows to compute Pearson correlation." nrows=nrow(corr_input)
end

if nrow(corr_input_r2) >= 2
    r_new_old_r2 = cor(corr_input_r2.New_Strength, corr_input_r2.Old_Strength)^2
    r_new_pricegridci_r2 = cor(corr_input_r2.New_Strength, corr_input_r2.Price_GridCI_Pearson)^2
    r_old_pricegridci_r2 = cor(corr_input_r2.Old_Strength, corr_input_r2.Price_GridCI_Pearson)^2

    println("\nR² values:")
    println("R²(New_Strength, Old_Strength)            = $(round(r_new_old_r2, digits=6))")
    println("R²(New_Strength, Price_GridCI_Pearson)    = $(round(r_new_pricegridci_r2, digits=6))")
    println("R²(Old_Strength, Price_GridCI_Pearson)    = $(round(r_old_pricegridci_r2, digits=6))")
else
    @warn "Not enough non-missing rows to compute R² values." nrows=nrow(corr_input_r2)
end

# Scatter plot
p = scatter(
    p_df.New_Strength, 
    p_df.Old_Strength, 
    group = p_df.Status,
    color = [:red :blue], # Competing = Red, Correlating = Blue
    xlabel = "New Correlation Strength (res.adjMatrix)",
    ylabel = "CSV Column 2 Strength",
    title = "Comparison of Correlation Strengths",
    legend = :outertopright,
    markersize = 4,
    markerstrokewidth = 0.5,
    alpha = 0.7
)

# savefig(p, "correlation_comparison_plot.png")
println("Done. Plot saved as 'correlation_comparison_plot.png'")
p2 = scatter(
    p_df.New_Strength, 
    p_df.area_metric, 
    group = p_df.Status,
    color = [:red :blue], # Competing = Red, Correlating = Blue
    xlabel = "New Correlation Strength (res.adjMatrix)",
    ylabel = "Area Metric",
    title = "Comparison of Correlation Strengths",
    legend = :outertopright,
    markersize = 4,
    markerstrokewidth = 0.5,
    alpha = 0.7
)
# savefig(p2, "correlation_vs_area_metric_plot.png")

p3 = scatter(
    p_df.New_Strength,
    p_df.Price_GridCI_Pearson,
    group = p_df.Status,
    color = [:red :blue], # Competing = Red, Correlating = Blue
    xlabel = "New Correlation Strength (res.adjMatrix)",
    ylabel = "Pearson Correlation (price, gridCI)",
    title = "New Correlation Strength vs Price-GridCI Pearson",
    legend = :outertopright,
    markersize = 4,
    markerstrokewidth = 0.5,
    alpha = 0.7
)
# savefig(p3, "new_strength_vs_price_gridci_pearson_plot.png")