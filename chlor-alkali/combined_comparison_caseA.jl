using JuMP, CPLEX, Plots, CSV, DataFrames, XLSX, Statistics

# Global plot defaults — no frame around any legend.
Plots.default(
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)

# ==========================================
# Paths — this file lives in the chlor-alkali root, so datasets/
# is a direct subdirectory (no "..").
# ==========================================
const REPO_ROOT = @__DIR__

# ==========================================
# Chlor-alkali physical constants
# ==========================================
const T_HOR      = 48
const I_MIN      = 0.0
const I_MAX      = 1e7
const RAMP_LIM   = 1.5e5
const NC         = 160.0
const ETA_F      = 0.95
const F_CONST    = 96485.0
const U_NOM      = 3.2
const ALPHA      = ETA_F * (NC / (2F_CONST)) * 3600.0 * 1e-3
const BETA       = (NC * U_NOM) / 1000.0
const D_GAS      = fill(150.0, T_HOR)

# ==========================================
# 1. Load shared datasets
# ==========================================
df_chlor   = CSV.read(joinpath(REPO_ROOT, "datasets",
                 "Tokyo_first_8000_correlation_strength_comparison.csv"), DataFrame)
df_ammonia = CSV.read(joinpath(REPO_ROOT, "datasets",
                 "total_factor_grouping_2022_Tokyo_DA.csv"), DataFrame)
df_prices  = DataFrame(XLSX.readtable(
                 joinpath(REPO_ROOT, "datasets", "Tokyo_price and emission intensity.xlsx"), 1))

n_rows = min(nrow(df_chlor), nrow(df_ammonia), 8000)
df_chlor   = df_chlor[1:n_rows, :]
df_ammonia = df_ammonia[1:n_rows, :]

new_strength = Float64.(df_chlor[!, :New_Strength])
old_strength = Float64.(df_ammonia[!, :Column2])

# ==========================================
# 2. Identify the analysis window
#    We still use the "Case B" window (chlor LOW / ammonia HIGH) for all
#    optimisation, but in the overview scatter (panel a) we highlight it
#    alone and *label it "Case A"*.
# ==========================================
score = new_strength .- old_strength
idx_A = argmax(score)
idx_B = argmin(score)

println("Selected window (Chlor LOW / Ammonia HIGH): window index = $idx_B")

function get_window(df_prices, numbegin)
    last_idx = numbegin + T_HOR - 1
    price  = Float64.(df_prices[!, "price(dollars/kwh)"][numbegin:last_idx])
    gridCI = Float64.(df_prices[!, "CO2 kg/kWh (LCA)"][numbegin:last_idx])
    return price, gridCI
end

price_B, gridCI_B = get_window(df_prices, idx_B)
time_axis = 1:T_HOR

# ==========================================
# 3. Chlor-alkali Pareto (weighted sum, continuous LP)
#    Identical to `pareto_weighted_sum` in case_study_analysis.jl:
#    plain weighted sum, weights linearly spaced in [0,1], output
#    sorted by cost. No normalisation, no perturbation, no line.
# ==========================================
function chlor_pareto(price, gridCI; n_pts=30)
    Tl = length(price)
    pareto_cost = Float64[]
    pareto_em   = Float64[]
    I_profiles  = Matrix{Float64}(undef, Tl, n_pts)

    for (k, w) in enumerate(range(0.0, 1.0, length=n_pts))
        mdl = Model(CPLEX.Optimizer);  set_silent(mdl)

        @variable(mdl, I_MIN <= I[1:Tl] <= I_MAX)
        @expression(mdl, P[t=1:Tl], BETA  * I[t])
        @expression(mdl, R[t=1:Tl], ALPHA * I[t])

        @constraint(mdl, [t=1:Tl], R[t] >= D_GAS[t])
        @constraint(mdl, sum(R[t] for t in 1:Tl) >= 9600)
        @constraint(mdl, [t=2:Tl], I[t] - I[t-1] <= RAMP_LIM)
        @constraint(mdl, [t=2:Tl], I[t-1] - I[t] <= RAMP_LIM)

        Jc = sum(price[t]  * P[t] for t in 1:Tl)
        Je = sum(gridCI[t] * P[t] for t in 1:Tl)
        @objective(mdl, Min, w * Jc + (1 - w) * Je)
        optimize!(mdl)

        push!(pareto_cost, value(Jc));  push!(pareto_em, value(Je))
        I_profiles[:, k] = value.(I)
    end

    ord = sortperm(pareto_cost)
    return pareto_cost[ord], pareto_em[ord], I_profiles[:, ord]
end

# ==========================================
# 4. Ammonia Pareto (weighted sum, MILP)
#    Normalised weighted sum (cost / emission have very different
#    magnitudes), eps-perturbed corner weights, output sorted by weight.
# ==========================================
function ammonia_solve_inner(price, gridCI, w_c, w_e)
    tl = length(price)
    il = 3
    rho_a  = [60.0, 0.8, 2.12]
    mmin_a = [9.99,  0.0,  750.0]
    mmax_a = [33.3, 2250.0, 1100.0]
    msmin  = [4035.0, 22601.0, 0.0]
    msmax  = [807197.0, 4.52e6, 0.0]
    nemax  = 15
    mnh3tar = 1000.0
    rmin_a  = 100.0;  rmax_a = 40.0;  rampt = 4;  damax_a = 1
    cnh3 = 1.46;  con_a = 50.0;  cel = 0.42;  cpsa = 0.10
    csh2 = 5.50;  csn2 = 0.14;   cbh2 = 2.30;  h2perh = 100.0
    h2fossil = 9.3
    mi0  = copy(msmin);  ne0 = 0;  mnh30 = mnh3tar;  da0 = [0, 0, 0]
    ceb  = price;  ces = ceb .- 0.03;  gridemm = gridCI
    wpow = zeros(tl)

    mdl = Model(CPLEX.Optimizer);  set_silent(mdl)

    @variable(mdl, da[-2:tl], Bin)
    @variable(mdl, de[1:tl] >= 0, Int)
    @variable(mdl, ms[1:il, 0:tl] >= 0)
    @variable(mdl, mp[1:il, 0:tl] >= 0)
    @variable(mdl, mnh3dev[1:tl])
    @variable(mdl, ne[0:tl], Int)
    @variable(mdl, peb[1:tl] >= 0)
    @variable(mdl, pei[1:il, 1:tl] >= 0)
    @variable(mdl, pes[1:tl] >= 0)
    @variable(mdl, u[1:il] >= 0)
    @variable(mdl, bh2[1:tl] >= 0)

    @constraint(mdl, [i=1:tl], wpow[i]+peb[i]-pes[i]-sum(pei[j,i] for j=1:il)==0)
    @constraint(mdl, [i=1:il, j=1:tl], pei[i,j] == rho_a[i]*mp[i,j])
    @constraint(mdl, [i=2:il, j=1:tl], mp[i,j] >= mmin_a[i])
    @constraint(mdl, [i=2:il, j=1:tl], mp[i,j] <= mmax_a[i])
    @constraint(mdl, [j=1:tl], ne[j] >= 0)
    @constraint(mdl, [j=1:tl], ne[j] <= nemax)
    @constraint(mdl, [j=1:tl], mmin_a[1]*ne[j] <= mp[1,j])
    @constraint(mdl, [j=1:tl], mp[1,j] <= mmax_a[1]*ne[j])
    @constraint(mdl, [i=1:il, j=1:tl], ms[i,j] >= msmin[i])
    @constraint(mdl, [i=1:il, j=1:tl], ms[i,j] <= msmax[i])
    @constraint(mdl, [j=1:tl], mnh3dev[j] == mnh3tar - mp[3,j])
    @constraint(mdl, [j=1:tl], bh2[j] <= h2perh)
    @constraint(mdl, [j=1:tl], ms[1,j]==ms[1,j-1]+bh2[j]+mp[1,j]-(3.0/17.0)*mp[3,j])
    @constraint(mdl, [j=1:tl], ms[2,j]==ms[2,j-1]+mp[2,j]-(14.0/17.0)*mp[3,j])
    @constraint(mdl, [i=1:il], u[i]==ms[i,0]-ms[i,tl])
    @constraint(mdl, [j=1:tl], de[j] >= ne[j]-ne[j-1])
    @constraint(mdl, [j=1:tl], -da[j]*rmin_a <= mp[3,j]-mp[3,j-1])
    @constraint(mdl, [j=1:tl],  mp[3,j]-mp[3,j-1] <= da[j]*rmax_a)
    @constraint(mdl, [j=1:tl], sum(da[i] for i in (j+1-rampt):j) <= damax_a)
    @constraint(mdl, [i=1:il], ms[i,0] == mi0[i])
    @constraint(mdl, ne[0] == ne0)
    @constraint(mdl, mp[3,0] == mnh30)
    @constraint(mdl, [j=-2:0], da[j] == da0[j+3])

    costs_e = sum(ceb[j]*peb[j]-ces[j]*pes[j]+cnh3*mnh3dev[j]+
                  con_a*de[j]+cel*mp[1,j]+cbh2*bh2[j]+cpsa*mp[2,j] for j=1:tl) +
              csh2*u[1]+csn2*u[2]
    emms_e  = sum(sum(rho_a[jj]*mp[jj,i]*gridemm[i] for jj=1:il) for i=1:tl) +
              sum(h2fossil*bh2[i] for i=1:tl)

    @objective(mdl, Min, w_c * costs_e + w_e * emms_e)
    optimize!(mdl)

    nh3 = [value(mp[3,j]) for j=1:tl]
    h2  = [value(mp[1,j]) for j=1:tl]
    return value(costs_e), value(emms_e), nh3, h2
end

function ammonia_pareto(price, gridCI; n_pts=20)
    tl = length(price)

    Jc_pc, Je_pc, _, _ = ammonia_solve_inner(price, gridCI, 1.0,  1e-3)
    Jc_pe, Je_pe, _, _ = ammonia_solve_inner(price, gridCI, 1e-3, 1.0)
    sc = 1.0 / max(abs(Jc_pe - Jc_pc), 1e-9)
    se = 1.0 / max(abs(Je_pc - Je_pe), 1e-9)

    eps_w   = 1e-3
    weights = collect(range(eps_w, 1.0 - eps_w, length=n_pts))
    pc = zeros(n_pts);  pe = zeros(n_pts)
    nh3_mat = Matrix{Float64}(undef, tl, n_pts)
    h2_mat  = Matrix{Float64}(undef, tl, n_pts)

    for (k, w) in enumerate(weights)
        Jcv, Jev, nh3v, h2v = ammonia_solve_inner(price, gridCI,
                                                   w * sc, (1.0 - w) * se)
        pc[k] = Jcv;  pe[k] = Jev
        nh3_mat[:, k] = nh3v;  h2_mat[:, k] = h2v
    end

    ord = sortperm(weights, rev=true)
    return pc[ord], pe[ord], nh3_mat[:, ord], h2_mat[:, ord], weights[ord]
end

# ==========================================
# 5. Run optimisations
# ==========================================
N_PTS_CHLOR = 30          # match case_study_analysis.jl default
N_PTS_AMMO  = 20

println("Computing Chlor-Alkali Pareto frontier (window $idx_B)...")
pc_chlor, pe_chlor, I_mat_B = chlor_pareto(price_B, gridCI_B; n_pts=N_PTS_CHLOR)

println("Computing Ammonia Pareto frontier (window $idx_B)...")
pc_ammo, pe_ammo, nh3_mat_B, h2_mat_B, w_ammo =
    ammonia_pareto(price_B, gridCI_B; n_pts=N_PTS_AMMO)

# ==========================================
# 6. Select 4 representative Pareto points (same scheme for both
#    systems): the two visual corners + two intermediates picked by
#    greedy farthest-point so they don't overlap each other or the
#    corner anchors.
# ==========================================
function pick_4_spread(pc, pe)
    n = length(pc)
    c_rng = max(maximum(pc) - minimum(pc), 1.0)
    e_rng = max(maximum(pe) - minimum(pe), 1.0)
    nd(i, j) = sqrt(((pc[i]-pc[j])/c_rng)^2 + ((pe[i]-pe[j])/e_rng)^2)

    # Pure cost: among ties at min cost, take the one with MAX emission
    c_min = minimum(pc)
    cmin_set = findall(c -> c <= c_min + 1e-6 * c_rng, pc)
    idx_pc = cmin_set[argmax(pe[cmin_set])]

    # Pure emission: among ties at min emission, take the one with MAX cost
    e_min = minimum(pe)
    emin_set = findall(e -> e <= e_min + 1e-6 * e_rng, pe)
    idx_pe = emin_set[argmax(pc[emin_set])]

    # Greedy farthest-point for two intermediates
    function min_d_to(set_idx, i)
        i in set_idx ? -Inf : minimum(nd(i, j) for j in set_idx)
    end
    sel = [idx_pc, idx_pe]
    mid1 = argmax([min_d_to(sel, i) for i in 1:n])
    push!(sel, mid1)
    mid2 = argmax([min_d_to(sel, i) for i in 1:n])

    inters = sort([mid1, mid2], by = i -> pc[i])
    return [idx_pc, inters[1], inters[2], idx_pe]
end

i4_chlor = pick_4_spread(pc_chlor, pe_chlor)
i4_ammo  = pick_4_spread(pc_ammo,  pe_ammo)

lab4 = ["Pure cost", "Intermediate 1", "Intermediate 2", "Pure emission"]
col4 = [:royalblue, :seagreen, :darkorange, :crimson]
sty4 = [:solid, :dash, :dot, :dashdot]

# ==========================================
# 6b. Export every result to CSV so plotting can be done in Python
#     (run this Julia file once; then run plot_caseA.py).
# ==========================================
res_dir = joinpath(@__DIR__, "results_caseA")
isdir(res_dir) || mkdir(res_dir)

# (a) overview scatter — all windows + the single highlighted point
CSV.write(joinpath(res_dir, "overview.csv"),
    DataFrame(chlor_strength = new_strength,
              ammonia_strength = old_strength))
CSV.write(joinpath(res_dir, "highlight.csv"),
    DataFrame(label = ["Case A"],
              chlor_strength = [new_strength[idx_B]],
              ammonia_strength = [old_strength[idx_B]],
              window_index = [idx_B]))

# (d) price & emission for the selected window
CSV.write(joinpath(res_dir, "price_emission.csv"),
    DataFrame(time = collect(time_axis),
              price = price_B,
              gridCI = gridCI_B))

# (b)/(e) Pareto frontiers, with a `role` tag on the 4 chosen points
function pareto_df(pc, pe, i4, labels)
    role = fill("", length(pc))
    for (k, idx) in enumerate(i4)
        role[idx] = labels[k]
    end
    DataFrame(cost = pc, emission = pe, role = role)
end
CSV.write(joinpath(res_dir, "ammonia_pareto.csv"),
    pareto_df(pc_ammo, pe_ammo, i4_ammo, lab4))
CSV.write(joinpath(res_dir, "chlor_pareto.csv"),
    pareto_df(pc_chlor, pe_chlor, i4_chlor, lab4))

# (c) ammonia NH₃ / H₂ profiles at the 4 chosen points
ammo_prof = DataFrame(time = collect(time_axis))
for (k, idx) in enumerate(i4_ammo)
    ammo_prof[!, "nh3_$k"] = nh3_mat_B[:, idx]
    ammo_prof[!, "h2_$k"]  = h2_mat_B[:, idx]
end
CSV.write(joinpath(res_dir, "ammonia_profiles.csv"), ammo_prof)

# (f) chlor-alkali current profiles at the 4 chosen points
chlor_prof = DataFrame(time = collect(time_axis))
for (k, idx) in enumerate(i4_chlor)
    chlor_prof[!, "I_$k"] = I_mat_B[:, idx]
end
CSV.write(joinpath(res_dir, "chlor_profiles.csv"), chlor_prof)

# point index → label mapping (k = 1..4)
CSV.write(joinpath(res_dir, "labels.csv"),
    DataFrame(k = 1:4, label = lab4))

println("Exported all CSV results to: $res_dir")

# Shared font sizes to keep all panels readable without overlap
const FS_TITLE  = 12
const FS_GUIDE  = 11
const FS_TICK   = 9
const FS_LEGEND = 9

# Y-axis configuration for the 4-stack sub-panels (3 ticks, shared scale)
const AMMO_YTICKS  = [0.0, 500.0, 1000.0]
const AMMO_YLIMS   = (-30.0, 1180.0)
const CHLOR_YTICKS = [1.0e5, 2.0e5, 3.0e5]
const CHLOR_YLIMS  = (3.0e4, 3.8e5)

# ==========================================
# 7.  p1 — Overview scatter  (Row 1, Col 1)
#     Only ONE window highlighted (the chlor-LOW / ammonia-HIGH window),
#     labelled "Case A" per request.
# ==========================================
p1 = scatter(new_strength, old_strength,
    xlabel="Chlor-Alkali Strength",
    ylabel="Ammonia Strength",
    title="(a) Selected Cases Overview",
    label="All 8000 windows", color=:lightgray, markersize=2,
    markerstrokewidth=0, alpha=0.5, legend=:bottomleft,
    titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
    tickfontsize=FS_TICK, legendfontsize=FS_LEGEND)
scatter!(p1, [new_strength[idx_B]], [old_strength[idx_B]],
    label="Case A", color=:crimson,
    markersize=9, markershape=:star5)

# ==========================================
# 8.  p2 — Ammonia Pareto frontier  (Row 1, Col 2)
# ==========================================
p2 = scatter(pc_ammo, pe_ammo,
    xlabel="Cost Objective (\$)",
    ylabel="Emission Objective (kg CO₂)",
    title="(b) Ammonia Pareto Frontier",
    label="Pareto pts", color=:gray, markersize=4, alpha=0.5,
    legend=:topright,
    titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
    tickfontsize=FS_TICK, legendfontsize=FS_LEGEND)
plot!(p2, pc_ammo, pe_ammo, label="", color=:gray, alpha=0.4, linewidth=1)
for (i, idx) in enumerate(i4_ammo)
    scatter!(p2, [pc_ammo[idx]], [pe_ammo[idx]],
        color=col4[i], label=lab4[i], markersize=9, markershape=:diamond)
end

# ==========================================
# 9.  p3 — Ammonia NH₃ & H₂ profiles  (Row 1, Col 3)
# ==========================================
p3_subs = Plots.Plot[]
for (i, idx) in enumerate(i4_ammo)
    is_first = (i == 1)
    is_last  = (i == 4)
    is_unit  = (i == 2)   # carries the y-axis unit label
    pi_ = plot(time_axis, nh3_mat_B[:, idx],
        label="NH₃", color=col4[i],
        linewidth=2, linestyle=:solid,
        title  = is_first ? "(c) Ammonia Profiles" : "",
        ylabel = is_unit ? "kg/h" : "",
        xlabel = is_last ? "Time Step (h)" : "",
        xticks = is_last ? :auto : :none,
        yticks = AMMO_YTICKS,
        ylims  = AMMO_YLIMS,
        framestyle = :axes,                      # L-shape: no top, no right border
        top_margin    = is_first ? 3Plots.mm : -6Plots.mm,
        bottom_margin = is_last  ? 5Plots.mm : -6Plots.mm,
        legend = :topright,
        titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
        tickfontsize=FS_TICK,  legendfontsize=FS_LEGEND)
    plot!(pi_, time_axis, h2_mat_B[:, idx],
        label="H₂", color=col4[i],
        linewidth=1.5, linestyle=:dash)
    push!(p3_subs, pi_)
end

# ==========================================
# 10. p4 — Price & emission dual-axis  (Row 2, Col 1)
# ==========================================
p4 = plot(time_axis, price_B,
    xlabel="Time Step (h)",
    ylabel="Price (\$/kWh)",
    title="(d) Case A: Price & Emission",
    label="Price", color=:steelblue, linewidth=2,
    legend=:bottomleft,
    titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
    tickfontsize=FS_TICK, legendfontsize=FS_LEGEND)
# Dummy series on the MAIN axis so the legend at bottom-left lists both.
plot!(p4, [NaN], [NaN],
    label="Emission", color=:firebrick, linewidth=2, linestyle=:dash)

p4b = twinx(p4)
plot!(p4b, time_axis, gridCI_B,
    ylabel="Emission (kg CO₂/kWh)",
    label="", color=:firebrick, linewidth=2, linestyle=:dash,
    legend=false,
    guidefontsize=FS_GUIDE, tickfontsize=FS_TICK)

# ==========================================
# 11. p5 — Chlor-alkali Pareto frontier  (Row 2, Col 2)
# ==========================================
p5 = scatter(pc_chlor, pe_chlor,
    xlabel="Cost Objective (\$)",
    ylabel="Emission Objective (kg CO₂)",
    title="(e) Chlor-Alkali Pareto Frontier",
    label="Pareto points", color=:purple, markersize=5,
    legend=:topright,
    titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
    tickfontsize=FS_TICK, legendfontsize=FS_LEGEND)
for (i, idx) in enumerate(i4_chlor)
    scatter!(p5, [pc_chlor[idx]], [pe_chlor[idx]],
        color=col4[i], label=lab4[i], markersize=10, markershape=:diamond,
        markerstrokewidth=1, markerstrokecolor=:black)
end

# ==========================================
# 12. p6 — Chlor-alkali current profiles  (Row 2, Col 3)
# ==========================================
p6_subs = Plots.Plot[]
for (i, idx) in enumerate(i4_chlor)
    is_first = (i == 1)
    is_last  = (i == 4)
    is_unit  = (i == 2)
    leg_pos = (i >= 3) ? :topleft : :topright
    pi_ = plot(time_axis, I_mat_B[:, idx],
        label=lab4[i], color=col4[i],
        linewidth=2, linestyle=sty4[i],
        title  = is_first ? "(f) Chlor-Alkali Profiles" : "",
        ylabel = is_unit ? "I (A)" : "",
        xlabel = is_last ? "Time Step (h)" : "",
        xticks = is_last ? :auto : :none,
        yticks = CHLOR_YTICKS,
        ylims  = CHLOR_YLIMS,
        framestyle = :axes,
        top_margin    = is_first ? 3Plots.mm : -6Plots.mm,
        bottom_margin = is_last  ? 5Plots.mm : -6Plots.mm,
        legend = leg_pos,
        titlefontsize=FS_TITLE, guidefontsize=FS_GUIDE,
        tickfontsize=FS_TICK,  legendfontsize=FS_LEGEND)
    push!(p6_subs, pi_)
end

# ==========================================
# 13. Assemble 2 × 3 figure with nested 4-stacks in col 3
# ==========================================
lay = @layout [a b [c1; c2; c3; c4]
               d e [f1; f2; f3; f4]]

fig = plot(p1, p2, p3_subs[1], p3_subs[2], p3_subs[3], p3_subs[4],
           p4, p5, p6_subs[1], p6_subs[2], p6_subs[3], p6_subs[4],
    layout = lay,
    size   = (2000, 1200),
    left_margin   = 12Plots.mm,
    bottom_margin = 8Plots.mm,
    top_margin    = 6Plots.mm,
    right_margin  = 12Plots.mm,
    plot_titlefontsize = 13)

out_path = joinpath(@__DIR__, "combined_comparison_caseA.png")
savefig(fig, out_path)
println("Saved: $out_path")
