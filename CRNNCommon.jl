"""
    CRNNCommon.jl

Shared utilities and functions for CRNN (Chemical Reaction Neural Network) models.
Reduces code duplication across different model configurations.
"""

using Flux, OrdinaryDiffEq, DiffEqSensitivity
using LinearAlgebra, Statistics
using Flux.Losses: mae, mse
using Printf, BSON: @save, @load
using DelimitedFiles, YAML
import Plots: plot, plot!, xlabel!, ylabel!, title!, annotate!, png

# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

"""
    CRNNConfig

Configuration structure for CRNN models.
"""
mutable struct CRNNConfig
    # Model architecture
    ns::Int64              # Number of species
    nr::Int64              # Number of reactions
    nc::Int64              # Number of catalysts (0 for no catalyst)
    
    # Training parameters
    lb::Float64            # Lower bound for clamping
    n_epoch::Int64         # Total training epochs
    n_plot::Int64          # Plotting frequency
    grad_max::Float64      # Gradient clipping threshold
    maxiters::Int64        # Max ODE solver iterations
    
    # Learning rate parameters
    lr_max::Float64        # Maximum learning rate
    lr_min::Float64        # Minimum learning rate
    lr_decay::Float64      # Learning rate decay factor
    lr_decay_step::Int64   # Steps between LR decays
    w_decay::Float64       # Weight decay (L2 regularization)
    
    # Paths
    expr_name::String      # Experiment name
    fig_path::String       # Figure save path
    ckpt_path::String      # Checkpoint save path
    
    # Flags
    is_restart::Bool       # Restart from checkpoint
end

"""
    CRNNConfig(config_file::String)
    
Load configuration from YAML file.
"""
function CRNNConfig(config_file::String)
    conf = YAML.load_file(config_file)
    
    expr_name = conf["expr_name"]
    results_dir = get(conf, "results_dir", "./results")
    
    fig_path = joinpath(results_dir, expr_name, "figs")
    ckpt_path = joinpath(results_dir, expr_name, "checkpoint")
    
    return CRNNConfig(
        Int64(conf["ns"]),
        Int64(conf["nr"]),
        Int64(get(conf, "nc", 0)),
        Float64(conf["lb"]),
        Int64(conf["n_epoch"]),
        Int64(conf["n_plot"]),
        Float64(conf["grad_max"]),
        Int64(conf["maxiters"]),
        Float64(conf["lr_max"]),
        Float64(conf["lr_min"]),
        Float64(conf["lr_decay"]),
        Int64(conf["lr_decay_step"]),
        Float64(conf["w_decay"]),
        expr_name,
        fig_path,
        ckpt_path,
        Bool(get(conf, "is_restart", false))
    )
end

# ============================================================================
# DATA LOADING
# ============================================================================

"""
    load_exp(filename::String; enforce_monotone::Bool=true)
    
Load experimental data from file and normalize.
Optional: enforce monotone decrease to remove TGA artifacts.
"""
function load_exp(filename::String; enforce_monotone::Bool=true)
    exp_data = readdlm(filename)
    
    # Remove duplicate time points
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    
    # Normalize third column (mass)
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    
    # Enforce monotone decrease (optional)
    if enforce_monotone
        for i in 2:size(exp_data, 1)
            if exp_data[i, 3] > exp_data[i-1, 3]
                exp_data[i, 3] = exp_data[i-1, 3]
            end
        end
    end
    
    return exp_data
end

"""
    load_experimental_dataset(data_dir::String, exp_indices::Vector{Int})
    
Load all experimental data files and their conditions.
Returns (exp_data_list, exp_info_matrix).
"""
function load_experimental_dataset(data_dir::String, exp_indices::Vector{Int}; 
                                   info_files::Dict{String,String}=Dict())
    l_exp_data = []
    n_exp = length(exp_indices)
    l_exp_info = zeros(Float64, n_exp, length(info_files) + 1)
    
    for (i_exp, value) in enumerate(exp_indices)
        filename = joinpath(data_dir, "expdata_no$(value).txt")
        exp_data = Float64.(load_exp(filename))
        push!(l_exp_data, exp_data)
        
        # Store initial temperature
        l_exp_info[i_exp, 1] = exp_data[1, 2]
    end
    
    # Load additional info files (beta, catalyst concentration, etc.)
    col = 2
    for (key, filename) in sort(info_files)
        data = readdlm(filename)[exp_indices]
        l_exp_info[:, col] = vec(data)
        col += 1
    end
    
    return l_exp_data, l_exp_info
end

# ============================================================================
# TEMPERATURE CALCULATION
# ============================================================================

"""
    getsampletemp(t::Float64, T0::Float64, beta::Float64, t0::Float64=0.0; mode::Symbol=:linear)
    
Convert time to temperature for linear or programmed heating.

# Arguments
- `t`: current time (seconds)
- `T0`: initial temperature (K)
- `beta`: heating rate (K/min)
- `t0`: reference time (default: 0.0)
- `mode`: `:linear` for linear heating, `:programmed` for multi-step heating
"""
function getsampletemp(t::Float64, T0::Float64, beta::Float64, t0::Float64=0.0; mode::Symbol=:linear)
    if mode == :linear
        T = T0 + beta / 60 * (t - t0)
        return T
    
    elseif mode == :programmed
        tc = [999.0, 1059.0] .* 60.0
        Tc = [beta, 370.0, 500.0] .+ 273.0
        HR = 40.0 / 60.0;
        
        if t <= tc[1]
            T = Tc[1]
        elseif t <= tc[2]
            T = minimum([Tc[1] + HR * (t - tc[1]), Tc[2]])
        else
            T = minimum([Tc[2] + HR * (t - tc[2]), Tc[3]])
        end
        return T
    else
        error("Unknown heating mode: $mode")
    end
end

# ============================================================================
# PARAMETER TRANSFORMATION
# ============================================================================

"""
    abstract type ParameterTransformer
    
Abstract base for parameter transformation strategies.
"""
abstract type ParameterTransformer end

"""
    struct SimplePTransformer <: ParameterTransformer
    
Parameter transformer for models without catalysts (5 species, 5 reactions).
"""
struct SimplePTransformer <: ParameterTransformer
    ns::Int64
    nr::Int64
    p_cutoff::Float64
end

"""
    struct CatalystPTransformer <: ParameterTransformer
    
Parameter transformer for catalyst models (6 species, 5 reactions, 1 catalyst).
"""
struct CatalystPTransformer <: ParameterTransformer
    ns::Int64
    nr::Int64
    nc::Int64
    nss::Int64  # non-solvent species (ns - nc)
    p_cutoff::Float64
end

"""
    p2vec(transformer::ParameterTransformer, p::Vector{Float64})
    
Convert flat parameter vector to (w_in, w_b, w_out) matrices.
Implementation varies by transformer type.
"""
function p2vec(t::SimplePTransformer, p::Vector{Float64})
    @assert length(p) == t.nr * (t.ns + 3) + 1 "Parameter vector length mismatch"
    
    slope = p[end] .* 1.e1
    w_b = p[1:t.nr] .* (slope * 10.0)
    w_b = clamp.(w_b, -23, 50)
    
    w_out = reshape(p[t.nr+1:t.nr*(t.ns-1)+1], t.ns-1, t.nr)
    @. w_out[1, :] = clamp(w_out[1, :], -3.0, 0.0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0.0, 3.0)
    
    if t.p_cutoff > 0.0
        w_out[findall(abs.(w_out) .< t.p_cutoff)] .= 0.0
    end
    
    w_out[t.ns-2:t.ns-2, :] .=
        -sum(w_out[1:t.ns-3, :], dims=1) .- sum(w_out[t.ns-1:t.ns-1, :], dims=1)
    
    w_in_Ea = abs.(p[t.nr*(t.ns-1)+2:t.nr*(t.ns-1)+t.nr+1] .* (slope * 100.0))
    w_in_Ea = clamp.(w_in_Ea, 100.0, 300.0)
    
    w_in_b = abs.(p[t.nr*(t.ns-1)+t.nr+2:end-1])
    
    w_in = vcat(clamp.(-w_out, 0.0, 4.0), w_in_Ea', w_in_b')
    
    return w_in, w_b, w_out
end

function p2vec(t::CatalystPTransformer, p::Vector{Float64})
    @assert length(p) == t.nr * (t.ns + t.nc + 3) + 1 "Parameter vector length mismatch"
    
    slope = p[end] .* 1.e1
    w_b = p[1:t.nr] .* (slope * 10.0)
    w_b = clamp.(w_b, -23, 50)
    
    w_out = reshape(p[t.nr+1:t.nr*(t.nss+1)], t.nss, t.nr)
    @. w_out[1, :] = clamp(w_out[1, :], -3.0, 0.0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0.0, 3.0)
    
    if t.p_cutoff > 0.0
        w_out[findall(abs.(w_out) .< t.p_cutoff)] .= 0.0
    end
    
    w_out[t.nss-1:t.nss-1, :] .=
        -sum(w_out[1:t.nss-2, :], dims=1) .- sum(w_out[t.nss:t.nss, :], dims=1)
    
    w_cat_in = p[t.nr*(t.nss+1)+1:t.nr*(t.ns+1)]
    w_cat_out = p[t.nr*(t.ns+1)+1:t.nr*(t.ns+t.nc+1)] * 0
    w_cat_in = abs.(w_cat_in)
    
    if t.p_cutoff > 0.0
        w_cat_in[findall(abs.(w_cat_in) .< t.p_cutoff)] .= 0.0
    end
    
    w_in_Ea = abs.(p[t.nr*(t.ns+t.nc+1)+1:t.nr*(t.ns+t.nc+2)] .* (slope * 100.0))
    w_in_Ea = clamp.(w_in_Ea, 80.0, 200.0)
    
    w_in_b = abs.(p[t.nr*(t.ns+t.nc+2)+1:t.nr*(t.ns+t.nc+3)])
    
    w_in = vcat(clamp.(-w_out, 0.0, 3.0), w_cat_in', w_in_Ea', w_in_b')
    w_out = vcat(w_out, w_cat_out')
    
    return w_in, w_b, w_out
end

# ============================================================================
# ODE SYSTEM
# ============================================================================

"""
    ODE system for CRNN model.
    
Parameters stored as globals: T0, beta, w_in, w_b, w_out, t0_exp, lb
"""
const R = -1.0 / 8.314e-3  # Universal gas constant, kJ/(mol·K)

function create_crnn!(T0, beta, t0_exp, w_in, w_b, w_out, lb; heating_mode::Symbol=:linear)
    """Create a CRNN ODE function with captured parameters."""
    function crnn!(du, u, p, t)
        logX = @. log(clamp(u, lb, 10.0))
        T = getsampletemp(t, T0, beta, t0_exp; mode=heating_mode)
        w_in_x = w_in' * vcat(logX, R / T, log(T))
        du .= w_out * (@. exp(w_in_x + w_b))
    end
    return crnn!
end

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

"""
    loss_neuralode(p, i_exp, transformer, l_exp_data, l_exp_info, pred_n_ode_fn, nc)
    
Calculate loss for a single experiment.
"""
function loss_neuralode(p, i_exp, transformer, l_exp_data, l_exp_info, 
                       pred_n_ode_fn, nc::Int64=0)
    exp_data = l_exp_data[i_exp]
    pred = Array(pred_n_ode_fn(p, i_exp, exp_data))
    
    if nc == 0
        masslist = sum(clamp.(@view(pred[1:end-1, :]), 0, Inf), dims=1)'
        gaslist = clamp.(@view(pred[end, :]), 0, Inf)
    else
        masslist = sum(clamp.(@view(pred[1:end-1-nc, :]), 0, Inf), dims=1)'
        gaslist = clamp.(@view(pred[end-nc, :]), 0, Inf)
    end
    
    loss = mae(masslist, @view(exp_data[1:length(masslist), 3])) + 
           mae(gaslist, 1 .- @view(exp_data[1:length(masslist), 3]))
    return loss
end

# ============================================================================
# VISUALIZATION
# ============================================================================

"""
    display_p(transformer, p; digits=2)
    
Display parameter matrix in readable format.
"""
function display_p(transformer::ParameterTransformer, p::Vector{Float64}; digits::Int=2)
    w_in, w_b, w_out = p2vec(transformer, p)
    println("\n species (column) reaction (row)")
    
    if transformer isa SimplePTransformer
        println("w_in | Ea | b | lnA | w_out")
    else
        println("w_in | w_cat_in | Ea | b | lnA | w_out | w_cat_out")
    end
    
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits=digits))
    println("\n")
end

"""
    plot_sol(i_exp, sol, exp_data, l_exp_info, heating_mode::Symbol=:linear; sol0=nothing)
    
Generate comprehensive plots of model solution vs experimental data.
"""
function plot_sol(i_exp, sol, exp_data, l_exp_info, heating_mode::Symbol=:linear; sol0=nothing)
    T0, beta = l_exp_info[i_exp, 1:2]
    
    t0_raw = sol.t[1]
    ts = (sol.t .- t0_raw) ./ 60.0;
    
    Tlist = [getsampletemp(t, T0, beta, t0_raw; mode=heating_mode) for t in sol.t]
    
    plt = plot(
        exp_data[:, 2], exp_data[:, 3],
        seriestype=:scatter, label="Exp", legend=:left
    )
    
    plot!(plt, Tlist, sum(clamp.(sol[1:end-1, :], 0, Inf), dims=1)',
        lw=4, label="CRNN")
    
    xlabel!(plt, "Temperature [K]")
    ylabel!(plt, "Normalized Mass")
    title!(plt, "Exp $(i_exp)")
    
    exp_cond = @sprintf("T₀ = %.1f K\nβ = %.2f K/min", T0, beta)
    annotate!(plt, Tlist[end] * 0.85, 0.4, exp_cond)
    
    p2 = plot(Tlist, sol[1, :], lw=4, legend=:right, label="Xylan")
    for i in 2:size(sol, 1)-1
        plot!(p2, Tlist, sol[i, :], lw=4, label="S$i")
    end
    plot!(p2, Tlist, sol[end, :], lw=4, label="Volatile")
    xlabel!(p2, "Temperature [K]")
    ylabel!(p2, "Mass")
    
    p3 = plot(ts, sol[1, :], lw=4, legend=:right, label="Xylan")
    for i in 2:size(sol, 1)-1
        plot!(p3, ts, sol[i, :], lw=4, label="S$i")
    end
    plot!(p3, ts, sol[end, :], lw=4, label="Volatile")
    xlabel!(p3, "Time [min]")
    ylabel!(p3, "Mass")
    
    combined = plot(plt, p2, p3, framestyle=:box, layout=@layout [a; b; c])
    plot!(combined, size=(800, 800))
    
    return combined
end

"""
    plot_loss_history(l_loss_train, l_loss_val, list_grad; fig_path="./")
    
Plot training and validation loss and gradient norm.
"""
function plot_loss_history(l_loss_train, l_loss_val, list_grad; fig_path="./")
    plt_loss = plot(l_loss_train, yscale=:log10, label="train")
    plot!(plt_loss, l_loss_val, yscale=:log10, label="val")
    xlabel!(plt_loss, "Epoch")
    ylabel!(plt_loss, "Loss")
    
    plt_grad = plot(list_grad, yscale=:log10, label="grad_norm")
    xlabel!(plt_grad, "Epoch")
    ylabel!(plt_grad, "Gradient Norm")
    
    plt_all = plot([plt_loss, plt_grad]..., legend=:top, framestyle=:box)
    plot!(plt_all, size=(1000, 450),
        xtickfontsize=11, ytickfontsize=11,
        xguidefontsize=12, yguidefontsize=12)
    
    png(plt_all, joinpath(fig_path, "loss_grad"))
    return plt_all
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    setup_directories(config::CRNNConfig)
    
Create necessary directories for results and checkpoints.
"""
function setup_directories(config::CRNNConfig)
    dirs = [config.fig_path, 
            joinpath(config.fig_path, "conditions"),
            config.ckpt_path]
    
    for dir in dirs
        if !ispath(dirname(dir))
            mkpath(dirname(dir))
        end
        if !ispath(dir)
            mkdir(dir)
        end
    end
end

"""
    cleanup_directories(config::CRNNConfig)
    
Remove existing result directories (when not restarting).
"""
function cleanup_directories(config::CRNNConfig)
    for path in [config.fig_path, config.ckpt_path]
        if ispath(path)
            rm(path, recursive=true)
        end
    end
end

"""
    initialize_optimizer(config::CRNNConfig, n_train::Int)
    
Create optimizer with learning rate schedule.
"""
function initialize_optimizer(config::CRNNConfig, n_train::Int)
    return Flux.Optimiser(
        Flux.Optimise.ExpDecay(
            config.lr_max, 
            config.lr_decay, 
            n_train * config.lr_decay_step, 
            config.lr_min
        ),
        Flux.Optimise.AdamW(config.lr_max, (0.9, 0.999), config.w_decay)
    )
end

"""
    split_train_val(n_exp::Int, val_indices::Vector{Int})
    
Split experiment indices into training and validation sets.
"""
function split_train_val(n_exp::Int, val_indices::Vector{Int})
    all_indices = 1:n_exp
    train_indices = [i for i in all_indices if !(i in val_indices)]
    return train_indices, val_indices
end

"""
    save_checkpoint(p, opt, l_loss_train, l_loss_val, list_grad, iter, ckpt_path)
    
Save model checkpoint.
"""
function save_checkpoint(p, opt, l_loss_train, l_loss_val, list_grad, iter, ckpt_path)
    @save joinpath(ckpt_path, "mymodel.bson") p opt l_loss_train l_loss_val list_grad iter
end

"""
    load_checkpoint(ckpt_path)
    
Load model checkpoint.
"""
function load_checkpoint(ckpt_path)
    @load joinpath(ckpt_path, "mymodel.bson") p opt l_loss_train l_loss_val list_grad iter
    return p, opt, l_loss_train, l_loss_val, list_grad, iter
end

end  # module CRNNCommon