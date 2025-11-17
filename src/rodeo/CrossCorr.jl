"""
    CrossCorr

Computes cross-correlation functions and time differences between signals.
This module is part of the `rodeo` Python package.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""

module CrossCorr

using DSP
using LinearAlgebra
using ProgressBars
using Statistics

export corr_matrix, dt_matrix

"""
    cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})::Float64

Compute the maximum value of the cross-correlation function between two signals.
"""
function cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})::Float64
    x1 = (sig1 .- mean(sig1)) ./ std(sig1)
    x2 = (sig2 .- mean(sig2)) ./ std(sig2)
    norm_factor = sqrt(sum(x1 .^ 2) * sum(x2 .^ 2))
    return maximum(xcorr(x1, x2) ./ norm_factor)
end


"""
    corr_matrix(data::Matrix{Float64})::Matrix{Float64}

Compute the matrix containing maximum values of the cross-correlation functions
between pairs of signals.
"""
function corr_matrix(data::Matrix{Float64})::Matrix{Float64}
    rows = size(data, 1)
    corr = Matrix{Float64}(undef, rows, rows)

    # Pre-normalize all signals once to avoid redundant computations
    normalized_data = similar(data)
    norms = Vector{Float64}(undef, rows)

    for i in 1:rows
        sig = @view data[i, :]
        normalized_data[i, :] = (sig .- mean(sig)) ./ std(sig)
        norms[i] = sqrt(sum(normalized_data[i, :] .^ 2))
    end

    pbar = ProgressBar(1:rows)
    set_description(pbar, "Computing cross-correlations between signals")

    Threads.@threads for i in pbar
        sig1 = @view normalized_data[i, :]
        @inbounds for j in i:rows
            sig2 = @view normalized_data[j, :]
            ccf = maximum(xcorr(sig1, sig2)) / (norms[i] * norms[j])
            corr[i, j] = ccf
            corr[j, i] = ccf
        end
    end

    return corr
end


"""
    dt_matrix(times::Vector{Float64})::Matrix{Float64}

Compute the matrix containing time differences between pairs of time points.
"""
function dt_matrix(times::Vector{Float64})::Matrix{Float64}
    rows = length(times)
    dt = Matrix{Float64}(undef, rows, rows)

    pbar = ProgressBar(1:rows)
    set_description(pbar, "Computing time differences between signals")

    Threads.@threads for i in pbar
        @inbounds for j in i:rows
            dt[i, j] = times[j] - times[i]
            dt[j, i] = -dt[i, j]
        end
        dt[i, i] = 0.0  # Ensure diagonal is explicitly zero
    end
    return dt
end


"""
    find_efolding_time(ccf::Vector{Float64}, fs::Float64)::Float64

Find the e-folding time of a cross-correlation function.
"""
function find_efolding_time(ccf::Vector{Float64}, fs::Float64)::Float64
    # Find peak index and value
    peak_idx = argmax(ccf)
    peak_value = ccf[peak_idx]
    efold_value = peak_value / exp(1)

    # Find first index where signal falls below e-folding value after peak
    below_idx = findfirst(x -> x <= efold_value, ccf[peak_idx:end])

    if isnothing(below_idx)
        return nothing
    end

    # Adjust index to absolute position
    idx = peak_idx - 1 + below_idx

    if idx > 1
        # Linear interpolation
        x1, x2 = (idx - 1) / fs, idx / fs
        y1, y2 = ccf[idx-1], ccf[idx]
        slope = (y2 - y1) / (x2 - x1)
        return x1 + (efold_value - y1) / slope
    else
        return idx / fs
    end
end


function num_computations(n::Int)::Int
    return n * (n - 1) ÷ 2
end

end