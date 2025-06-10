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
# using ProgressMeter
using Statistics


"""
    cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64})::Float64

Compute the maximum value of the cross-correlation function between two signals.
"""
function cross_corr(sig1::Vector{Float64}, sig2::Vector{Float64}, fs::Float64)::Tuple{Float64,Float64}
    x1 = @. (sig1 - mean(sig1)) / std(sig1)
    x2 = @. (sig2 - mean(sig2)) / std(sig2)
    norm_factor = sqrt(sum(x1 .^ 2) * sum(x2 .^ 2))
    ccf = xcorr(x1, x2) ./ norm_factor
    et = find_efolding_time(ccf, fs)
    return maximum(ccf), et
end


"""
    corr_matrix(data::Matrix{Float64})::Matrix{Float64}

Compute the matrix containing maximum values of the cross-correlation functions
between pairs of signals.
"""
function corr_matrix(data::Matrix{Float64}, fs::Float64)::Tuple{Matrix{Float64},Matrix{Float64}}
    rows = size(data, 1)
    corr = Matrix{Float64}(undef, rows, rows)
    etmat = Matrix{Float64}(undef, rows, rows)

    results = Vector{Tuple{Int,Int,Float64,Float64}}()
    results_lock = ReentrantLock()

    # num_calcs = num_computations(rows)
    # pb = Progress(num_calcs)
    # update!(pb, 0)
    # pb = ProgressBar(1:rows)

    Threads.@threads for i in 1:rows
        local_results = Vector{Tuple{Int,Int,Float64,Float64}}()
        sig1 = data[i, :]
        for j in i:rows
            sig2 = data[j, :]
            ccf, et = cross_corr(sig1, sig2, fs)
            push!(local_results, (i, j, ccf, et))
        end
        lock(results_lock) do
            append!(results, local_results)
        end
    end

    for (i, j, ccf, et) in results
        corr[i, j] = ccf
        corr[j, i] = ccf
        etmat[i, j] = et
        etmat[j, i] = et
    end

    return corr, etmat
end


"""
    dt_matrix(times::Vector{Float64})::Matrix{Float64}

Compute the matrix containing time differences between pairs of time points.
"""
function dt_matrix(times::Vector{Float64})::Matrix{Float64}
    rows = length(times)
    dt = Matrix{Float64}(undef, rows, rows)

    Threads.@threads for i in ProgressBar(1:rows)
        for j in i:rows
            dt[i, j] = times[j] - times[i]
            dt[j, i] = -dt[i, j]
        end
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


export corr_matrix, dt_matrix

end