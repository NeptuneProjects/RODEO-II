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
using FFTW
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
    n_samples = size(data, 2)
    corr = Matrix{Float64}(undef, rows, rows)

    # Pre-normalize all signals once to avoid redundant computations
    normalized_data = similar(data)
    norms = Vector{Float64}(undef, rows)

    for i in 1:rows
        sig = @view data[i, :]
        normalized_data[i, :] = (sig .- mean(sig)) ./ std(sig)
        norms[i] = sqrt(sum(normalized_data[i, :] .^ 2))
    end

    # Pre-allocate FFT buffers per thread to avoid repeated allocations
    n_fft = nextpow(2, 2 * n_samples - 1)  # FFT size for cross-correlation
    n_threads = Threads.nthreads()

    # Create buffers for each thread
    fft_buf1 = [zeros(ComplexF64, n_fft) for _ in 1:n_threads]
    fft_buf2 = [zeros(ComplexF64, n_fft) for _ in 1:n_threads]
    result_buf = [zeros(Float64, n_fft) for _ in 1:n_threads]

    # Create FFT plans (reusable across all computations)
    plan_fwd = plan_fft!(fft_buf1[1])
    plan_inv = plan_ifft!(fft_buf1[1])

    pbar = ProgressBar(1:rows)
    set_description(pbar, "Computing cross-correlations between signals")

    Threads.@threads for i in pbar
        tid = Threads.threadid()
        buf1 = fft_buf1[tid]
        buf2 = fft_buf2[tid]
        result = result_buf[tid]

        sig1 = @view normalized_data[i, :]

        @inbounds for j in i:rows
            sig2 = @view normalized_data[j, :]

            # Zero out buffers
            fill!(buf1, 0.0)
            fill!(buf2, 0.0)

            # Copy signals into buffers
            buf1[1:n_samples] .= sig1
            buf2[1:n_samples] .= sig2

            # Compute FFTs in-place
            mul!(buf1, plan_fwd, buf1)
            mul!(buf2, plan_fwd, buf2)

            # Multiply: FFT(sig1) .* conj(FFT(sig2))
            buf1 .= buf1 .* conj.(buf2)

            # Inverse FFT to get cross-correlation
            mul!(buf1, plan_inv, buf1)

            # Extract real part and find maximum
            result .= real.(buf1)
            ccf = maximum(result) / (norms[i] * norms[j])

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