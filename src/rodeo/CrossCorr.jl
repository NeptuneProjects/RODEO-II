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
using ProgressMeter
using Statistics

export corr_matrix, dt_matrix, lag_matrix

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
    corr_matrix(data::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Int}}

Compute the matrix containing maximum values of the cross-correlation functions
between pairs of signals, and the sample lags at which they occur.

Returns:
- corr: Matrix of maximum cross-correlation values
- shifts: Matrix of sample lags at which maximum correlation occurs
  (positive lag means signal j leads signal i)
"""
function corr_matrix(data::Matrix{Float64})::Tuple{Matrix{Float64},Matrix{Int}}
    rows = size(data, 1)
    n_samples = size(data, 2)
    corr = Matrix{Float64}(undef, rows, rows)
    shifts = Matrix{Int}(undef, rows, rows)

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
    max_tid = Threads.maxthreadid()

    # Create buffers for each thread (use maxthreadid to handle thread pool size)
    fft_buf1 = [zeros(ComplexF64, n_fft) for _ in 1:max_tid]
    fft_buf2 = [zeros(ComplexF64, n_fft) for _ in 1:max_tid]
    result_buf = [zeros(Float64, n_fft) for _ in 1:max_tid]

    # Create separate FFT plans per thread for thread safety
    # Use FFTW.ESTIMATE to avoid expensive planning that can accumulate state
    plans_fwd = [plan_fft!(fft_buf1[t], flags=FFTW.ESTIMATE) for t in 1:max_tid]
    plans_inv = [plan_ifft!(fft_buf1[t], flags=FFTW.ESTIMATE) for t in 1:max_tid]

    # Work-stealing with atomic counter for perfect load balancing
    # Each thread grabs one row at a time (finest granularity)
    # Start from last row (least work) to first row (most work)
    # This ensures all threads stay busy until completion with no idle time
    next_row = Threads.Atomic{Int}(rows)
    progress = Progress(rows, dt=0.5, desc="Computing cross-correlations: ")

    Threads.@threads for _ in 1:Threads.nthreads()
        tid = Threads.threadid()
        buf1 = fft_buf1[tid]
        buf2 = fft_buf2[tid]
        result = result_buf[tid]
        plan_fwd = plans_fwd[tid]
        plan_inv = plans_inv[tid]

        while true
            # Atomically grab next row (work-stealing with chunk size = 1)
            i = Threads.atomic_add!(next_row, -1)
            if i < 1
                break  # No more work
            end

            sig1 = @view normalized_data[i, :]

            @inbounds for j in i:rows
                sig2 = @view normalized_data[j, :]

                # Zero out buffers and copy signals
                # Use explicit loops to avoid slice allocation
                @inbounds for k in 1:n_samples
                    buf1[k] = sig1[k]
                    buf2[k] = sig2[k]
                end
                @inbounds for k in (n_samples+1):n_fft
                    buf1[k] = 0.0
                    buf2[k] = 0.0
                end

                # Compute FFTs in-place
                mul!(buf1, plan_fwd, buf1)
                mul!(buf2, plan_fwd, buf2)

                # Multiply: FFT(sig1) .* conj(FFT(sig2))
                # Use in-place conjugation to avoid allocation
                @inbounds for k in eachindex(buf1, buf2)
                    buf1[k] = buf1[k] * conj(buf2[k])
                end

                # Inverse FFT to get cross-correlation
                mul!(buf1, plan_inv, buf1)

                # Extract real part and find maximum (avoid allocation)
                @inbounds for k in eachindex(result, buf1)
                    result[k] = real(buf1[k])
                end
                max_idx = argmax(result)
                ccf = result[max_idx] / (norms[i] * norms[j])

                # Convert FFT index to actual lag
                # Index 1 corresponds to lag -(n_samples-1)
                # Index n_samples corresponds to lag 0
                lag = max_idx - n_samples

                corr[i, j] = ccf
                corr[j, i] = ccf
                shifts[i, j] = lag
                shifts[j, i] = -lag
            end
            corr[i, i] = 1.0  # Self-correlation is always 1
            shifts[i, i] = 0  # Zero lag for self-correlation

            # Update progress (thread-safe)
            next!(progress)
        end
    end

    return corr, shifts
end


"""
    dt_matrix(times::Vector{Float64})::Matrix{Float64}

Compute the matrix containing time differences between pairs of time points.
"""
function dt_matrix(times::Vector{Float64})::Matrix{Float64}
    rows = length(times)
    dt = Matrix{Float64}(undef, rows, rows)

    # Work-stealing with atomic counter for perfect load balancing
    next_row = Threads.Atomic{Int}(rows)
    progress = Progress(rows, dt=0.5, desc="Computing time differences: ")

    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            i = Threads.atomic_add!(next_row, -1)
            if i < 1
                break
            end

            @inbounds for j in i:rows
                dt[i, j] = times[j] - times[i]
                dt[j, i] = -dt[i, j]
            end
            dt[i, i] = 0.0  # Ensure diagonal is explicitly zero

            next!(progress)
        end
    end
    return dt
end


"""
    lag_matrix(n::Int)::Matrix{Int}

Compute the matrix containing sample lags between pairs of indices.
For indices i and j, lag_matrix[i,j] = j - i.
"""
function lag_matrix(n::Int)::Matrix{Int}
    lags = Matrix{Int}(undef, n, n)

    # Work-stealing with atomic counter for perfect load balancing
    next_row = Threads.Atomic{Int}(n)
    progress = Progress(n, dt=0.5, desc="Computing sample lags: ")

    Threads.@threads for _ in 1:Threads.nthreads()
        while true
            i = Threads.atomic_add!(next_row, -1)
            if i < 1
                break
            end

            @inbounds for j in i:n
                lags[i, j] = j - i
                lags[j, i] = i - j
            end
            lags[i, i] = 0  # Ensure diagonal is explicitly zero

            next!(progress)
        end
    end
    return lags
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