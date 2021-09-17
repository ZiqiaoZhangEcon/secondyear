# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 1: Solve numerically the McGee-Livshits-Tertilt economy described
# on the slides in partial equilibrium.
#

cd(dirname(@__FILE__))

using UnPack, LinearAlgebra
include("tauchen.jl")

u(c) = log(c)
u(c::Vector) = log.(c)


mutable struct Params
    β::Real
    r::Real
    q̅::Real
    γ::Real
    a::AbstractVector{Real}
    y::AbstractVector{Real}
    N::Integer
    MC::MarkovChain
 
    function Params(MC::MarkovChain; β = 0.9, r = 0.02, q̅ = 1 / 1.02, γ = 0, N = 100)
	ygrid = exp.(MC.state_values)
	amin = -ygrid[end] #amin = -ygrid[1]/r
	amax = ygrid[end]  #amax = ygrid[end]/r
	a = range(amin, stop = amax, length = N)

	return new(β, r, q̅, γ, a, ygrid, N, MC)
    end

end


mutable struct ValueAndPolicy
	V::Array{Float64,2} # Value function
	value_market::Array{Float64,2} # Value function when in the mkt
	value_default::Array{Float64,2} # Value function when in autarky (default)
	q::Array{Float64,2} # Prices (inverse return to savings)
	savings_pol::Array{Int64,2} # savings policy function
	c_pol::Array{Float64,2} # consumption policy function
	D_pol::Array{Int8, 2} # default decision

	function ValueAndPolicy(P::Params)
		@unpack β, r, q̅, γ, a, y, N, MC = P

		V = Array{Float64,2}(undef, (N, states)) # Vij = V(a[i], y[j])
		value_market =	 Array{Float64,2}(undef, (N, states))
		value_default = Array{Float64,2}(undef, (1, states))
		for i in 1:N
			for j in 1:states
				c = max(y[j]+r*a[i],1e-2)
				# c = max(y[j] + a[i], 1 )
				value_market[i, j] = (1 / (1 - β) * u(c)) # Initial guess
				V[i, j] = (1 / (1 - β) * u(c)) # Initial guess
				c̃ = y[j]
				value_default[1, j] = (1 / (1 - β)) * u(c̃) 
			end
		end

		q = ones(Float64, N, states)
		c_pol = Array{Float64,2}(undef, (N, states)) #cij = c(a[i], y[j])
		savings_pol = round.(Int, zeros((N, states))) # savings policy fuction (in indices)
		D_pol = zeros(Int8, N, states)
		return new(V, value_market, value_default, q, savings_pol, c_pol, D_pol)
	end
end

function OneStepUpdate!(VP, EV, Evalue_market, Evalue_default)
	@unpack β, r, q̅, γ, a, y, N, MC = P
	objective = Array{Float64}(undef, N)
	for j in 1:states
		valuedefault =  u(y[j]) + β * Evalue_default[1, j] 
		VP.value_default[1, j] = valuedefault

		for i in 1:N
			for k in 1:N
				# From Ziqiao: I think the problem is here! 'saving_pol' is an index, rather than a saving amount!!
				consumption = max(y[j] + VP.savings_pol[i, j] - VP.q[k, j] * VP.savings_pol[k], 1e-5) # for each choice of savings, consumption is residual; can't be negative
				objective[k] = u(consumption) + β * Evalue_market[k, j]
			end

			VP.savings_pol[i,j] = round.(Int,argmax(objective)) 
			VP.value_market[i,j] = maximum(objective)
			if VP.value_market[i, j] < VP.value_default[1, j]
				VP.V[i, j] = VP.value_default[1, j]
				VP.D_pol[i, j] = 1
			else
				VP.V[i, j] = VP.value_market[i, j]
				VP.D_pol[i, j] = 0
			end
		end
	end
end

function FindPrice!(P::Params, VP::ValueAndPolicy)
	@unpack β, r, q̅, γ, a, y, N, MC = P
	#vd_compat = repeat(VP.value_default, N)
    	#default_states = vd_compat .> VP.value_market
	default_states = VP.D_pol

   	# update price
	Π = transpose(MC.transition)
    	θ = default_states * Π # prob of default
    	copyto!(VP.q, (1 .- θ) * q̅ )

end

function ValueFunctionIteration!(P::Params, VP::ValueAndPolicy; max_iter = 1000, tolerance = 1e-5)
	@unpack β, r, q̅, γ, a, y, N, MC = P

	Π = transpose(MC.transition)
	ε = 10.0
	iterations = 0

	while ε > tolerance && iterations < max_iter
		EV = VP.V * Π
		Evalue_market = VP.value_market * Π
		Evalue_default = VP.value_default * Π
		old_V = copy(VP.V)
		OneStepUpdate!(VP, EV, Evalue_market, Evalue_default)	
		FindPrice!(P, VP)
		ε = maximum(abs.(VP.V - old_V))
		iterations += 1
		if iterations % 10 == 0
			veps = round(ε, digits = 6)
			println("iteration: $iterations, ε: $veps")
		end
	end
	println("Converged")
end


# PARAMS
# AR1 process　
ρ = 0.9
σ = 0.1
states = 3

MC = Tauchen(ρ, σ, states; m = 2)

P = Params(MC)
VP = ValueAndPolicy(P)
ValueFunctionIteration!(P, VP)
