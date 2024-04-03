using ITensors
using LsqFit
using Plots

# free-free
function H_ising(; N, sites = siteinds(N, "S=1/2"), J = 1, Γ = 1)
  os = OpSum()
  for j in 1:N-1
    os += -4J, "Sz", j, "Sz", j + 1
    os += -2Γ, "Sx", j
  end
  os += -2Γ, "Sx", N
  sites, MPO(os, sites)
end

function U_ising(; sites, J = 1, Γ = 1, tau)
  N = length(sites)
  gates = ITensor[]

  hj = -4J * op("Sz", sites[1]) * op("Sz", sites[2]) - 2Γ * op("Sx", sites[1]) * op("I", sites[2]) - Γ * op("I", sites[1]) * op("Sx", sites[2])
  push!(gates, exp(tau / 2 * hj))
  for j in 2:N-2
    hj = -4J * op("Sz", sites[j]) * op("Sz", sites[j + 1]) - Γ * op("Sx", sites[j]) * op("I", sites[j + 1]) - Γ * op("I", sites[j]) * op("Sx", sites[j + 1])
    push!(gates, exp(tau / 2 * hj))
  end
  hj = -4J * op("Sz", sites[end - 1]) * op("Sz", sites[end]) - Γ * op("Sx", sites[end - 1]) * op("I", sites[end]) - 2Γ * op("I", sites[end - 1]) * op("Sx", sites[end])
  push!(gates, exp(tau / 2 * hj))
  append!(gates, reverse(gates))
  gates
end

function autocor(; state, tau, T, cutoff = 1e-8)
  sites = siteinds(state)
  op = OpSum()
  op += "Sz", 1
  Sz1 = MPO(op, sites)
  a = apply(Sz1, state)

  gates = U_ising(;sites, tau)

  c = Float64[]
  for _ in tau:tau:T
    state = apply(gates, state; cutoff)
    b = apply(Sz1, state)
    push!(c, inner(a, b))
  end
  c
end

function dmrg_tfising(; N, sites = siteinds("S=1/2", N), maxdim, cutoff)
  _, H = H_ising(; N, sites)
  psi0 = randomMPS(sites, linkdims=40)
  dmrg(H, psi0; nsweeps = length(maxdim), maxdim, cutoff)
end

function impl(; N, tau, T, cutoff)
  if !ispath("data")
    mkdir("data")
  end
  _, state = dmrg_tfising(;N, maxdim = [20, 80, 80, 120, 120, 120, 120], cutoff)
  c = autocor(; state, tau, T, cutoff)
  open("data/N$N-T$T-tau$tau.txt", "w") do fp
    println(fp, "# cutoff : $cutoff")
    Base.print_array(fp, c)
  end
  c
end


function read(filename)
  data = []
  open(filename, "r") do fp
    for line in eachline(fp)
      if line[1] == '#'
        continue
      end
      push!(data, parse.(Float64, split(line)))
    end
  end
  data
end

function plotdata(; N, tau, T, fitrange)
  data = read("data/N$N-T$T-tau$tau.txt")
  v = 1
  c = vcat(data...)
  t = tau:tau:T
  scatter(inv.(t), c, xlabel = "1 / τ", ylabel = "<σ^z_1(τ)σ^z_1(0)>")

  f(x, p) = @. p[1] * x + p[2]
  fit = curve_fit(f, inv.(t)[fitrange], c[fitrange], [0., 0.])
  xs = [minimum(inv.(t)):1e-2:maximum(inv.(t));]
  plot!(xs, f(xs, fit.param), title = "fitrange$fitrange", label = "y = $(fit.param[1])x + $(fit.param[2])")
  savefig("plot/N$N-T$T-tau$tau.png")

  fit.param
end

begin
  if length(ARGS) < 3
    println("usage:")
    println("julia ising-autocor.jl N tau T [cutoff]")
    return
  end
  cutoff = 1e-10
  if length(ARGS) >= 4
    cutoff = parse(Float64, ARGS[4])
  end
  impl(;N = ARGS[1], tau = ARGS[2], T = ARGS[3], cutoff)
end
