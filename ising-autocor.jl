using ITensors
using ITensorTDVP
using LsqFit
using Plots
using Printf
using LaTeXStrings

default(
  fontfamily="serif-roman",
  titlefontsize=20,
  guidefontsize=20,
  tickfontsize=15,
  legendfontsize=15,
  markerstrokewidth=2,
  margin=2Plots.mm,
  size=(700, 600),
  grid=false,
  foreground_color_legend=nothing,
  background_color_legend=colorant"rgba(255,255,255,0.6)",
  linewidth=2
)

# free-free
function H_ising(; N, sites=siteinds("S=1/2", N), J=0.25, Γ=0.5)
  os = OpSum()
  for j in 1:N-1
    os += -4J, "Sz", j, "Sz", j + 1
    os += -2Γ, "Sx", j
  end
  os += -2Γ, "Sx", N
  MPO(os, sites), sites
end

function mag(sites)
  os = OpSum()
  for j in eachindex(sites)
    os += inv(length(sites)), "Sz", j
  end
  MPO(os, sites)
end

function mag2(sites)
  m = mag(sites)
  apply(m, m)
end

function binder(state)
  sites = siteinds(state)
  m2 = mag2(sites)
  m4 = apply(m2, m2)
  abs(dot(state', m4, state)) / abs(dot(state', m2, state))^2
end

function binder4N(; H, Ns=[10, 20, 40, 80], Γs=[0.95:1e-2:1.05;])
  χ = [[40, 60, 60, 60, 80, 80]; [120 for _ in 1:15]]

  plt = plot()
  for N in Ns
    b = []
    for Γ in Γs
      H_mpo, sites = H(; N, J=1, Γ)
      _, psi0 = dmrg(H_mpo, randomMPS(sites; linkdims=40); nsweeps=length(χ), maxdim=χ, cutoff=1e-12, noise=1e-10)
      push!(b, binder(psi0))
    end
    scatter!(plt, Γs, b, label="N=$N", xlabel="Γ", ylabel=L"<m^4>/<m^2>")
  end

  savefig(plt, "binder.png")
end

function linfit(xs, ys)
  f(x, p) = @. p[1] * x + p[2]
  fit = curve_fit(f, xs, ys, [0.0, 0.0])
  x -> f(x, fit.param), fit.param...
end

function linsp(xs; step=1e-4)
  [minimum(xs):step:maximum(xs);]
end

function f2s(x)
  @sprintf "%.3f" x
end

function sf2s(x)
  @sprintf "%+.3f" x
end

function check_criticality(; H, Ns=[10:10:100;])
  χ = [[40, 60, 60, 60, 80, 80]; [120 for _ in 1:15]]
  energy0 = []
  energy1 = []
  for N in Ns
    H_mpo, sites = H(; N)
    E0, psi0 = dmrg(H_mpo, randomMPS(sites; linkdims=40); nsweeps=length(χ), maxdim=χ, cutoff=1e-12, noise=1e-10)
    E1, _ = dmrg(H_mpo, [psi0], randomMPS(sites; linkdims=40); nsweeps=length(χ), maxdim=χ, cutoff=1e-12, noise=1e-10, weight=100)
    push!(energy0, E0)
    push!(energy1, E1)
  end

  fitrange = 5:10

  xs = Ns .^ -1
  ys = energy0 ./ Ns
  scatter(xs, ys, xlabel="N^-1", ylabel="E0/N")
  f, a, b = linfit(xs[fitrange], ys[fitrange])
  plot!(linsp(xs), f(linsp(xs)), label="y=$(f2s(a))x+$(f2s(b))")
  savefig("plot/gs.png")
  vc = a * 6 / pi

  xs = Ns .^ -1
  ys = energy1 - energy0
  scatter(xs, ys, xlabel="N^-1", ylabel="E1 - E0")
  f, a, b = linfit(xs[fitrange], ys[fitrange])
  plot!(linsp(xs), f(linsp(xs)), label="y=$(f2s(a))x+$(f2s(b))")
  savefig("plot/gap.png")
  v = a * 4 / pi
  c = vc / v
  println("v=$v, c=$c")
end

function U_ising(; sites, J=1, Γ=1, tau)
  N = length(sites)
  gates = ITensor[]

  s1 = sites[1]
  s2 = sites[2]
  hj = -4J * op("Sz", s1) * op("Sz", s2) - 2Γ * op("Sx", s1) * op("I", s2) - Γ * op("I", s1) * op("Sx", s2)
  push!(gates, exp(tau / 2 * hj))
  for j in 2:N-2
    s1, s2 = sites[j], sites[j+1]
    hj = -4J * op("Sz", s1) * op("Sz", s2) - Γ * op("Sx", s1) * op("I", s2) - Γ * op("I", s1) * op("Sx", s2)
    push!(gates, exp(tau / 2 * hj))
  end
  s1, s2 = sites[end-1], sites[end]
  hj = -4J * op("Sz", s1) * op("Sz", s2) - Γ * op("Sx", s1) * op("I", s2) - 2Γ * op("I", s1) * op("Sx", s2)
  push!(gates, exp(tau / 2 * hj))
  append!(gates, reverse(gates))
  gates
end

function autocor(; state, tau, T, cutoff=1e-8)
  sites = siteinds(state)
  op = OpSum()
  op += "Sz", 1
  Sz1 = MPO(op, sites)
  a = apply(Sz1, state)

  gates = U_ising(; sites, tau)
  gates_inv = U_ising(; sites, tau=-tau)

  c = Float64[]
  for i in eachindex(tau:tau:T)
    state = apply(gates, state; cutoff)
    b = apply(Sz1, state)
    b = apply(vcat([gates_inv for _ in 1:i]...), b; cutoff)
    push!(c, inner(a, b))
  end
  c
end

function autocor2(; state, tau, T, cutoff=1e-8)
  sites = siteinds(state)
  H, _ = H_ising(; N=length(sites), sites)
  op = OpSum()
  op += "Sz", 1
  Sz1 = MPO(op, sites)
  a = apply(Sz1, state)

  c = Float64[]
  for t in tau:tau:T
    state2 = tdvp(H, state, -t; cutoff, normalize=true, outputlevel=1, reverse_step=false)
    b = apply(Sz1, state2)
    push!(c, abs(inner(a, b)))
  end
  c
end

function dmrg_tfising(; N, sites=siteinds("S=1/2", N), maxdim, cutoff)
  H, _ = H_ising(; N, sites)
  psi0 = randomMPS(sites, linkdims=40)
  dmrg(H, psi0; nsweeps=length(maxdim), maxdim, cutoff)
end

function impl(; N, tau, T, cutoff)
  !ispath("data") && mkdir("data")
  _, psi0 = dmrg_tfising(; N, maxdim=[20, 80, 80, 120, 120, 120, 120], cutoff)
  c = autocor(; state=psi0, tau, T, cutoff)
  open("data/N$N-tau$tau.txt", "w") do fp
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

function plotdata(; N, tau, fitrange=1:3, subtrexp=true)
  !ispath("plot") && mkdir("plot")
  data = read("data/N$N-tau$tau.txt")
  c = vcat(data...)
  y = log.(abs.(c))
  t = tau:tau:(tau*length(c))
  sslabel = "\\ln\\ \\langle \\sigma^z_1(\\tau)\\sigma^z_1(0)\\rangle"

  scatter(t, y, xlabel=L"\tau", ylabel=L"%$sslabel", label="", stroke=1, marker=:+)
  f, a, b = linfit(t[fitrange], y[fitrange])
  plot!(linsp(t), f(linsp(t)), label=L"y=%$(f2s(a))x%$(sf2s(b))"*"\n"*L"\quad\quad (\mathrm{fitrange}=%$(fitrange))")
  savefig("plot/N$N-tau$tau.png")

  y -= a * t .+ b
  scatter(t, y, xlabel=L"\tau", ylabel=L"%$sslabel %$(sf2s(-a)) \tau %$(sf2s(-b))", label="", stroke=1, marker=:+)
  savefig("plot/N$N-tau$tau-subtrexp.png")
end

function subtrexp(; N, tau, fitrange=1:3)
  !ispath("plot") && mkdir("plot")
  data = read("data/N$N-tau$tau.txt")
  v = 1
  c = vcat(data...)
  y = log.(abs.(c))
  t = tau:tau:(tau*length(c))

  scatter(t, y, xlabel="τ", ylabel="ln <σ^z_1(τ)σ^z_1(0)>")
  f, a, b = linfit(t[fitrange], y[fitrange])
  plot!(linsp(t), f(linsp(t)), title="fitrange$fitrange", label="y=$(a)x+$(b))")
  savefig("plot/N$N-tau$tau.png")
end

function a4N(; tau=1e-3, Ns=[10:10:100;])
  a = []
  for N in Ns
    push!(a, plotdata(; N, tau, fitrange=1:3)[1])
  end
  scatter(Ns, a)
  f(x, p) = @. p[1] * x + p[2]
  fit = curve_fit(f, Ns, a, [0.0, 0.0])
  plot!(Ns, f(Ns, fit.param), label="y = $(fit.param[1])x + $(fit.param[2])")
  savefig("plot/a4N.png")
  a
end

begin
  if length(ARGS) < 3
    println("usage:")
    println("julia ising-autocor.jl N tau T [cutoff]")
    return
  end
  maxerr = 1e-10
  if length(ARGS) >= 4
    maxerr = parse(Float64, ARGS[4])
  end
  if contains(ARGS[1], ':')
    Ns = nothing
    v = split(ARGS[1], ':')
    if length(v) == 2
      Ns = collect(parse(Int, v[1]):parse(Int, v[2]))
    elseif length(v) == 3
      Ns = collect(parse(Int, v[1]):parse(Int, v[2]):parse(Int, v[3]))
    end
    for N in Ns
      impl(; N, tau=parse(Float64, ARGS[2]), T=parse(Float64, ARGS[3]), cutoff=maxerr)
    end
  else
    impl(; N=ARGS[1], tau=parse(Float64, ARGS[2]), T=parse(Float64, ARGS[3]), cutoff=maxerr)
  end
end
