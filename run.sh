#!/bin/sh

#SBATCH -p F1cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1

julia ising-autocor.jl 10:10:100 0.001 0.05 1e-10