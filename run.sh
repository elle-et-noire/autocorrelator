#!/bin/sh

#SBATCH -p F2fat
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1

julia ising-autocor.jl 10:10:100 0.01 1.0 1e-10