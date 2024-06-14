using Documenter, DistributionTools
push!(LOAD_PATH,"../src/")
makedocs(modules = [DistributionTools], sitename="DistributionTools.jl")