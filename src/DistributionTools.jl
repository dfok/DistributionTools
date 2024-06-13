module DistributionTools

using Distributions
using GaussianDistributions
import GaussianDistributions: cdf

using LinearAlgebra

export marginal, conditional, my_quantile, cor2var, var2cor, cdf

"""
    var2cor(m)
    
Transform a variance matrix to a correlation matrix
"""
function var2cor(m)
    s = sqrt.(diag(m))
    return (m ./ s) ./ s'
end

"""
    cor2var(cor, var)
    
Transform a correlation + vector of variances to a 2x2 variance matrix
"""
function cor2var(cor, var)
    m = [1 cor; cor 1]
    Diagonal(sqrt.(var))*m*Diagonal(sqrt.(var))
end

"""
    cdf(d::MixtureModel{Multivariate,Continuous, <:MvNormal, <:DiscreteNonParametric}, x::Vector{T}) where T<:Real    

Return cdf of mixture of multivariate normals evaluated at `x` 

Throws error when dimension of `d` ≠ 2
"""
function cdf(d::MixtureModel{Multivariate,Continuous, <:MvNormal, <:DiscreteNonParametric}, x::Vector{T}) where T<:Real
    @assert length(d) == 2 "Dimension needs to be 2 for cdf of mixture of multivariate normals"

    cdfs = [cdf(Gaussian(c.μ, c.Σ), x) for c in d.components]        
    return cdfs'd.prior.p
end

"""
    my_quantile(d::UnivariateMixture{Continuous}, p::Real)

Return the `p` quantile of the univariate mixture distribution `d`

Adapted from quantile() from Distributions.jl, original has a problem with the bisect algorithm. 

The initial interval does not contain solution, solved by ±0.000001
"""
function my_quantile(d::UnivariateMixture{Continuous}, p::Real)
    ps = probs(d)
    min_q, max_q = extrema(quantile(component(d, i), p) for (i, pi) in enumerate(ps) if pi > 0)
    min_q = min_q - 0.000001
    max_q = max_q + 0.000001
    Distributions.quantile_bisect(d, p, min_q, max_q)
end

# Marginals of various multivariate distributions
"""
    marginal(d::MvNormal, ind::Int)

Return the marginal distribution of element `ind` from the multivariate normal distribution `d`
"""
function marginal(d::MvNormal, ind::Int)
    return Normal(d.μ[ind], sqrt(d.Σ[ind,ind]))
end

"""
    marginal(d::MvLogNormal, ind::Int)

Return the marginal distribution of element `ind` from the multivariate log normal distribution `d`
"""
function marginal(d::MvLogNormal, ind::Int)
    return LogNormal(d.normal.μ[ind], sqrt(d.normal.Σ[ind,ind]))
end

"""
    marginal(d::MvNormal, ind::Array{Int,1})

Return the marginal distribution of elements `ind` (vector) from the multivariate normal distribution `d`
"""
function marginal(d::MvNormal, ind::Array{Int,1})
    return MvNormal(d.μ[ind], d.Σ[ind,ind])
end

"""
    marginal(d::MvLogNormal, ind::Array{Int,1})

Return the marginal distribution of elements `ind` (vector) from the multivariate log normal distribution `d`
"""
function marginal(d::MvLogNormal, ind::Array{Int,1})
    return MvLogNormal(marginal(d.normal, ind))
end

"""
    marginal(d::Mixture, ind::Int)

Return the marginal distribution of element `ind` from a multivariate mixture distribution `d`
"""
function marginal(d::MixtureModel{Multivariate,Continuous,<:Distribution,<:DiscreteNonParametric}, ind::Int)
      K = length(d.prior.p)

      marg = [marginal(d.components[k], ind) for k in 1:K]

      return MixtureModel(marg, d.prior.p)
end

"""
    marginal(d::Mixture, ind::Array{Int,1})

Return the marginal distribution of elements `ind` (vector) from a multivariate mixture distribution `d`
"""
function marginal(d::MixtureModel{Multivariate,Continuous,<:Distribution,<:DiscreteNonParametric}, ind::Array{Int,1})
      K = length(d.prior.p)

      marg = [marginal(d.components[k], ind) for k in 1:K]

      return MixtureModel(marg, d.prior.p)
end

"""
    marginal(d::MvTDist, ind::Int)

Return the marginal distribution of element `ind` from a multivariate t distribution `d`
"""
function marginal(d::MvTDist, ind::Int)
    return MvTDist(d.df, [d.μ[ind]], fill(d.Σ[ind,ind],1,1))
end

"""
    marginal(d::MvTDist, ind::Array{Int,1})

Return the marginal distribution of elements `ind` (vector) from a multivariate t distribution `d`
"""
function marginal(d::MvTDist, ind::Array{Int,1})
    return MvTDist(d.df, d.μ[ind], d.Σ[ind,ind])
end


# conditionals multivariate distribtuions
"""
    conditional(d::MultivariateDistribution, cond::Int, xval::T) where T<:Real

Return conditional distribution when conditioning element `cond` of distribution `d` to value `xval`
"""
function conditional(d::MultivariateDistribution, cond::Int, xval::T) where T<:Real
    conditional(d, [cond], [xval])
end

"""
    conditional(d::MvLogNormal, cond::Array{Int,1}, xval::Array{<:Real,1}) where T<:Real

Return conditional distribution when conditioning element vector `cond` of LogNormal distribution `d` to value `xval`
"""
function conditional(d::MvLogNormal, cond::Array{Int,1}, xval::Array{<:Real,1})
    condNormal = conditional(d.normal, cond, log.(xval))
    if condNormal isa Normal
    # Resulting distribution is univariate
        LogNormal(condNormal.μ, condNormal.σ)
    else
        return MvLogNormal(condNormal)
    end
end

"""
    conditional(d::MvNormal, cond::Array{Int,1}, xval::Array{<:Real,1}) where T<:Real

Returns conditional distribution when conditioning element vector `cond` of MvNormal distribution `d` to value `xval`
"""
function conditional(d::MvNormal, cond::Array{Int,1}, xval::Array{<:Real,1})
    all = 1:length(d)
    focal = setdiff(all, cond)

    condMean = d.μ[focal] + d.Σ[focal,cond]*inv(d.Σ[cond,cond])*(xval-d.μ[cond])
    condVar = Symmetric(d.Σ[focal,focal] - d.Σ[focal,cond]*inv(d.Σ[cond,cond])*d.Σ[cond,focal])
    
    if length(condMean)==1
        return Normal(condMean[], sqrt(condVar[]))
    else
        return MvNormal(condMean, condVar)
    end
end

""" 
    conditional(d::MvTDist, cond::Array{Int,1}, xval::Array{<:Real,1})

Obtain the conditional distribution of a multivariate T distribution 

See Peng Ding (2016) "On the Conditional Distribution of the Multivariate t-Distribution", The American Statistician, 70:3, 293-295, DOI: [10.1080/00031305.2016.1164756](https://doi.org/10.1080/00031305.2016.1164756)
"""
function conditional(d::MvTDist, cond::Array{Int,1}, xval::Array{<:Real,1})
    all = 1:length(d)
    focal = setdiff(all, cond)

    condLoc = d.μ[focal] + d.Σ[focal,cond]*inv(d.Σ[cond,cond])*(xval-d.μ[cond])
        
    d1 = (xval .- d.μ[cond])'inv(d.Σ[cond,cond])*(xval .- d.μ[cond])
    p1 = length(cond)
    
    condScale = Symmetric(d.Σ[focal,focal] - d.Σ[focal,cond]*inv(d.Σ[cond,cond])*d.Σ[cond,focal])
    condScale = ((d.df+d1)/(d.df+p1)).*condScale

    return MvTDist(d.df + p1, condLoc, condScale)
end

"""
    conditional(d::MixtureModel{Multivariate,Continuous,<:Distribution,<:DiscreteNonParametric}, cond::Array{Int,1}, xval::Array{<:Real,1})

Return the conditional distribution when conditioning element vector `cond` of Continuous Mixture distribution `d` to values `xval`

Drops elements from the mixture where the conditional mixture probability < eps()
"""
function conditional(d::MixtureModel{Multivariate,Continuous,<:Distribution,<:DiscreteNonParametric}, cond::Array{Int,1}, xval::Array{<:Real,1})
    K = length(d.prior.p)

    marg = [marginal(d.components[k], cond) for k in 1:K]
    newp = [log(d.prior.p[k]) + logpdf(marg[k], xval) for k in 1:K]

    newp = exp.(newp .- maximum(newp))
    newp ./= sum(newp)

    # skip components with cond prob ≈ 0
    sel = newp .> eps()
    newd = [conditional(d.components[k], cond, xval)  for k in (1:K)[sel]]
    
    return MixtureModel(newd, newp[sel]/sum(newp[sel]))
end

end # module DistributionTools
