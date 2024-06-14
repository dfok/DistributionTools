# DistributionTools.jl Documentation
This package is not registered, to install use
`Pkg.add("https://github.com/dfok/DistributionTools.jl)`

DistributionTools provides some functions to work with a few multivariate distributions from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). The main additions are functions to obtain marginal and conditional distributions.

It also provides a function to obtain the cdf of a mixture of bivariate normals. For this [GaussianDistributions.jl](https://github.com/mschauer/GaussianDistributions.jl) is used.

```@contents
```

## Obtaining marginal distributions
```@docs
marginal
```

## Obtaining conditional distributions
```@docs
conditional
```

## Other helper functions
```@docs
cor2var
var2cor
my_quantile
cdf
```