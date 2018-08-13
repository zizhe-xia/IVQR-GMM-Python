# IVQR-GMM-Python-codes

Python codes for exact computation of the IVQR GMM estimator. Detailed description of the estimator and computation can be found in the paper:

Chen, Le-Yu and Lee, Sokbae (September 2017), "Exact computation of GMM estimators for instrumental variable quantile regression models".

The paper has been published at Journal of Applied Econometrics. See https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2619.

Le-Yu Chen also provided a MATLAB version of this model here: https://github.com/LeyuChen/IVQR-GMM-computation-codes. The Python and R codes are inspired by the MATLAB codes.

Both the Python function named IVQR_GMM can be used to calculate the GMM estimator for instrumental variable quantile regression models. The implementation involves Gurobi solver and some additional function defined in the same file. The Gurobi solver is freely available for academic purposes.

There is also an empirical application of the functions on the New York Fulton fish market data to test the functions. The data is collected by Kathryn Graddy available at http://people.brandeis.edu/~kgraddy/datasets/fish.out. It consists of 111 observation on the price and quantity of whiting transactions everyday. The dependent variable Y is the logarithm of total amount of whitings sold each day. The endogenous explanatory variable D is the logarithm of the average daily price. The exogenous explanatory variables are the day indicators (Monday, Tuesday, Wednesday and Thursday). The instrumental variables are weather indicators (Stormy and Mixed). The application codes are also appended in the files.
