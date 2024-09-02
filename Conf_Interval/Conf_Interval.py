#####################

import scipy.stats as stats

# z-distribution
# cdf => cumulative distributive function
stats.norm.cdf(680, 711, 29)  # Given a value, find the probability

# ppf => Percent point function;
stats.norm.ppf(0.1425, 0, 1) # Given probability, find the Z value

# t-distribution
stats.t.cdf(1.98, 139) # Given a value, find the probability
stats.t.ppf(0.9752, 139) # Given probability, find the t value