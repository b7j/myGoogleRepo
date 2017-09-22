from __future__ import division
import numpy as np
from scipy import stats

def mmk_test(v, alpha = 0.02):
    """
    Performs the Modified Mann-Kendall test to check if there is any trend present.
    Based on Mann-Kendall test code by Sat Kumar Tomer. Modified to account for autocorrelation (Hamed and Rao 1998)
    
    Input:
        v: a vector
        alpha: significance level
        (example: if alpha is 0.05, 95% confidence)
    
    Output:
        h: True (if trend is present) or False (if trend is absent)
        trend: the slope as the median of all slopes between paired values (Sen, 1968)
        p: p value of the significance test
        z: normalised test statistic
    """
    
    n = v.shape[0]
    
    # calculate s
    s = 0
    for i in xrange(n-1):
        for j in xrange(i+1,n):
            s += np.sign(v[j] - v[i])
            
    # calculate variance of s
    v_uniq = np.unique(v)
    g = v_uniq.shape[0]
    if n==g: # no tie
        var_s = n*(n-1)*(2*n+5)/18
    else: # tied groups
        tp = np.zeros(g)
        for i in xrange(g):
            tp[i] = sum(v_uniq[i] == v)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
        
    # detrend
    t = stats.theilslopes(v)
    xx = range(1,n+1)
    v_detrend = v - np.multiply(xx,t[0])
    
    # account for autocorrelation
    I = np.argsort(v_detrend)
    d = n * np.ones(2 * n - 1)
    acf = (np.correlate(I, I, 'full') / d)[n - 1:]
    acf = acf / acf[0]
    interval = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    u_bound = 0 + interval;
    l_bound = 0 - interval;
    
    sni = 0
    for i in xrange(1,n-1):
        if (acf[i] > u_bound or acf[i] < l_bound):
            sni += (n-i) * (n-i-1) * (n-i-2) * acf[i]
    n_ns = 1 + (2 / (n * (n-1) * (n-2))) * sni
    v_s = var_s * n_ns
    
    # calculate z (normalised test statistic)    
    if s > 0:
        z = (s - 1)/np.sqrt(v_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(v_s)
        
    # significance
    p = 2*(1-stats.norm.cdf(abs(z))) # two tail test
    h = abs(z) > stats.norm.ppf(1-alpha/2)
    
    # trend
    if h:
        trend = t[0]
    else:
        trend = 0
    
    return h, trend, p, z