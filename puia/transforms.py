import numpy as np

def inv(s):
    return 1./s

def diff(s):
    s = s.diff()
    s[0] = 0.
    return s

def log(s):
    return np.log10(s)

def zsc(s):
    # log transform data
    log_s = np.log10(s).replace([np.inf, -np.inf], np.nan).dropna()
    
    # compute mean/std/min
    mn = np.mean(log_s)
    std = np.std(log_s)
    minzsc = np.min(log_s)                                                    

    # Calculate percentile
    s=(np.log10(s)-mn)/std
    s=s.fillna(minzsc)
    s=10**s
    return s
    
def log_zsc(s):
    return log(zsc(s))

def zsc2(s):
    s=zsc(s)
    s=s.rolling(window=2).min()
    s[0]=s[1]
    return s

def log_zsc2(s):
    return log(zsc2(s))
    
def diff_zsc2(s):
    return diff(zsc2(s))

transform_functions={
    'inv':inv,
    'diff':diff,
    'log':log,
    'zsc':zsc,
    'log_zsc':log_zsc,
    'zsc2':zsc2,
    'log_zsc2':log_zsc2,
    'diff_zsc2':diff_zsc2
    }