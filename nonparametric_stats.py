import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import matplotlib.colors as mcolors
from scipy.special import erfinv

# Measure run times of generator functions run in random sequence
def genDurationsDists(nReps, *genFuncs):
    
    nFuncs = len(genFuncs) # Count number of function arguments
    durations = np.zeros(nFuncs*nReps)  # initialize output array
    # initialize an array to contain the run-sequence of functions
    funcSeq = np.ones(nFuncs*nReps).astype(int) 
    
    # Build sequence block-by-block before shuffling.
    for m in ((i,(i + 1)) for i in range(0,nFuncs)):
        funcSeq[slice(m[0]*nReps, m[1]*nReps)] = m[0] # assign function indices
    
    # shuffle the ordering of the functions in sequence
    np.random.shuffle(funcSeq)  
    
    # Iterate through function sequence and measure run-times
    for i in range(0, funcSeq.shape[0]):
        tStart = time.time() # start timestamp
        genFuncs[funcSeq[i]]() # run one of the functions based on funcSeq
        durations[i] = time.time() - tStart # end timestamp, calc duration
    
    # separate and return the n sets of durations
    return (durations[funcSeq == i] for i in range(0, nFuncs)) # generator object!

# Plot multiple histgrams on single axes o bject
def histPlotter(nBins, *dists, **kwargs):
    
    nDists = len(dists) # count number of distributions to plot
    
    # determine common domain that encompasses all distributions
    if nDists == 1: # no concatenation in single histogram case 
        _, binEdges = np.histogram(*(d for d in dists), bins=nBins)
    else:
        _, binEdges = np.histogram(np.concatenate([d for d in dists]), bins=nBins)

    # initalize array that will hold y values for each histogram
    histArray = np.zeros((nDists, binEdges.shape[0] - 1)).astype(int)
    
    # Generate histo for each dist within common bin domain
    for d in range(0, nDists):
        histArray[d,:], _ = np.histogram(dists[d], bins=binEdges)
    
    # If an axes object was given as arg, use it for plotting,
    # otherwise, create a new one and plot on it.
    if "axes" in kwargs.keys():
        if (str(type(kwargs["axes"])) == 
            "<class 'matplotlib.axes._subplots.AxesSubplot'>"):
            ax = kwargs["axes"]
    else:
        _, ax = plt.subplots(nrows=1, ncols=1)

    # Use specified labels or label histos by their ordering in function call
    if "labels" in kwargs.keys():
        labels = kwargs["labels"]
    else:
        labels = np.arange(1, nDists +1).astype(str)

    if "colors" in kwargs.keys():
        colors = kwargs["colors"]
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    bWidth = binEdges[1] - binEdges[0] # calc width of bars for plotter
    
    # Plot each histo as translucent bar graph on common axes object
    for d in range(0, nDists):
        ax.bar(binEdges[:-1], histArray[d,:], align='edge', width=bWidth, 
               alpha=0.5, label=labels[d], color=mcolors.to_rgb(colors[d]))
    ax.legend() # include legend

    return ax # return axes pointer

# Difference between means of two distributions
def diffMeans(dist1, dist2):
    return np.mean(dist2) - np.mean(dist1)

# Difference between medians of two distributions
def diffMedians(dist1, dist2):
    return np.median(dist2) - np.median(dist1)

# Mean of distribution medians
def avgOfMedians(*dists):
    return np.mean([np.median(d) for d in dists])

# Monte Carlo simulated permutation tester
def permutatorMC(statFunc, nReps, *dists):

    statDist = np.zeros((nReps,)) # initialize the output array of simulated dist. stats 
    nDists = len(dists) # count number of dists. to process
    distIndices = range(0,nDists) # generate index list to select dists.
    sourceDists = np.concatenate([d for d in dists]) # pool together all distributions
    distLengths = np.array([d.shape[0] for d in dists]) # record length of each distrib

    # List of starting indices for each distribution in the pooled array
    transInds = np.concatenate([np.zeros((1,)), np.cumsum(distLengths)])

    # Initialize array of masks to all false
    selectionMasks = np.zeros((nDists, sourceDists.shape[0])).astype(bool)
    for i in distIndices: # flip corresp range of bits to true in each distrib mask
        selectionMasks[i, int(transInds[i]):int(transInds[i+1])] = True

    # For each repetition shuffle elements of pooled distribution array, extract 
    # dist-length snippets and insert into statistical function. Record output
    for r in range(0, nReps):
        np.random.shuffle(sourceDists)
        statDist[r] = statFunc(*(sourceDists[selectionMasks[i]] for i in distIndices))

    return statDist # return array of output from statistical function

def bootstrapMC(statFunc, nReps, *dists):

    statDist = np.zeros((nReps,)) # initialize the output array of simulated dist. stats 
    #nDists = len(dists) # count number of dists. to process
    #distIndices = range(0,nDists) # generate index list to select dists.
    sourceDists = np.concatenate([d for d in dists]) # pool together all distributions
    distLengths = np.array([d.shape[0] for d in dists]) # record length of each distrib
    totalLength = np.sum(distLengths)

    for r in range(0, nReps):
        
        statDist[r] = statFunc(*(sourceDists[
            np.rint((totalLength-1)*np.random.random_sample(size=l)).astype(int)
            ] for l in distLengths))

    return statDist

def confLimits(statFunc, nReps, dist, alpha, **kwargs):
    
    t_star = np.zeros((nReps,))
    v_star = np.zeros((nReps,))
    totalLength = dist.shape[0]
    lowIndex = np.ceil((nReps + 1)*alpha).astype(int)
    highIndex = np.floor((nReps + 1)*(1 - alpha)).astype(int)
    
    for r in range(0, nReps):
        selMask = np.rint((totalLength-1)*np.random.random_sample(
                          size=totalLength)).astype(int)
        t_star[r] = statFunc(dist[selMask])
        v_star[r] = np.var(dist[selMask])
        
    t = statFunc(dist)
    v = np.var(dist)
    
    t_star = np.hstack((t_star, np.array([t])))
    v_star = np.hstack((v_star, np.array([v])))
    z_star = np.sort(np.divide((t_star - t*np.ones_like(t_star)), np.sqrt(v_star)))
    
    confLimit_low = t - np.sqrt(v)*z_star[highIndex]
    confLimit_high = t - np.sqrt(v)*z_star[lowIndex]
    
    return confLimit_low, confLimit_high

def confLimits2(tFunc, vLFunc, nReps, alpha, *dists, **kwargs):
    
    t_star = np.zeros((nReps,))
    vL_star = np.zeros((nReps,))
    totalLength = dists[0].shape[0]
    lowIndex = np.ceil((nReps + 1)*alpha).astype(int)
    highIndex = np.floor((nReps + 1)*(1 - alpha)).astype(int)
    
    for r in range(0, nReps):
        selMask = np.rint((totalLength-1)*np.random.random_sample(
                          size=totalLength)).astype(int)
        
        t_star[r] = tFunc(*(d[selMask] for d in dists))
        vL_star[r] = vLFunc(*(d[selMask] for d in dists))
        
    t = tFunc(*(dists))
    vL = vLFunc(*(dists))
    
    t_star = np.hstack((t_star, np.array([t])))
    vL_star = np.hstack((vL_star, np.array([vL])))
    z_star = np.sort(np.divide((t_star - t*np.ones_like(t_star)), np.sqrt(vL_star)))
    
    confLimit_low = t - np.sqrt(vL)*z_star[highIndex]
    confLimit_high = t - np.sqrt(vL)*z_star[lowIndex]
    
    return confLimit_low, confLimit_high

# Ratio of means function
def ratioOfMeans(distX, distU):
    return np.mean(distX)/np.mean(distU)

# Estimated variance of ratio of means via delta method
def vL_ratioOfMeans(distX, distU):
    n = distX.shape[0]
    t = ratioOfMeans(distX, distU)
    l_j = (distX - t*distU)/np.mean(distU)
    return (1./(n**2))*np.sum(np.power(l_j, 2))

# Normalized Gaussian pdf function
def normalPDF(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mu)/sigma)**2)

# Inverse of cumulative normal distribution
def inverseCDF_norm(p, mu, sigma):
    return mu + sigma*np.sqrt(2)*erfinv(2.*p - 1.)

# Normal distribution quantile constructor 
def histConst_norm(dist, binEdges):
    mu = np.mean(dist)
    sigma = np.std(dist)
    hist_norm = normalPDF(binEdges[1:], mu, sigma)
    return binEdges, hist_norm 

# Q-Q plotter: data quantiles over theoretical normal quantiles
def qqPlotter_normal(dist, nBins, **kwargs):
    
    # Calculate parameters for construction of theoretical normal
    mu = np.mean(dist) # calculate mean
    sigma = np.std(dist) # calculate standard deviation

    # Compute histogram of data distribution
    hist, binEdges = np.histogram(dist, bins=nBins)
    histCDF = np.cumsum(hist) # compute the cumulative histogram
    histCDF = histCDF/histCDF[-1] # normalize cumulative histogram
    
    # Use quantile progression from data distribution to calculate
    # corresponding quantile locations in a theoretical normal distribution
    inverseCDF = inverseCDF_norm(histCDF, mu, sigma)
    
    # If an axes object was given as arg, use it for plotting,
    # otherwise, create a new one and plot on it.
    if "axes" in kwargs.keys():
        if (str(type(kwargs["axes"])) == 
            "<class 'matplotlib.axes._subplots.AxesSubplot'>"):
            ax = kwargs["axes"]
    else:
        _, ax = plt.subplots(nrows=1, ncols=1)

    if "color" in kwargs.keys():
        color = kwargs["color"]
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color = prop_cycle.by_key()['color'][0]

    # Include identity line in background for reference
    ax.plot(binEdges, binEdges, '--', color='gray', linewidth=1.0)

    # plot coordinate locations of datapoints; 
    # data along y-axis, theoretical along x axis
    marker = markers.MarkerStyle(marker='+')
    ax.scatter(inverseCDF, binEdges[1:], marker=marker, c=color) 
    ax.set_aspect('equal') # set aspect-ratio of plot axes to equal
    
    return ax # return pointer to axes object