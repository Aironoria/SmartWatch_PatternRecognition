import numpy as np
from scipy.interpolate import CubicSpline      # for warping
import TripletLoss.triplet_data as triplet_data
import Utils
import matplotlib.pyplot as plt

def GenerateRandomCurves(X, sigma=0.2, knot=4,same_for_axis=True):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cubic_splines = [ CubicSpline(xx[:,i],yy[:,i]) for i in range(X.shape[1])]
    return np.array([cubic_splines[0 if same_for_axis else  i](x_range) for i in range(X.shape[1])]).transpose()

def DistortTimesteps(X, sigma=0.2,same_for_axis = True):
    tt = GenerateRandomCurves(X, sigma,same_for_axis=same_for_axis) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale= [ (X.shape[0]-1)/tt_cum[-1,i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:,i] *= t_scale[i]

    return tt_cum





def TimeWarping(X,sigma=0.2,same_for_axis = True): #[N,channels]
    tt = DistortTimesteps(X, sigma,same_for_axis)
    X_new = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(np.arange(X.shape[0]), tt[:,i], X[:,i])
    return X_new

def MagnitudeWarping(X,sigma=0.1):
    return X * GenerateRandomCurves(X, sigma)
def Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise

if __name__ == '__main__':
    train,validate,test = triplet_data. load_dataset("../assets/input/ten_data_","overall")
    sample = train[0][0].numpy().T


    after_sample = TimeWarping(sample,sigma=1.2,same_for_axis= False)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    # ax.set_ylim([-10, 10])

    for i in range(sample.shape[1]):
        ax.plot(sample[:,i],label=i)

    ax = fig.add_subplot(212)
    # ax.set_ylim([-10, 10])

    for i in range(after_sample.shape[1]):
        ax.plot(after_sample[:, i], label=i)
    plt.show()