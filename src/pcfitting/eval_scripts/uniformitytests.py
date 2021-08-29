import random
import numpy as np
import statistics

intervals = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
# intervals = [[0, 5], [6, 10]]
# intervals = [[0, 10]]
# intervals = [[0, 3], [7, 10]]
# intervals = [[0, 2], [4, 6], [8, 10]]
# intervals = [[0, 6]]
spacelen = 10
spaceintegral = 5# 6
nrpoints = 3000

points = []

for interval in intervals:
    pointsininterval = int(nrpoints / len(intervals))
    points.extend([random.uniform(interval[0], interval[1]) for x in range(pointsininterval)])
points.sort()

points = np.array(points)

nndists = [np.abs(points - sample).min() for sample in np.arange(0, spacelen, (1.0/nrpoints))]
densities = []

for sample in np.arange(0, spacelen, (1.0/nrpoints)):
    done = False
    for i in intervals:
        if sample >= i[0] and sample <= i[1]:
            densities.append((1/len(intervals))/(i[1] - i[0]))
            done = True
            break
    if not done:
        densities.append(0)

avgdist = statistics.mean(nndists)
stddist = statistics.stdev(nndists)
cvdist = stddist / avgdist
avgdens = statistics.mean(densities)
stddens = statistics.stdev(densities)
cvdens = stddens / avgdens
print("Sigma_D: ", stddens)
print("V_D:     ", cvdens)
print("STD_E:   ", stddist)
print("CV_E:    ", cvdist)

# intervals = [[0, 2], [4, 6], [8, 10]]
# Sigma_D:  0.08164875008301828
# V_D:      0.81639679007573
# STD_E:    0.3055369317469711
# CV_E:     1.5197496166294757

# intervals = [[0, 6]]
# Sigma_D:  0.08164988471561417
# V_D:      0.8164534886289957
# STD_E:    1.2216019590172222
# CV_E:     1.5259383634657029

# intervals = [[0, 3], [7, 10]]
# Sigma_D:  0.08164988471561417
# V_D:      0.8164534886289957
# STD_E:    0.6111942633402606
# CV_E:     1.5234352703556235

# intervals = [[0, 10]]
# Sigma_D:  0.0
# V_D:      0.0
# STD_E:    0.0016755323919209293
# CV_E:     1.0033282732497124

# intervals = [[0, 5], [6, 10]]
# Sigma_D:  0.035351213734457626
# V_D:      0.35350035399944296
# STD_E:    0.0883447083750404
# CV_E:     3.3088401458132495

# intervals = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
# Sigma_D:  0.10000166115268619
# V_D:      0.9996833837322845
# STD_E:    0.21046471807452993
# CV_E:     1.3909906393570273