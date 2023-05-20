import matplotlib.pyplot as plt
import numpy as np

X, Y = [], []
with open('originalsCOUNTERFACTUALS_age_0_CHESTXRAY.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            X.append(float(x))

with open('cfsCOUNTERFACTUALS_age_0_CHESTXRAY.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        Y.append(float(x))

X = np.array(X)
Y = np.array(Y)
    
fig = plt.figure("figure")
plt.axhline(y=0.5, color='b', linestyle='-')
plt.axvline(x=0.5, color='b', linestyle='-')
plt.scatter(X,Y, c='red')
plt.savefig("plots/fairness_counterfactuals_cfs_age_0.png")
