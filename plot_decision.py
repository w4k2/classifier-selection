from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def plot_decision(ax, clf, X, y, points = True, h=.1):
    xx, yy = np.meshgrid(np.arange(-3, 3.1, h),
                         np.arange(-3, 3.1, h))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    A = Z[:,0] - .5
    B = Z[:,0] > 0

    A = A.reshape(xx.shape)
    B = B.reshape(xx.shape)

    #ax.contourf(xx, yy, Z, cmap='bwr', levels = 25)
    #ax.contour(xx, yy, A, colors='black', levels = 0, linewidths=1, linestyles=":")
    ax.contour(xx, yy, B, colors='black', levels = 0, linewidths=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=1)

    return B

base_concept = make_classification(n_samples=200,n_features=2,n_classes=2,weights=[.1, .9],n_informative=2, n_redundant=0, random_state=30, n_clusters_per_class=1, flip_y=0)

X, y = base_concept
X = StandardScaler().fit(X).transform(X)
X[y==0] += [-1, 4]

v_s = [[0,0], [.5,-.5], [1,-.5]]

B_s = []
for i in range(3):
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    X_ = np.copy(X)
    X_[y==0] = X[y==0] + v_s[i]

    clf = KNeighborsClassifier()
    clf.fit(X_,y)

    ax.set_xlim((-2,2))
    ax.set_ylim((-2,3))

    b = plot_decision(ax, clf, X_, y)
    B_s.append(b)
    plt.tight_layout()
    plt.savefig("figures/fuser_%i.eps" % i)
    plt.savefig("figures/fuser_%i.png" % i)

B_s = np.array(B_s)
B = np.max(B_s, axis=0)
print(B_s.shape)
print(B.shape)

fig, ax = plt.subplots(1,1,figsize=(3,3))
h = .1
xx, yy = np.meshgrid(np.arange(-3, 3.1, h),
                     np.arange(-3, 3.1, h))
ax.contour(xx, yy, B, colors='black', levels = 0, linewidths=3)
ax.set_xlim((-2,2))
ax.set_ylim((-2,3))
ax.set_xticks([])
ax.set_yticks([])


for i in range(3):
    ax.contour(xx, yy, B_s[i], colors='black', levels = 0, linewidths=1,linestyles=":")


ax.scatter(X_[:,0], X_[:,1], c=y, cmap='bwr', alpha=1)
plt.tight_layout()
plt.savefig("figures/fuser_cmb.eps")
plt.savefig("figures/fuser_cmb.png")
