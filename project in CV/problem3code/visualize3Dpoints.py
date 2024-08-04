from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as io

def visualize3Dpoints(XYZ):
    adj = io.loadmat("connectMat.mat")["connectMat"]
    pointX = [v[2] for v in XYZ]
    pointY = [v[0] for v in XYZ]
    pointZ = [v[1] for v in XYZ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("3D Points")
    ax.scatter(pointX, pointY, pointZ)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                X = [pointX[i], pointX[j]]
                Y = [pointY[i], pointY[j]]
                Z = [pointZ[i], pointZ[j]]
                ax.plot(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    XYZ = io.loadmat("hw1_problem3.mat")["XYZ"]
    visualize3Dpoints(XYZ)
