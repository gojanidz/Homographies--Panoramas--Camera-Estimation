import matplotlib.pyplot as plt
import scipy.io as io

def visualize2Dpoints(xy):
    adj = io.loadmat("connectMat.mat")["connectMat"]
    pointX = [v[0] for v in xy]
    pointY = [v[1] for v in xy]
    plt.figure(figsize=(10, 10))
    plt.title("2D Points")
    plt.scatter(pointX, pointY)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                X = [pointX[i], pointX[j]]
                Y = [pointY[i], pointY[j]]
                plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    xy = io.loadmat("hw1_problem3.mat")["xy"]
    visualize2Dpoints(xy)
