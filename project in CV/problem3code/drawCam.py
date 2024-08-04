import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

def drawCam(R, t):
    scale = 6.0
    P = scale * np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.8],
                          [0.5, -0.5, 0.8],
                          [-0.5, 0.5, 0.8],
                          [-0.5, -0.5, 0.8]])
    t = t.reshape(-1, 1)
    P1_ = R.T @ (P.T - np.tile(t, (1, 5)))
    P1_ = P1_.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def P1(i, j):
        return P1_[i - 1, j - 1]

    def line(X, Y, Z, color = 'blue', **kwargs):
        ax.plot(Y, X, Z, color = color)

    line([P1(1,1), P1(2,1)], [P1(1,3), P1(2,3)], [P1(1,2), P1(2,2)], color = 'black')
    line([P1(1,1), P1(3,1)], [P1(1,3), P1(3,3)], [P1(1,2), P1(3,2)], color = 'black')
    line([P1(1,1), P1(4,1)], [P1(1,3), P1(4,3)], [P1(1,2), P1(4,2)], color = 'black')
    line([P1(1,1), P1(5,1)], [P1(1,3), P1(5,3)], [P1(1,2), P1(5,2)], color = 'black')

    line([P1(2,1), P1(3,1)], [P1(2,3), P1(3,3)], [P1(2,2), P1(3,2)], color = 'black')
    line([P1(3,1), P1(5,1)], [P1(3,3), P1(5,3)], [P1(3,2), P1(5,2)], color = 'black')
    line([P1(5,1), P1(4,1)], [P1(5,3), P1(4,3)], [P1(5,2), P1(4,2)], color = 'black')
    line([P1(4,1), P1(2,1)], [P1(4,3), P1(2,3)], [P1(4,2), P1(2,2)], color = 'black')

    cameraPlane = [[P1(2,1), P1(2,3),  P1(2,2)], [P1(4,1), P1(4,3), P1(4,2)], [P1(3,1), P1(3,3), P1(3,2)], [P1(5,1), P1(5,3), P1(5,2)]]
    faces =[1, 0, 2, 3]
    cX = [cameraPlane[p][0] for p in faces]
    cY = [cameraPlane[p][1] for p in faces]
    cZ = [cameraPlane[p][2] for p in faces]

    verts = [list(zip(cY, cX, cZ))]
    patch = art3d.Poly3DCollection(verts, facecolors='green')
    ax.add_collection3d(patch, zs='z')

    C1 = np.array([P1(2,1), P1(2,3), P1(2,2)])
    C2 = np.array([P1(3,1), P1(3,3), P1(3,2)])
    C3 = np.array([P1(4,1), P1(4,3), P1(4,2)])
    C4 = np.array([P1(5,1), P1(5,3), P1(5,2)])

    O = np.array([P1(1,1), P1(1,3), P1(1,2)])
    Cmid = 0.25 * (C1 + C2 + C3 + C4);

    Lz = np.stack([O, O + 0.5 * (Cmid - O)])
    Lx = np.stack([O, O + 0.5 * (C1 - C3)])
    Ly = np.stack([O, O + 0.5 * (C1 - C2)])
    line(Lz[:, 0], Lz[:, 1], Lz[:, 2], color = 'blue', linewidth = 2)
    line(Lx[:, 0], Lx[:, 1], Lx[:, 2], color = 'green', linewidth = 2)
    line(Ly[:, 0], Ly[:, 1], Ly[:, 2], color = 'red', linewidth = 2)

    plt.show()

if __name__ == "__main__":
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float32)
    t = np.array([0, 0, 0], dtype = np.float32)
    drawCam(R, t)
