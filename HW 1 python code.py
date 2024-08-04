#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import function
import cv2
import numpy as np
from typing import Tuple
from cpselect.cpselect import cpselect
from scipy.optimize import minimize


# In[19]:


#Following is the implementation of the homography estimation
#Q2.2
def computeH(p1:np.array, p2:np.array):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i, 0], p1[i, 1]
        u, v = p2[i, 0], p2[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)
    L = V[-1,:] / V[-1,-1]
    H = L.reshape(3, 3)
    return H


# In[3]:


#2.3
def normalize(x:np.array, nd:int=2):
    '''
    Normalization of coordinates
    Args:
        x: the data to be normalized
        nd: number of dimensions
    '''
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
    
    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate([x.T, np.ones((1,x.shape[0]))]))
    x = x[0:nd, :].T

    return Tr, x

def computeH_norm(p1:np.array, p2:np.array) -> np.array:
    T1, p1 = normalize(p1)
    T2, p2 = normalize(p2)

    H_ = computeH(p1, p2)
    H = np.dot(np.dot(np.linalg.pinv(T2), H_), T1)
    H = H / H[-1, -1]
    return H


# In[4]:


#selecting points
#2.4
im1 = cv2.imread("taj1.jpg")
im2 = cv2.imread("taj2.jpg")
point_list = cpselect("taj1.jpg", "taj2.jpg")
p1 = []
p2 = []
for point in point_list:
    p1.append([point["img1_x"], point["img1_y"]])
    p2.append([point["img2_x"], point["img2_y"]])
p1 = np.array(p1)
p2 = np.array(p2)
np.savez("taj_points.npz", p1 = p1, p2 = p2)


# In[5]:


#calculate homgraphy
im1 = cv2.cvtColor(cv2.imread("taj1.jpg"), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread("taj2.jpg"), cv2.COLOR_BGR2RGB)
points = np.load("taj_points.npz")
p1 = points["p1"]
p2 = points["p2"]
H2to1 = computeH_norm(p2,p1)
print(H2to1)


# In[6]:


#showing control points
p2_t = np.dot(H2to1, np.concatenate((p2.T, np.ones((1, p2.shape[0])))))
p2_t = p2_t / p2_t[2, :]
p2_t = p2_t.T[:,:2]
import matplotlib.pyplot as plt
plt.imshow(im1)
plt.scatter(p1[:,0], p1[:,1], c="r")
plt.savefig("img1_p1.png")
plt.show()
plt.imshow(im2)
plt.scatter(p2[:,0], p2[:,1], c="g")
plt.savefig("img2_p2.png")
plt.show()
plt.imshow(im1)
plt.scatter(p1[:,0], p1[:,1], c="r")
plt.scatter(p2_t[:,0], p2_t[:,1], c="g")
plt.savefig("q2_4.png")
plt.show()
np.savez("q2_4.npz", H2to1, p1, p2, p2_t)


# In[7]:


#error function
err = np.sqrt(np.mean(np.sum((p2_t - p1)**2, 1)))
print(err)


# In[8]:


#image wraping
def warpH(im, H, out_size, fill_value = 0):
    """
    im: an image HxWx3
    H: a 3x3 transfomration matrix
    out_size: a tuple (W, H)
    fill_value: fill into empty regions
    """
    return cv2.warpPerspective(im, H, out_size, borderValue = fill_value)


# In[9]:


im1 = cv2.cvtColor(cv2.imread("taj1.jpg"), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread("taj2.jpg"), cv2.COLOR_BGR2RGB)
points = np.load("taj_points.npz")
p1 = points["p1"]
p2 = points["p2"]
H2to1 = computeH_norm(p2,p1)
warp_im = warpH(im2, H2to1, (im1.shape[1], im1.shape[0]))
import matplotlib.pyplot as plt
plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
plt.imshow(warp_im)
plt.show()


# In[10]:


#Q2.5 Following code finds a matrix M and the output size. The matrix M does the scaling and translation.
def calcM(img1: np.array, img2: np.array, H:np.array):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2).astype(np.
    float32)
    pts2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2).astype(np.
    float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5).astype(np.int32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5).astype(np.int32)
    scale = 1280/(ymax-ymin)
    out_x = int(scale*(xmax-xmin))
    out_y = int(scale*(ymax-ymin))
    out_shape = (out_x, out_y)
    M = np.array([
    [scale,0,-scale*xmin],
    [0,scale,-scale*ymin],
    [0,0,1]])
    return M, out_shape


# In[11]:


#2.6
def warpTwoImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2).astype(np.
    float32)
    pts2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2).astype(np.
    float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5).astype(np.int32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5).astype(np.int32)
    scale = 1280/(ymax-ymin)
    out_x = int(scale*(xmax-xmin))
    out_y = int(scale*(ymax-ymin))
    out_shape = (out_x, out_y)
    M = np.array([
    [scale,0,-scale*xmin],
    [0,scale,-scale*ymin],
    [0,0,1]])
    result = cv2.warpPerspective(img2, M.dot(H), out_shape)
    tmp = np.zeros_like(result)
    tmp[int(-scale*ymin):int(-scale*ymin)+int(scale*h1),
    int(-scale*xmin):int(-scale*xmin)+int(scale*w1)] =\
    cv2.resize(img1, (int(scale*w1), int(scale*h1)))
    mask = np.logical_and(result > 0, tmp> 0)
    blend = 0.5*result[mask] + 0.5*tmp[mask]
    result[int(-scale*ymin):int(-scale*ymin)+int(scale*h1),
    int(-scale*xmin):int(-scale*xmin)+int(scale*w1)] =\
    cv2.resize(img1, (int(scale*w1), int(scale*h1)))
    result[mask] = blend
    return result
im1 = cv2.cvtColor(cv2.imread("taj1.jpg"), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread("taj2.jpg"), cv2.COLOR_BGR2RGB)


# In[12]:


points = np.load("taj_points.npz")
p1 = points["p1"]
p2 = points["p2"]
H2to1 = computeH_norm(p2,p1)
warp_im = warpTwoImages(im1, im2, H2to1)


# In[13]:


plt.imshow(warp_im)
plt.savefig("q2_7.png")
plt.show()


# In[14]:


#QX1 optimization
points = np.load("taj_points.npz")
p1 = points["p1"]
p2 = points["p2"]
alpha = 1274.4
beta = 1274.4
u0 = 814.1
v0 = 526.6
H2to1 = computeH_norm(p2,p1)
K = np.array([
[alpha, 0, u0],
[0, beta, v0],
[0, 0, 1]])
Ry = lambda x: np.array([
[np.cos(np.radians(x)), 0, np.sin(np.radians(x))],
[0, 1, 0],
[-np.sin(np.radians(x)), 0, np.cos(np.radians(x))]])
H = lambda x: K @ Ry(x) @ np.linalg.inv(K)
def objective(x):
    p2_t = np.dot(H(x), np.concatenate((p2.T, np.ones((1, p2.shape[0])))))
    p2_t = p2_t / p2_t[2, :]
    p2_t = p2_t.T[:,:2]
    return np.sqrt(np.mean(np.sum((p2_t - p1)**2, 1)))

res = minimize(objective, 0, tol=1e-3, method="Powell")
print(res)
p2_t = np.dot(H(res["x"]), np.concatenate((p2.T, np.ones((1, p2.shape[0])))))
p2_t = p2_t / p2_t[2, :]
p2_t = p2_t.T[:,:2]
plt.imshow(im1)
plt.scatter(p1[:,0], p1[:,1], c="r")
plt.scatter(p2_t[:,0], p2_t[:,1], c="b")
plt.show()


# In[15]:


#QX2
A = cv2.cvtColor(cv2.imread(r"C:\Users\Administrator\Desktop\Computer vision\apple.png"), cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(cv2.imread(r"C:\Users\Administrator\Desktop\Computer vision\orange.png"), cv2.COLOR_BGR2RGB)

# generate Gaussian pyramid
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid
lpA = [gpA[5]]
for i in range(5,0,-1):
    size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
    GE = cv2.pyrUp(gpA[i], dstsize = size)
    L = cv2.subtract(gpA[i-1], GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5,0,-1):
    size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
    GE = cv2.pyrUp(gpB[i], dstsize = size)
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# reconstruct
ls_ = LS[0]
for i in range(1,6):
    size = (LS[i].shape[1], LS[i].shape[0])
    ls_ = cv2.pyrUp(ls_, dstsize = size)
    ls_ = cv2.add(ls_, LS[i])

plt.imshow(ls_)
plt.show()

real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
plt.imshow(real)
plt.show()


# In[16]:


# Q3.1 Estimating the camera
def DLT(xyz, uv):
    '''
        Args:
        xyz: coordinates in the object 3D space.
        uv: coordinates in the image 2D space.
    '''
    n = xyz.shape[0]
    assert n >= 6, "[*] DLT requires at least 6 calibration points."
    assert uv.shape[0] == n
    assert xyz.shape[1] == 3
    Txyz, xyzn = normalize(xyz, 3)
    Tuv, uvn = normalize(uv, 2)
    A = []
    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)
    L = V[-1, :] / V[-1, -1]
    H = L.reshape(3, 3 + 1)
    # Denormalization
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    H = H / H[-1, -1]
    return H
def estimateCamera(xy:np.array, XYZ:np.array) -> Tuple[np.array, np.array]:
    # assume K = np.eye(3)
    # estimate camera matrix (3x4)
    P = DLT(XYZ, xy)
    return P[:,0:3], P[:,3]


# In[17]:


# Q3.5 Following is the code to estimate camera rotation and transformation.
from scipy.linalg import rq
def estimateCamera2(xy:np.array, XYZ:np.array) -> Tuple[np.array, np.array]:
    # estimate camera matrix (3x4)
    P = DLT(XYZ, xy)
    # Estimate C
    U, S, V = np.linalg.svd(P)
    C = V[-1,0:3] / V[-1, -1]
    # Estimating K and R by RQ decomposition.
    K, R = rq(P[0:3,0:3])
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R
    t = -R @ C
    return R, t


# In[18]:


# Q3.6
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as io
import mpl_toolkits.mplot3d.art3d as art3d
def visualize3Dpoints(XYZ, fig, ax):
    adj = io.loadmat("connectMat.mat")["connectMat"]
    pointX = [v[2] for v in XYZ]
    pointY = [v[0] for v in XYZ]
    pointZ = [v[1] for v in XYZ]
    plt.title("3D Points")
    ax.scatter(pointX, pointY, pointZ)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                X = [pointX[i], pointX[j]]
                Y = [pointY[i], pointY[j]]
                Z = [pointZ[i], pointZ[j]]
                ax.plot(X, Y, Z)
def drawCam(R, t, fig, ax):
    scale = 6.0
    P = scale * np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.8],
                          [0.5, -0.5, 0.8],
                          [-0.5, 0.5, 0.8],
                          [-0.5, -0.5, 0.8]])
    t = t.reshape(-1, 1)
    P1_ = R.T @ (P.T - np.tile(t, (1, 5)))
    P1_ = P1_.T
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

    cameraPlane = [[P1(2,1), P1(2,3), P1(2,2)], [P1(4,1), P1(4,3), P1(4,2)], [P1(3,1), P1(3,3), P1(3,2)], [P1(5,1), P1(5,3), P1(5,2)]]
    faces =[1, 0, 2, 3]
    cX = [cameraPlane[p][0] for p in faces]
    cY = [cameraPlane[p][1] for p in faces]
    cZ = [cameraPlane[p][2] for p in faces]

    verts = [list(zip(cY, cX, cZ))]
    patch = art3d.Poly3DCollection(verts, facecolors='green')
    # ax.add_collection3d(patch, zs='z')

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
    
xyz = io.loadmat("hw1_problem3.mat")["XYZ"]
uv = io.loadmat("hw1_problem3.mat")["xy"][:,:2]
R, t = estimateCamera2(uv, xyz)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([xyz[:,0].max()-xyz[:,0].min(), xyz[:,1].max()-xyz[:,1].min(), t[2]-xyz[:,2].min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xyz[:,0].max()+xyz[:,0].min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(xyz[:,1].max()+xyz[:,1].min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(t[2]+xyz[:,2].min())
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

visualize3Dpoints(xyz, fig, ax)
drawCam(R, t, fig, ax)
plt.show()


# In[ ]:




