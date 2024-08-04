import cv2
from warpH import warpH
from cpselect.cpselect import cpselect
import numpy as np

if __name__ == "__main__":
    im1 = cv2.imread("taj1.jpg")
    im2 = cv2.imread("taj2.jpg")
    if False:
        point_list = cpselect("taj1.jpg", "taj2.jpg")
        p1 = []
        p2 = []
        for point in point_list:
            p1.append([point["img1_x"], point["img1_y"]])
            p2.append([point["img2_x"], point["img2_y"]])

        p1 = np.array(p1)
        p2 = np.array(p2)
        np.savez("taj_points.npz", p1 = p1, p2 = p2)
    else:
        points = np.load("taj_points.npz")
        p1 = points["p1"]
        p2 = points["p2"]
