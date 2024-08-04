import cv2

def warpH(im, H, out_size, fill_value = 0):
    """
        im: an image HxWx3
        H: a 3x3 transfomration matrix
        out_size: a tuple (W, H)
        fill_value: fill into empty regions
    """
    return cv2.warpPerspective(im, H, out_size, borderValue = fill_value)

