import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from itertools import product
from scipy import signal

# ==================== Convolution ====================

def conv2(image, kernel):
    """
    Implement 2D convolution - convolve the kernel over
    the entire image while maintaining its original size
    """
    img = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return img

# ==================== Gaussian Smoothing ====================

def gaussian_kernel(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def GaussSmoothing(i, n, sigma):
    """
    Perform gaussian smoothing on image i with a kernel of size 
    (n,n) and standard deviation sigma.
    """
    gmask = gaussian_kernel((n, n), sigma)
    img = conv2(i, gmask)
    return img

# ==================== Calculating Image Gradient ====================

def ImageGradient(img):
    """
    Perform the Sobel operator. Obtain magnitude and direction
    of the edge map, which is the image gradient of input image
    i. The magnitude is rescaled to [0, 1] and direction reported
    in degrees. 
    """
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gx_img = conv2(img, gx)
    gy_img = conv2(img, gy)
    gxgy_img = gx_img + gy_img*1j
    mag_img = np.absolute(gxgy_img)/np.max(np.absolute(gxgy_img))
    theta_img = np.angle(gxgy_img, True)
    return mag_img, theta_img

# ==================== Selecting High and Low Thresholds ====================

def FindThreshold(Mag, percentageOfNonEdge):
    """
    Find the low and high thresholds in determining whether the
    pixel is an edge or not
    """
    hist, bins = np.histogram((Mag*255).ravel(),256,[0,256])
    T_high = np.percentile(Mag, percentageOfNonEdge)
    T_low  = T_high/2
    return T_high, T_low

# ==================== Supressing Nonmaxima ====================

def NonmaximaSupress(Mag, Theta):
    """
    Uses the theta edge map  and a look-up table method to perform 
    nonmaxima suppression on the input magnitude edge map.
    """
    Max = np.zeros(Mag.shape)
    LUT = {1:(1, 0), 2:(1, -1), 3:(0, -1), 4:(-1, -1),
           5:(-1, 0), 6:(-1, 1), 7:(0, 1), 0:(1, 1)}
    Theta = np.copy(Theta)
    Theta[Theta < 0] += 180
    Theta += 45/2
    dirs = Theta//45+1
    dirs[dirs==9] = 1
    height, width = Max.shape
    for (y, x) in product(range(height), range(width)):
        try:
            left = dirs[y, x]%8
            right = (dirs[y, x]+4)%8
            ldx, ldy = LUT[left]
            rdx, rdy = LUT[right]
            if Mag[y, x] >= Mag[y+ldy, x+ldx] and Mag[y, x] >= Mag[y+rdy, x+rdx]:
                Max[y, x] = Mag[y, x]
        except IndexError as e:
            pass
    return Max

# ==================== Thresholding and Edge Linking ====================

def EdgeLinking(Mag_low, Mag_high):
    """
    Obtain canny edge by linking the edges between Mag_low and 
    Mag_high. Mag_low and Mag_high are arrays whose values are binary,
    either 255 (edge) or 0 (not edge) have the datatype np.uint8.
    """
    # Gather label of pixels in Mag_low that also show up in Mag_high
    num_high, im_high = cv.connectedComponents(Mag_high)
    num_low, im_low = cv.connectedComponents(Mag_low)
    if (num_high == 0):
        return Mag_high
    low_labels_to_include = set()
    for curr_label in range(1, num_high):
        curr_pixels = np.argwhere(im_high==curr_label)
        curr_low_labels = im_low[curr_pixels.T[0,:], curr_pixels.T[1,:]]
        low_labels_to_include.update(curr_low_labels.tolist())
    low_labels_to_include = list(low_labels_to_include)

    # Create edge from labels gathered above
    low_pixels_to_include_row = np.array([], dtype=np.uint32)
    low_pixels_to_include_col = np.array([], dtype=np.uint32)
    for low_label in low_labels_to_include:
        curr_low_pixels = np.argwhere(im_low==low_label)
        low_pixels_to_include_row =  np.append(curr_low_pixels.T[0,:], 
                                               low_pixels_to_include_row)
        low_pixels_to_include_col =  np.append(curr_low_pixels.T[1,:],
                                               low_pixels_to_include_col)
    rows = low_pixels_to_include_row.tolist()
    cols = low_pixels_to_include_col.tolist()
    linked_image = np.zeros(Mag_high.shape, dtype=np.uint8)
    linked_image[rows, cols] = 255
    return linked_image

# ==================== Canny Edge Detection ====================

def canny_edge(img, kernal_size, sigma, percentageOfNonEdge):
    
    # Apply gaussian smoothing to input image img
    S = GaussSmoothing(img, kernal_size, sigma)

    # Apply Sobel's operator over the gaussian smoothed image
    M, T = ImageGradient(S)

    # Perform nonmaxima suppression on the magnitude of the gradient
    mag = NonmaximaSupress(M, T)

    # Compute the high and low thresholds
    T_high, T_low = FindThreshold(M, percentageOfNonEdge)

    # Apply high and low thresholds to magnitude of the gradient
    mag_high = np.copy(mag)
    mag_high[mag_high < T_high] = 0
    mag_high[mag_high != 0] = 255
    mag_low = np.copy(mag)
    mag_low[mag_low < T_low] = 0
    mag_low[mag_low != 0] = 255

    # Link all the edges of mag_high and mag_low
    linked_image = EdgeLinking(mag_low.astype(np.uint8), mag_high.astype(np.uint8))

    # Return canny edge image
    return linked_image

# ==================== Homework ====================

images = ["gun1.bmp", "joy1.bmp", "pointer1.bmp", "test1.bmp", "lena.bmp"]
for i in images:
    img = cv.imread(i, cv.IMREAD_GRAYSCALE)
    cv.imwrite(os.path.join("results" , i), canny_edge(img, 3, 1, 80))

img = cv.imread("lena.bmp", cv.IMREAD_GRAYSCALE)

cv.imwrite(os.path.join("results" , "edge-3-1-10.bmp"), canny_edge(img, 3, 1, 10))
cv.imwrite(os.path.join("results" , "edge-3-1-30.bmp"), canny_edge(img, 3, 1, 30))
cv.imwrite(os.path.join("results" , "edge-3-1-50.bmp"), canny_edge(img, 3, 1, 50))
cv.imwrite(os.path.join("results" , "edge-3-1-70.bmp"), canny_edge(img, 3, 1, 70))
cv.imwrite(os.path.join("results" , "edge-3-1-90.bmp"), canny_edge(img, 3, 1, 90))

cv.imwrite(os.path.join("results" , "edge-1-1-90.bmp"), canny_edge(img, 1, 1, 90))
cv.imwrite(os.path.join("results" , "edge-5-1-90.bmp"), canny_edge(img, 5, 1, 90))
cv.imwrite(os.path.join("results" , "edge-9-1-90.bmp"), canny_edge(img, 9, 1, 90))
cv.imwrite(os.path.join("results" , "edge-13-1-90.bmp"), canny_edge(img, 13, 1, 90))
cv.imwrite(os.path.join("results" , "edge-17-1-90.bmp"), canny_edge(img, 17, 1, 90))

cv.imwrite(os.path.join("results" , "edge-7-1-90.bmp"), canny_edge(img, 7, 1, 90))
cv.imwrite(os.path.join("results" , "edge-7-3-90.bmp"), canny_edge(img, 7, 3, 90))
cv.imwrite(os.path.join("results" , "edge-7-5-90.bmp"), canny_edge(img, 7, 5, 90))
cv.imwrite(os.path.join("results" , "edge-7-7-90.bmp"), canny_edge(img, 7, 7, 90))
cv.imwrite(os.path.join("results" , "edge-7-9-90.bmp"), canny_edge(img, 7, 9, 90))

# ==================== Test ====================

# img = cv.imread("lena.bmp", cv.IMREAD_GRAYSCALE)

# S = GaussSmoothing(img, 3, 1)

# M, T = ImageGradient(S)

# Mag = NonmaximaSupress(M, T)
# T_high, T_low = FindThreshold(M, 90)
# print(T_high, T_low)


# # Convert all pixels above the threshold to white pixels,
# # other wise black, and then compute the proportion of
# # pixels that are above the threshold
# Mag_copy = np.copy(Mag)
# Mag_copy2 = np.copy(Mag)
# Mag_copy[Mag_copy<T_high]=0
# Mag_copy[Mag_copy!=0]=255
# Mag_copy2[Mag_copy2<T_low]=0
# Mag_copy2[Mag_copy2!=0]=255
# # print(np.sum(Mag_copy==0)/(np.sum(Mag_copy==0)+np.sum(Mag_copy==255)))

# linked_image = EdgeLinking(Mag_copy2.astype(np.uint8), Mag_copy.astype(np.uint8))

# # edges = EdgeLinking(Mag_copy2.astype(np.uint8), Mag_copy.astype(np.uint8))
# # print(edges)

# cv.imshow("test", S.astype(np.uint8))
# cv.waitKey(0)
# cv.imshow("test", (M*255).astype(np.uint8))
# cv.waitKey(0)
# # cv.imshow("test", (T/360*255).astype(np.uint8))
# # cv.waitKey(0)
# cv.imshow("test", (Mag*255).astype(np.uint8))
# cv.waitKey(0)
# cv.imshow("test", Mag_copy.astype(np.uint8))
# cv.waitKey(0)
# # cv.imshow("test", Mag_copy2.astype(np.uint8))
# # cv.waitKey(0)
# cv.imshow("test", linked_image)
# cv.waitKey(0)
# cv.destroyAllWindows()