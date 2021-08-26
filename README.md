# Implementation Description

My implementation of the Canny edge detection algorithm can be separated into five general steps: (1) smoothing the input image with a Gaussian filter, (2) applying Sobel’s operator the smoothened image to find the its gradient, (3) performing non-maximum suppression to highlight the edges, (4) using the gradient magnitude to find appropriate threshold values, and (5) linking the filtered edges. The next few paragraphs further elaborate each of the five steps.

The implementation of the Gaussian smoothing is relatively simple. The function gaussian_kernel creates a Gaussian kernel, which is then convoluted (via SciPy’s convolve2d function) over the input image to smoothen the sharp edges.

Sobel’s operator is then convolved over the smoothened image from part (1) to retrieve the image’s gradient. Two arrays are obtained as the result of this operation: first is the magnitude of the gradient and second is the direction of the gradient. The use of complex numbers allows for an easier implementation of Sobel’s operator; the horizontal axis is treated as the real axis while the vertical axis is treated as the imaginal axis. NumPy’s absolute and angle methods are then used to compute the magnitude and direction arrays of the gradient respectively.

Next, we try to highlight the edges by performing non-maximum suppression. This operation essentially eliminates the soft region of the edges, leaving behind finer, brighter edges. The operation works by utilizing the direction gradient obtained from part (3) and calculating the pixel’s neighbors in the direction of the gradient. If the neighbors are dimmer than the current pixel, then we have a maximum.

To further emphasize on the bright edges, we perform thresholding. Any edges whose pixel intensities are less than a certain threshold value will be ignored (converted to background). Two thresholds are used, one low and one high. The high threshold is determined by selecting a value such that the proportion of non-edges in the image corresponds to percentageOfNonEdge. The low threshold is half that of the high threshold. The low threshold is used to find relatively noisier edges and high threshold for cleaner edges.

The two types of edges found in part (4) are then linked via the EdgeLinking function implemented in this section. The edge linking algorithm works by iterating through every connected component in the clear edges (those obtained by using a high threshold) and whenever we reach the end an edge, we search for the same edge in the noisier image in hope of being able to connect the edges. This step is repeated until all edges in the cleaner image are processed.

# Performance

To view these figures, see ``figures.pdf``.
 
The algorithm was then tested on a few images shown in Figure 1 and 2. The four images in Figure 1 are computed using a Gaussian kernel size of 3x3, a Gaussian standard deviation of 1, and a percentage of non-edge value of 90%.

Figure 2 attempts to show how different parameters can affect the detected Canny edge. The three parameters we can change are Gaussian kernel size, Gaussian standard deviation, and the percentage of non-edge. From Figure 2, we observe a general trend among all three parameters. As the value of the parameter increases, we see less edges being detected. This is most noticeable with the percentageOfNonEdge parameter; there is a major difference between when 10% is used versus when 90% is used.

Finally, Figure 3 attempts to compare how well Canny edge performs against its counterparts, such as those obtained by using Sobel’s, Roberts’, and zero-crossing methods. We observe that the Canny edge detection is able to capture the most important edges, being able to eliminate noise (non-significant edges) very well, while also retaining the edges. The zero-cross method has a relatively large amount of noise, and both the Sobel and Roberts method are not able to capture a big portion of the edges.

