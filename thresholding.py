import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
import skimage.filters as filters

def kde_statsmodels_u(x, x_grid, bandwidth, **kwargs):
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

def rosin(heatmap, maxPercent = 5):
    heatmap_list = heatmap.flatten().tolist()
    new_data = np.array(heatmap_list)
    #new_data = heatmap.flatten()
    #new_data = f_heatmap - np.min(f_heatmap) + 0.001
    # declare kernel estimation parameters
    bandwidth = 0.06
    # estimate kernel
    x_grid = np.linspace(0, np.max(new_data), 90)  # x-coordinates for data points in the kernel
    kernel = kde_statsmodels_u(new_data, x_grid, bandwidth)  # get kernel

    # get the index of the kernal peak
    maxIndex = np.argmax(kernel)

    # Assign percent below the max kernel value for the 'zero' peak i.e. a value of 2 = 2% the maximum value
    #maxPercent = 5

    # assign x and y coords for peak-to-base line
    x1 = x_grid[maxIndex]
    y1 = kernel[maxIndex]
    # find all local minima in the kernel
    local_mins = np.where(np.r_[True, kernel[1:] < kernel[:-1]] & np.r_[kernel[:-1] < kernel[1:], True])
    local_mins = local_mins[0]  # un 'tuple' local mins
    # filter for points below a certain kernel max
    local_mins = local_mins[(np.where(kernel[local_mins] < (y1 / (100 / maxPercent))))]
    # get local minima beyond the peak
    local_mins = local_mins[(np.where(local_mins > maxIndex))]  # get local minima that meet percent max threshold
    x2_index = local_mins[0]  # find minumum beyond peak of kernel
    x2 = x_grid[x2_index]  # index to local min beyond kernel peak
    y2 = kernel[x2_index]

    # calculate line slope and get perpendicular line
    slope = (y2 - y1) / (x2 - x1)
    # find y_intercept for line
    y_int = y1 - (slope * x1)
    slopeTan = -1 / slope  # perpendicular line slope

    # allocate lists for x-y coordinates and distance values
    dist = list()
    # save x-y coords of intersect points
    yii = list()
    xii = list()

    # iterate and generate perpendicular lines
    for i in range(maxIndex + 1, x2_index):
        # find intersection point between lines
        # determine equation of the perpendicular line based on current bin coordinate
        xt1 = x_grid[i]
        yt1 = kernel[i]
        y_int_tan = yt1 - (slopeTan * xt1)
        # calculate intersection point between lines
        b1 = y_int
        b2 = y_int_tan
        m1 = slope
        m2 = slopeTan
        # y = mx + b
        # Set both lines equal to find the intersection point in the x direction, y1=y2, x1=x2
        # y1 = m1 * x + b1, y2 = m2 * x + b2
        # if y1 == y2...
        # m1 * x + b1 = m2 * x + b2
        # m1 * x - m2 * x = b2 - b1
        # x * (m1 - m2) = b2 - b1
        # x = (b2 - b1) / (m1 - m2)
        xi = (b2 - b1) / (m1 - m2)
        # Now solve for y -- use either line, because they are equal here
        # y = mx + b
        yi = m1 * xi + b1
        # assert that the new line generated is equal or very close to the correct perpendicular value of the max deviation line
        assert ((m2 - m2 * .01) < ((yi - y_int_tan) / (xi - 0)) < (
                    m2 + m2 * .01))  # an error will throw if this statement is false
        # save x-y coordinates of the point
        yii.append(yi)
        xii.append(xi)
        # get euclidean distance between kernel coordinate and intersect point
        euc = np.sqrt((xi - xt1) ** 2 + (yi - yt1) ** 2)
        # store the euclidean distance
        dist.append(euc)

    # get kernel point with the maximum distance from the Rosin line
    # remeber, we started at maxIndex+1, so the index of the optimalPoint in the kernel array will be maxIndex+1
    # + the index in the 'dist' array
    optimalPoint = np.argmax(dist) + maxIndex + 1
    # plot the optimal point over the kernel with Rosin line we plotted before
    threshold = x_grid[optimalPoint]
    #final_threhold = threshold + np.min(f_heatmap)
    #return heatmap < final_threhold
    return threshold


def threshold_rosin2(heatmap):
    try:
        val4 = rosin(heatmap, maxPercent=15)
    except:
        try:
            val4 = rosin(heatmap, maxPercent=20)
        except:
            try:
                val4 = rosin(heatmap, maxPercent=25)
            except:
                try:
                    val4 = rosin(heatmap, maxPercent=30)
                except:
                    try:
                        val4 = rosin(heatmap, maxPercent=35)
                    except:
                        val4 = filters.threshold_otsu(heatmap)

    return heatmap > val4


def threshold_rosin3(heatmap):
    try:
        val4 = rosin(heatmap, maxPercent=15)
    except:
        try:
            val4 = rosin(heatmap, maxPercent=20)
        except:
            try:
                val4 = rosin(heatmap, maxPercent=25)
            except:
                try:
                    val4 = rosin(heatmap, maxPercent=30)
                except:
                    try:
                        val4 = rosin(heatmap, maxPercent=35)
                    except:
                        val4 = filters.threshold_otsu(heatmap)

    return val4

def threshold_rosin(heatmap):

    val4 = filters.threshold_otsu(heatmap)

    return heatmap > val4

def threshold_rosin4(heatmap):

    val4 = filters.threshold_otsu(heatmap)

    return val4