import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import interact, interactive, fixed
from cam_calib import ret, mtx, dist, rvecs, tvecs

y_ratio_cvt = np.float32(512/720)
x_ratio_cvt = np.float32(640/1080)

src = np.float32([
    [200*x_ratio_cvt, 430*y_ratio_cvt],
    [522*x_ratio_cvt, 200*y_ratio_cvt], 
    [630*x_ratio_cvt, 200*y_ratio_cvt], 
    [950*x_ratio_cvt, 430*y_ratio_cvt]
])
# Points for the new image
dst = np.float32([
    [300*x_ratio_cvt, 720*y_ratio_cvt],
    [300*x_ratio_cvt, 0*y_ratio_cvt], 
    [(1080-300)*x_ratio_cvt, 0*y_ratio_cvt], 
    [(1080-300)*x_ratio_cvt, 720*y_ratio_cvt]
])


ym_per_pix = 30/720 # meters per pixel in y dimension
# xm_per_pix = 3.7/700 # meters per pixel in x dimension

# ym_per_pix = 3.0/100 # meters per pixel in y dimension, lane line is 10 ft = 3 meters
xm_per_pix = 3.7/550 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters


def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def warper(img):
    
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarp(img):
    
    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    
    return unwarped

def applyThreshold(channel, thresh):
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(channel)
    
    # Inclusive thresholds 
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    
    # 3) Return a binary image of threshold result
    return binary_output

def rgb_rthresh(img, thresh=(125, 255)):
    # Pull out the R channelc
    channel = img[:,:,0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

# and applies a threshold
def calc_sobel(img, sobel_kernel=25, mag_thresh=(25, 255)):
    
    # Convert to grayscale - sobel can only have one color channel
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]
    
    # Take the sobel gradient in x and y
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(L, cv2.CV_64F, 0, 1)
    
    # Calculate the magnitude 
#     mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
#     scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
    
    # Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return sxbinary

def binaryPipe(img):
    
    # Copy the image
    img = np.copy(img)

    # Undistort the image based on the camera calibration
    undist = undistort(img, mtx, dist)

    # warp the image based on our perspective transform
    warped = warper(undist)
    
    # Get the Red and saturation images
    red = rgb_rthresh(warped, thresh=(225, 255))
    # sat = hls_sthresh(warped, thresh=(175, 255))
    
    # Run the sobel magnitude calculation
    sobel = calc_sobel(warped, sobel_kernel=15,  mag_thresh=(50, 220))
    
    # combine these layers
    combined_binary = np.zeros_like(sobel)
    combined_binary[ (red == 1) | sobel == 1] = 1 #(sat == 1) |
    
    return combined_binary

def calc_line_fits(img):

    ### Settings
    # Choose the number of sliding windows (for these images, the height is 450)
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int32(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img

def calc_radius(y_eval, left_fit_m, right_fit_m):
    """
    left_fit and right_fit are assumed to have already been converted to meters
    """
    y_eval = y_eval * ym_per_pix
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    right_curverad = ((1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
    
    return left_curverad, right_curverad

def calc_center_dist(y_eval, midpoint, left_fit, right_fit):
        
    # Calculate the distance from the center
    x_left_line = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_line = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
    
    position_from_center = ((x_left_line + x_right_line)/2 - midpoint) * xm_per_pix
    
    return position_from_center

def lane_detect(img):

    # Run this through our binary pipeline
    binary_warped = binaryPipe(img)

    # Run the warped, binary image from the pipeline through the fitter
    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = binary_warped.shape[0]
    midx = binary_warped.shape[1] / 2

    # calc the radius and center distance
    left_curverad, right_curverad = calc_radius(y_eval, left_fit_m, right_fit_m)
    center = calc_center_dist(y_eval, midx, left_fit, right_fit)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)

    curve_rad = (left_curverad + right_curverad) / 2.

    return newwarp, curve_rad, center


cap = cv2.VideoCapture(1)
assert cap.isOpened(), 'Failed to open'

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (640,512))
    img_og = np.copy(frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# img = mpimg.imread('test_vn2.jpg')

    newwarp, curve_rad, center = lane_detect(img)

    # Combine the result with the original image
    result = cv2.addWeighted(img_og, 1, newwarp, 0.5, 0)
    
    # Add the radius and center
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.0f}'.format(curve_rad) + 'm'
    cv2.putText(result, text, (50,50), font, 0.7, (0,255, 0), 1, cv2.LINE_AA)
    if center < 0:
        dir = 'left'
    else:
        dir = 'right'
    text = '{:03.2f}'.format(center) + 'm '+ dir + ' of center'
    cv2.putText(result, text, (50,70), font, 0.7, (0,255, 0), 1, cv2.LINE_AA,)

    # result = cv2.resize(result, (1080,720))
    cv2.imshow('image',result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()