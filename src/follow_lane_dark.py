#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from self_driving_turtlebot3.msg import Lane
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

img = np.zeros_like((480, 640, 3))
warped_size = (640, 480)
num_rows = warped_size[1]

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to threshold an image for a given direction range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def get_thresholded_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[0], gray.shape[1]

    # b channel threshold can select lane lines out
    color_threshold = 10
    B = img[:,:,0]
    b_condition = B < color_threshold
    b_condition = b_condition*1

    # apply the region of interest mask
    mask = np.zeros_like(gray)
    region_of_interest_vertices = np.array([[0,height-1], [0, int(0.65*height)], 
                                            [width-1, int(0.65*height)],
                                            [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(b_condition, b_condition, mask=mask)
    
    return thresholded

# Vertices extracted manually for performing a perspective transform
bottom_left = [105, 480]
bottom_right = [530, 480]
top_left = [205, 300]
top_right = [435, 300]

source = np.float32([bottom_left,bottom_right,top_right,top_left])

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [200, 480]
bottom_right = [440, 480]
top_left = [200, 1]
top_right = [440, 1]

dst = np.float32([bottom_left,bottom_right,top_right,top_left])
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
warp_size = (640, 480)

def measure_curvature(x_values):
    ym_per_pix = 15.0/48000 # meters per pixel in y dimension
    xm_per_pix = 17.5/24000 # meters per pixel in x dimension
    # If no pixels were found return None
    y_points = np.linspace(0, num_rows-1, num_rows)
    y_eval = np.max(y_points)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit((y_points*ym_per_pix), (x_values*xm_per_pix), 3)
    curverad = ((1 + (3*fit_cr[0]*(y_eval*ym_per_pix)**2 + 2*fit_cr[1]*y_eval*ym_per_pix+
                fit_cr[2])**2)**1.5) / np.absolute(6*fit_cr[0]*y_eval*ym_per_pix+2*fit_cr[1])
    
    # add correction factor
    curverad *= 3
    
    return curverad

# Some global variables
polyfit_left=None
polyfit_right=None

past_good_left_lines = []
past_good_right_lines = []

running_mean_difference_between_lines = 0

def get_line_predictions(non_zeros_x, non_zeros_y, left_coordinates, right_coordinates, num_rows):
    """
        Given ncoordinates of non-zeros pixels and coordinates of non-zeros pixels within the sliding windows,
        this function generates a prediction for the lane line.
    """
    left_x = non_zeros_x[left_coordinates]
    left_y = non_zeros_y[left_coordinates]
    
    # If no pixels were found return None
    if(left_y.size == 0 or left_x.size == 0):
        return None, None

    # Fit the polynomial
    polyfit_left = np.polyfit(left_y, left_x, 3)

    right_x = non_zeros_x[right_coordinates]
    right_y = non_zeros_y[right_coordinates]
    
    # If no pixels were found return None
    if(right_y.size == 0 or right_x.size == 0):
        return None, None

    # Fit the polynomial
    polyfit_right = np.polyfit(right_y, right_x, 3)

    # If no pixels were found return None
    y_points = np.linspace(0, num_rows-1, num_rows)
    
    # Generate the lane lines from the polynomial fit
    left_x_predictions = polyfit_left[0]*y_points**3 + polyfit_left[1]*y_points**2 + polyfit_left[2]*y_points + polyfit_left[3]
    right_x_predictions = polyfit_right[0]*y_points**3 + polyfit_right[1]*y_points**2 + polyfit_right[2]*y_points + polyfit_right[3]
    
    return left_x_predictions, right_x_predictions

def brute_search(warped):
    """
        This function searches for lane lines from scratch.
        Thresholding & performing a sliding window search.
    """
    non_zeros = warped.nonzero()
    non_zeros_y = non_zeros[0]
    non_zeros_x = non_zeros[1]
    
    num_rows = warped.shape[0]
    
    histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)

    half_width = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:half_width])
    rightx_base = np.argmax(histogram[half_width:]) + half_width

    num_windows = 10
    window_height = np.int(num_rows/num_windows)
    window_half_width = 50

    min_pixels = 100
    
    left_coordinates = []
    right_coordinates = []

    for window in range(num_windows):
        y_max = num_rows - window*window_height
        y_min = num_rows - (window+1)* window_height

        left_x_min = leftx_base - window_half_width
        left_x_max = leftx_base + window_half_width

        good_left_window_coordinates = ((non_zeros_x >= left_x_min) & (non_zeros_x <= left_x_max) & (non_zeros_y >= y_min) & (non_zeros_y <= y_max)).nonzero()[0]
        left_coordinates.append(good_left_window_coordinates)

        if len(good_left_window_coordinates) > min_pixels:
            leftx_base = np.int(np.mean(non_zeros_x[good_left_window_coordinates]))

        right_x_min = rightx_base - window_half_width
        right_x_max = rightx_base + window_half_width

        good_right_window_coordinates = ((non_zeros_x >= right_x_min) & (non_zeros_x <= right_x_max) & (non_zeros_y >= y_min) & (non_zeros_y <= y_max)).nonzero()[0]
        right_coordinates.append(good_right_window_coordinates)

        if len(good_right_window_coordinates) > min_pixels:
            rightx_base = np.int(np.mean(non_zeros_x[good_right_window_coordinates]))

    left_coordinates = np.concatenate(left_coordinates)
    right_coordinates = np.concatenate(right_coordinates)
    
    left_x_predictions, right_x_predictions = get_line_predictions(non_zeros_x, non_zeros_y, left_coordinates, right_coordinates, num_rows)
    return left_x_predictions, right_x_predictions, left_coordinates, right_coordinates

def get_averaged_line(previous_lines, new_line):
    """
        This function computes an averaged lane line by averaging over previous good frames.
    """
    
    # Number of frames to average over
    num_frames = 4
    
    if new_line is None:
        # No line was detected
        
        if len(previous_lines) == 0:
            # If there are no previous lines, return None
            return previous_lines, None
        else:
            # Else return the last line
            return previous_lines, previous_lines[-1]
    else:
        if len(previous_lines) < num_frames:
            # we need at least num_frames frames to average over
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # average over the last num_frames frames
            previous_lines[0:num_frames-1] = previous_lines[1:]
            previous_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            return previous_lines, new_line
        
        
def get_mean_distance_between_lines(left_line, right_line, running_average):
    """
        Returns running weighted average of simple difference between left and right lines
    """
    mean_distance = np.mean(right_line - left_line)
    if running_average == 0:
        running_average = mean_distance
    else:
        running_average = 0.7*running_average + 0.3*mean_distance
    return running_average
    

def pipeline_final(img):
    # global variables to store the polynomial coefficients of the line detected in the last frame
    global polyfit_right
    global polyfit_left
    
    # global variables to store the line coordinates in previous n (=4) frames
    global past_good_right_lines
    global past_good_left_lines
    
    # global variable which contains running average of the mean difference between left and right lanes
    global running_mean_difference_between_lines
    
    warp_size = (640, 480)
    
    # get thresholded image
    thresholded = get_thresholded_image(img)
    
    # perform a perspective transform
    warped = cv2.warpPerspective(thresholded, M, warped_size , flags=cv2.INTER_NEAREST)
    
    out_img = np.uint8(np.dstack((warped, warped, warped))*255)
    out_img_2 = np.uint8(np.dstack((warped, warped, warped))*255)
    
    non_zeros = warped.nonzero()
    non_zeros_y = non_zeros[0]
    non_zeros_x = non_zeros[1]
    
    num_rows = warped.shape[0]
    y_points = np.linspace(0, num_rows-1, num_rows)
    
    if (polyfit_left is None) or (polyfit_right is None):
        # If the polynomial coefficients of the previous frames are None then perform a brute force search
        brute = True
        left_x_predictions, right_x_predictions, left_coordinates, right_coordinates = brute_search(warped)
    else:
        # Else search in a margin of 100 pixels on each side of the pervious polynomial fit
        brute = False
        margin = 100
        left_x_predictions = polyfit_left[0]*non_zeros_y**3 + polyfit_left[1]*non_zeros_y**2 + polyfit_left[2]*non_zeros_y + polyfit_left[3]
        left_coordinates = ((non_zeros_x >= left_x_predictions - margin) & (non_zeros_x <= left_x_predictions + margin)).nonzero()[0]

        right_x_predictions = polyfit_right[0]*non_zeros_y**3 + polyfit_right[1]*non_zeros_y**2 + polyfit_right[2]*non_zeros_y + polyfit_right[3]
        right_coordinates = ((non_zeros_x >= right_x_predictions - margin) & (non_zeros_x <= right_x_predictions + margin)).nonzero()[0]
        
        left_x_predictions, right_x_predictions = get_line_predictions(non_zeros_x, non_zeros_y, left_coordinates, right_coordinates, num_rows)
    
    if (left_x_predictions is None or right_x_predictions is None):
        if not brute:
            left_x_predictions, right_x_predictions, left_coordinates, right_coordinates = brute_search(warped)
            
    bad_lines = False
            
    if (left_x_predictions is None or right_x_predictions is None):
        bad_lines = True
    else:
        mean_difference = np.mean(right_x_predictions - left_x_predictions)
        
        if running_mean_difference_between_lines == 0:
            running_mean_difference_between_lines = mean_difference
        
        if (mean_difference < 0.7*running_mean_difference_between_lines or mean_difference > 1.3*running_mean_difference_between_lines):
            bad_lines = True
            if not brute:
                left_x_predictions, right_x_predictions, left_coordinates, right_coordinates = brute_search(warped)
                if (left_x_predictions is None or right_x_predictions is None):
                    bad_lines = True
                else:
                    mean_difference = np.mean(right_x_predictions - left_x_predictions)
                    if (mean_difference < 0.7*running_mean_difference_between_lines or mean_difference > 1.3*running_mean_difference_between_lines):
                        bad_lines = True
                    else:
                        bad_lines = False
        else:
            bad_lines = False
            
    if bad_lines:
        polyfit_left = None
        polyfit_right = None
        if len(past_good_left_lines) == 0 and len(past_good_right_lines) == 0:
            return img
        else:
            left_x_predictions = past_good_left_lines[-1]
            right_x_predictions = past_good_right_lines[-1]
    else:
        past_good_left_lines, left_x_predictions = get_averaged_line(past_good_left_lines, left_x_predictions)
        past_good_right_lines, right_x_predictions = get_averaged_line(past_good_right_lines, right_x_predictions)
        mean_difference = np.mean(right_x_predictions - left_x_predictions)
        running_mean_difference_between_lines = 0.9*running_mean_difference_between_lines + 0.1*mean_difference

    left_line_window = np.array(np.transpose(np.vstack([left_x_predictions, y_points])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions, y_points]))))
    
    # color the detected lane
    out_img_2[non_zeros_y[left_coordinates], non_zeros_x[left_coordinates]] = [255, 0, 0]
    out_img_2[non_zeros_y[right_coordinates], non_zeros_x[right_coordinates]] = [0, 0, 255]
    
    # judge which side of lane get detected and thus calculate the deviation
    xm_per_pix = 17.5/32000 # meters per pixel in x dimension
    
    if sum(left_coordinates) > 2000 and sum(right_coordinates) > 2000:
        left_curve_rad = measure_curvature(left_x_predictions)
        right_curve_rad = measure_curvature(right_x_predictions)
        average_curve_rad = (left_curve_rad + right_curve_rad)/2
        lane_center = (right_x_predictions[num_rows-1] + left_x_predictions[num_rows-1])/2
        
    elif sum(left_coordinates) <= 2000: # only right lane get detected
        right_line_fit = np.polyfit(y_points, right_x_predictions, 1)
        if right_line_fit[0] > 0: # judge the side of lane by its gradient's sign
            offset = -175*(1 + right_line_fit[0]**2)**0.5
        else:
            offset = 175*(1 + right_line_fit[0]**2)**0.5
        average_curve_rad = measure_curvature(right_x_predictions)
        lane_center = right_x_predictions[num_rows-1] + offset
    
    else: # only left lane get detected
        left_line_fit = np.polyfit(y_points, left_x_predictions, 1)
        if left_line_fit[0] > 0: # judge the side of lane by its gradient's sign
            offset = -175*(1 + left_line_fit[0]**2)**0.5
        else:
            offset = 175*(1 + left_line_fit[0]**2)**0.5
        average_curve_rad = measure_curvature(left_x_predictions)
        lane_center = left_x_predictions[num_rows-1] + offset
    
    curvature_string = "Curvature: %.2f m" % average_curve_rad
    
    center_offset_pixels = lane_center - warped_size[0]/2
    center_offset_mtrs = xm_per_pix*center_offset_pixels
    offset_string = "Deviation: %.2f m" % center_offset_mtrs
    
    # mark the area of lanes
    poly_points = np.vstack([left_line_window, right_line_window])
    cv2.fillPoly(out_img, np.int_([poly_points]), [0,255, 0])
    unwarped = cv2.warpPerspective(out_img, M_inv, (640, 480) , flags=cv2.INTER_NEAREST)
    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)  
    
    # add bird-view lanes to the output image
    road = cv2.resize(out_img_2, (200, 120))
    result[:road.shape[0], :road.shape[1], :] = road
    
    # annotate the road condition to the output image
    cv2.putText(result,curvature_string , (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)
    cv2.putText(result, offset_string, (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)
    
    return result, center_offset_mtrs, average_curve_rad



def publish_lane(image, args):
	track_lane_pub = args[0]
	lane_condition_pub = args[1]
	cv2_img = bridge.imgmsg_to_cv2(image, "bgr8")

	rospy.loginfo("image received")

	draw, deviation, curvature = pipeline_final(cv2_img)
	drawmsg = bridge.cv2_to_imgmsg(draw, "bgr8")
	lane = Lane()
	lane.deviation = deviation
	lane.curvature = curvature

	track_lane_pub.publish(drawmsg)
	lane_condition_pub.publish(lane)

	return

def get_lane():
	rospy.init_node('get_lane')

	image_topic = "/raw_image"

	track_lane_pub = rospy.Publisher("lane_finding", Image, queue_size=1)
	lane_condition_pub = rospy.Publisher("lane_condition", Lane, queue_size=1)
	rospy.Subscriber(image_topic, Image, publish_lane, callback_args=(track_lane_pub, lane_condition_pub))

	rospy.spin()

if __name__ == '__main__':
	get_lane()
