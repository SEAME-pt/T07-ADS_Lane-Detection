# Lane Detection and Steering Angle Estimation - OpenCV


## Overview

This project is designed to detect lane markings in a video stream, calculate the curvature of the lanes, estimate the vehicle's offset from the center, and determine the steering angle required for proper alignment. Utilizing computer vision techniques provided by OpenCV and numerical operations via NumPy, the code processes each video frame through several stages to output a visual overlay of detected lanes with critical driving metrics.

## Detailed Explanation

### 1. Video Capture and Preprocessing
- Video Input: The script starts by capturing frames from a video file (specified as ../output.mp4) using OpenCV's VideoCapture.
- Frame Resizing: Each captured frame is resized to 800x600 pixels to maintain uniform processing dimensions.
- Perspective Points: Four key points (top-left, bottom-left, top-right, and bottom-right) are defined to create a perspective transformation matrix, allowing the transformation of the frame into a bird’s-eye view.

### 2. Trackbars for Dynamic Thresholding
- Interactive Adjustment: Trackbars are set up to adjust HSV threshold values in real-time. This enables fine-tuning of the lower and upper bounds for color segmentation, which is crucial for robust lane detection under varying lighting conditions.

### 3. Perspective Transformation and Data Augmentation
- Perspective Transformation: Using the predefined points, a transformation matrix is computed with cv2.getPerspectiveTransform, and the frame is warped into a top-down view.
- Data Augmentation: Brightness and contrast adjustments are applied via cv2.convertScaleAbs. Furthermore, the V channel of the HSV representation is enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve the frame's overall contrast.

### 4. Thresholding and Mask Generation\n- HSV Conversion: The augmented frame is converted to the HSV color space.
- Mask Creation: A binary mask is generated using cv2.inRange based on dynamically adjustable HSV threshold values from the trackbars, highlighting potential lane markings.
- Noise Reduction: Gaussian blur, dilation, and erosion are applied to the mask to smooth out noise and close gaps in the detected regions.

### 5. Lane Detection with Sliding Windows
- Histogram Analysis: The lower half of the mask is analyzed to create a histogram that helps in identifying the base positions of the left and right lane lines.
- Sliding Window Search: A sliding window approach is employed to search for lane line pixels vertically through the frame. This method dynamically adjusts the search window based on previously detected positions to ensure consistent tracking.
- Contour Detection and Validation: Contours are detected within each window and filtered based on area. The centroids of valid contours are then used as lane points, with checks to ensure they do not deviate significantly from previous positions.

### 6. Polynomial Fitting and Lane Parameter Calculation
- Curve Fitting: Lane points are fitted with a second-degree polynomial using np.polyfit, which models the curvature of the lane lines.
- Lane Metrics:
- Curvature: The radius of curvature for both lanes is computed and averaged to give a single curvature metric.
- Offset Calculation: The vehicle's lateral offset is determined by comparing the lane center to a predefined car position.
- Steering Angle: Using the computed offset and curvature, the necessary steering angle is calculated to adjust the vehicle's trajectory.

### 7. Visualization and Overlay
- Lane Overlay: The detected lane area is mapped back onto the original frame using an inverse perspective transformation. A semi-transparent overlay highlights the lane area on the original view.
- Metric Display: Key metrics such as offset and error angle are annotated on the result frame.
- Display Windows: Multiple windows display the original frame, the sliding window mask, and the final result with overlays, providing real-time visual feedback.

## Dependencies
- OpenCV: Used for video capture, image processing, transformations, and display.
- NumPy: Used for numerical computations and polynomial fitting.

## How to Run
1. Install Dependencies:\n bash\n pip install opencv-python numpy

2. Prepare Video File: Ensure the video file (e.g., output.mp4) is placed in the correct directory (../output.mp4).
3. Execute the Script:
 bash
 python your_script_name.py
4. Adjust Trackbars: Use the trackbars in the displayed window to fine-tune HSV threshold values if needed.



Developed by: Team07 - SEA:ME Portugal 

