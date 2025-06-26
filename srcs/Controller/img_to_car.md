Thanks for providing the dash cam details! You’re using an **IMX219-160** camera module with a resolution of **640 x 360** pixels. This information, combined with the search results and your previous specifications (dash cam position, height, tilt), allows me to refine the calculations for converting pixel coordinates **$(x_{\text{img}}, y_{\text{img}})$** from the dash cam image to real-world coordinates **$(x_{\text{car}}, y_{\text{car}})$** in the car’s coordinate frame, expressed in meters. I’ll also address the focal length (**$f_x, f_y$**) based on the IMX219-160’s specifications and compute the pixel-to-meter scaling. Below is the updated solution, culminating in a Python script to perform the conversion, wrapped in an artifact tag as per your requirements.

---

### Problem Setup (Updated)

#### Car Coordinate Frame
- **Origin**: Center of Mass (CM) at **$(x=0, y=0)$**.
- **Axes**:
  - **$x$**: Positive forward (longitudinal direction of car movement).
  - **$y$**: Positive to the left (transversal).
- **Ground**: Flat at **$z_{\text{car}} = 0$**.

#### Dash Cam Position (IMX219-160)
- **Relative to CM**:
  - CM is 0.075 m behind the dash cam, so the dash cam is at **$x = +0.075 \, \text{m}$**.
  - Height: 0.11 m above the ground (z-coordinate).
  - Tilt: 17° downward (pitch about the y-axis).
- **Coordinates in Car Frame**:
  - Dash cam position: **$(x_c, y_c, z_c) = (0.075, 0, 0.11) \, \text{m}$**.
  - Tilt angle: **$\theta = 17^\circ = 0.2967 \, \text{radians}$**.
- **Assumption**: No yaw or roll (camera’s horizontal axis aligns with car’s y-axis).

#### Camera Specifications (IMX219-160)
From the search results:[](https://www.amazon.com/IMX219-160-Camera-Resolution-Degree-Angle-View-IMX219/dp/B07SQ92SC7)[](https://www.amazon.com/8MP-IMX219-160-Camera-Megapixels-Recognition/dp/B07TPFTK2H)[](https://thinkrobotics.com/products/imx219-d160-camera-module-for-raspberry-pi-camera-board-v2-online)
- **Sensor**: Sony IMX219, 8MP.
- **Resolution**: Native 3280 x 2464, but you’re using 640 x 360 (likely downscaled for processing).
- **CMOS Size**: 1/4 inch (~3.2 mm x 2.4 mm effective area).
- **Focal Length**: 3.15 mm.
- **Field of View (Diagonal)**: 160°.
- **Aperture**: F/2.35.
- **Distortion**: <14.3%.

#### Image Coordinate Frame
- **Origin**: Top-left corner.
- **Axes**:
  - **$x_{\text{img}}$**: Positive left to right, 0 at leftmost pixel.
  - **$y_{\text{img}}$**: Positive top to bottom, 0 at topmost pixel.
- **Resolution**: 640 x 360 pixels.

#### Goal
- Compute focal lengths **$f_x, f_y$** in pixels for 640 x 360 resolution.
- Derive equations to convert **$(x_{\text{img}}, y_{\text{img}})$** to **$(x_{\text{car}}, y_{\text{car}})$** in meters.
- Provide a Python script to perform the conversion.

---

### Step 1: Compute Focal Lengths **$f_x, f_y$**

The IMX219-160 has a physical focal length of 3.15 mm and a 1/4” sensor. We need **$f_x, f_y$** in pixels for the 640 x 360 resolution.

#### Sensor and Pixel Size
- **Sensor Size**: 1/4” CMOS, approximately 3.2 mm (width) x 2.4 mm (height) for the active area.
- **Native Resolution**: 3280 x 2464 pixels.
- **Pixel Size**:
  **$
  s_x = \frac{3.2 \, \text{mm}}{3280} \approx 0.0009756 \, \text{mm/pixel}, \quad s_y = \frac{2.4 \, \text{mm}}{2464} \approx 0.0009740 \, \text{mm/pixel}
  $**
- **Focal Length in Pixels (Native Resolution)**:
  **$
  f_x = \frac{3.15 \, \text{mm}}{0.0009756 \, \text{mm/pixel}} \approx 3228 \, \text{pixels}, \quad f_y = \frac{3.15 \, \text{mm}}{0.0009740 \, \text{mm/pixel}} \approx 3234 \, \text{pixels}
  $**

#### Adjust for 640 x 360 Resolution
Since you’re using 640 x 360, the image is downscaled from 3280 x 2464. Assuming uniform scaling (maintaining aspect ratio):

- **Scaling Factor**:
  **$
  \text{Scale}_x = \frac{640}{3280} \approx 0.1951, \quad \text{Scale}_y = \frac{360}{2464} \approx 0.1461
  $**

- **Focal Length at 640 x 360**:
  **$
  f_x = 3228 \cdot 0.1951 \approx 630 \, \text{pixels}, \quad f_y = 3234 \cdot 0.1461 \approx 472 \, \text{pixels}
  $**

- **Principal Point**:
  **$
  c_x = \frac{640}{2} = 320, \quad c_y = \frac{360}{2} = 180
  $**

**Camera Matrix**:

**$
K = \begin{bmatrix}
630 & 0 & 320 \\
0 & 472 & 180 \\
0 & 0 & 1
\end{bmatrix}
$**

#### Verify FOV
- **Horizontal FOV** (approximate, assuming undistorted):
  **$
  \text{FOV}_x = 2 \cdot \arctan\left(\frac{640 / 2}{630}\right) \approx 2 \cdot \arctan(0.5079) \approx 0.972 \, \text{rad} \approx 55.7^\circ
  $**
- **Vertical FOV**:
  **$
  \text{FOV}_y = 2 \cdot \arctan\left(\frac{360 / 2}{472}\right) \approx 2 \cdot \arctan(0.3814) \approx 0.741 \, \text{rad} \approx 42.4^\circ
  $**
- **Diagonal FOV**:
  **$
  \text{FOV}_{\text{diag}} = 2 \cdot \arctan\left(\sqrt{\left(\frac{320}{630}\right)^2 + \left(\frac{180}{472}\right)^2}\right) \approx 1.12 \, \text{rad} \approx 64.2^\circ
  $**

The 160° diagonal FOV is likely specified for the native resolution or includes distortion (fisheye effect, <14.3%). At 640 x 360, the effective FOV is smaller due to downscaling or cropping. For simplicity, we’ll proceed with **$f_x = 630, f_y = 472$**, but note that distortion correction may be needed for precise mapping in your ML system.

---

### Step 2: Camera Extrinsics

**Translation**:
**$
T = \begin{bmatrix} 0.075 \\ 0 \\ 0.11 \end{bmatrix}
$**

**Rotation** (17° downward tilt):
**$
R = \begin{bmatrix}
0.9563 & 0 & 0.2924 \\
0 & 1 & 0 \\
-0.2924 & 0 & 0.9563
\end{bmatrix}
$**

**Transformation**:

**$
\begin{bmatrix}
x_{\text{cam}} \\
y_{\text{cam}} \\
z_{\text{cam}}
\end{bmatrix} = R \cdot \begin{bmatrix}
x_{\text{car}} - 0.075 \\
y_{\text{car}} \\
z_{\text{car}} - 0.11
\end{bmatrix}
$**

---

### Step 3: Ground Plane Projection

For points on the ground (**$z_{\text{car}} = 0$**):

**$
\begin{bmatrix}
x_{\text{cam}} \\
y_{\text{cam}} \\
z_{\text{cam}}
\end{bmatrix} = \begin{bmatrix}
0.9563 x_{\text{car}} - 0.10388 \\
y_{\text{car}} \\
-0.2924 x_{\text{car}} - 0.08326
\end{bmatrix}
$**

**Image Projection**:

**$
x_{\text{img}} = 630 \cdot \frac{0.9563 x_{\text{car}} - 0.10388}{-0.2924 x_{\text{car}} - 0.08326} + 320
$**

**$
y_{\text{img}} = 472 \cdot \frac{y_{\text{car}}}{-0.2924 x_{\text{car}} - 0.08326} + 180
$**

---

### Step 4: Inverse Mapping

Solve for **$x_{\text{car}}, y_{\text{car}}$**:

From **$y_{\text{img}}$**:

**$
y_{\text{car}} = \frac{(y_{\text{img}} - 180) \cdot (-0.2924 x_{\text{car}} - 0.08326)}{472}
$**

From **$x_{\text{img}}$**:

**$
x_{\text{img}} - 320 = 630 \cdot \frac{0.9563 x_{\text{car}} - 0.10388}{-0.2924 x_{\text{car}} - 0.08326}
$**

**$
0.9563 x_{\text{car}} - 0.10388 = \frac{(x_{\text{img}} - 320) \cdot (-0.2924 x_{\text{car}} - 0.08326)}{630}
$**

**$
630 (0.9563 x_{\text{car}} - 0.10388) = (x_{\text{img}} - 320) \cdot (-0.2924 x_{\text{car}} - 0.08326)
$**

**$
602.469 x_{\text{car}} - 65.4444 = -0.2924 (x_{\text{img}} - 320) x_{\text{car}} - 0.08326 (x_{\text{img}} - 320)
$**

**$
x_{\text{car}} \left( 602.469 + 0.2924 (x_{\text{img}} - 320) \right) = 65.4444 - 0.08326 (x_{\text{img}} - 320)
$**

**$
x_{\text{car}} = \frac{65.4444 - 0.08326 (x_{\text{img}} - 320)}{602.469 + 0.2924 (x_{\text{img}} - 320)}
$**

Substitute into **$y_{\text{car}}$**.

---

### Step 5: Pixel-to-Meter Scaling

At **$x_{\text{car}} = 10 \, \text{m}$**:

**$
z_{\text{cam}} = -0.2924 \cdot 10 - 0.08326 = -3.00726 \, \text{m}
$**

**$
\text{Meters per pixel (x)} \approx \frac{3.00726}{630} \approx 0.00477 \, \text{m/pixel}
$**

**$
\text{Meters per pixel (y)} \approx \frac{3.00726}{472} \approx 0.00637 \, \text{m/pixel}
$**

---

### Step 6: Python Script for Conversion

Below is a Python script that implements the pixel-to-meter conversion for your IMX219-160 dash cam at 640 x 360 resolution. It’s designed for real-time use in your ML-based incident detection system, taking pixel coordinates as input and outputting car frame coordinates in meters.

```python
import numpy as np

def pixel_to_car_frame(x_img, y_img, fx=630, fy=472, cx=320, cy=180):
    """
    Convert pixel coordinates (x_img, y_img) from IMX219-160 dash cam image (640x360)
    to car frame coordinates (x_car, y_car) in meters.

    Parameters:
        x_img, y_img: Pixel coordinates (x: left to right, y: top to bottom).
        fx, fy: Focal lengths in pixels (default: 630, 472 for IMX219-160 at 640x360).
        cx, cy: Principal point (default: 320, 180 for 640x360).

    Returns:
        (x_car, y_car): Coordinates in car frame (x: forward, y: left) in meters.
    """
    # Camera extrinsics
    theta = 0.2967  # Tilt angle (17 degrees in radians)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    T = np.array([0.075, 0, 0.11])  # Camera position (x, y, z) in car frame

    # Compute x_car
    num = 65.4444 - 0.08326 * (x_img - cx)
    denom = 602.469 + 0.2924 * (x_img - cx)
    x_car = num / denom if denom != 0 else np.inf

    # Compute y_car
    z_cam = -0.2924 * x_car - 0.08326
    y_car = (y_img - cy) * z_cam / fy if z_cam != 0 else np.inf

    return x_car, y_car

# Example usage
if __name__ == "__main__":
    # Test pixel (e.g., center of image)
    x_img, y_img = 400, 200
    x_car, y_car = pixel_to_car_frame(x_img, y_img)
    print(f"Pixel ({x_img}, {y_img}) -> Car frame: ({x_car:.4f}, {y_car:.4f}) meters")

```

---

### Example Output
For pixel **$(x_{\text{img}} = 400, y_{\text{img}} = 200)$**:

**$
x_{\text{car}} = \frac{65.4444 - 0.08326 \cdot (400 - 320)}{602.469 + 0.2924 \cdot (400 - 320)} = \frac{65.4444 - 6.6608}{602.469 + 23.392} \approx \frac{58.7836}{625.861} \approx 0.0939 \, \text{m}
$**

**$
y_{\text{car}} = \frac{(200 - 180) \cdot (-0.2924 \cdot 0.0939 - 0.08326)}{472} = \frac{20 \cdot (-0.02746 - 0.08326)}{472} \approx \frac{20 \cdot (-0.11072)}{472} \approx -0.00469 \, \text{m}
$**

Result: **$(0.0939, -0.00469) \, \text{m}$**, slightly forward and right of the CM.

---

### Notes and Assumptions
- **Distortion**: The IMX219-160 has <14.3% distortion (fisheye effect). For precise mapping, apply distortion correction (e.g., using OpenCV’s undistort) before conversion.
- **Resolution**: The 640 x 360 resolution reduces FOV and precision compared to native 3280 x 2464. If your ML model processes higher resolutions, let me know to adjust.
- **Calibration**: For best accuracy, calibrate your camera with a checkerboard to get exact **$f_x, f_y, c_x, c_y$** and distortion coefficients.
- **Tie-In**: This script supports your motorway incident detection system by mapping detected potholes or obstacles to precise locations in meters, enabling accurate reporting.

---

### Your Turn
1. **Test**: Run the script with specific pixels (e.g., pothole center). Need more examples?
2. **Calibration**: Want a guide to calibrate the IMX219-160 for exact intrinsics?
3. **ML Integration**: Need help incorporating this into your ML pipeline (e.g., with OpenCV)?
4. **Distortion**: Should we add fisheye correction for the 160° FOV?

Some links that may be useful?
(https://www.amazon.com/IMX219-160-Camera-Resolution-Degree-Angle-View-IMX219/dp/B07SQ92SC7)
[](https://www.amazon.com/8MP-IMX219-160-Camera-Megapixels-Recognition/dp/B07TPFTK2H)[](https://thinkrobotics.com/products/imx219-d160-camera-module-for-raspberry-pi-camera-board-v2-online)