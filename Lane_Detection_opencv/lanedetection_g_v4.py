import cv2
import numpy as np
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=np.RankWarning)

vidcap = cv2.VideoCapture("../../outputcarla.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

class LaneDetector:
    def __init__(self):
        self.prev_left_points = deque(maxlen=5)
        self.prev_right_points = deque(maxlen=5)
        self.left_miss_count = 0
        self.right_miss_count = 0
        self.LANE_RESET_THRESHOLD = 15
        self.MIN_LANE_POINTS = 5

        self.kalman_left = cv2.KalmanFilter(4, 2)
        self.kalman_left.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_left.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                      [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman_left.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
        self.kalman_left.statePost = np.array([400, 400, 0, 0], dtype=np.float32)

        self.kalman_right = cv2.KalmanFilter(4, 2)
        self.kalman_right.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_right.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman_right.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
        self.kalman_right.statePost = np.array([400, 400, 0, 0], dtype=np.float32)

    def adaptive_masks(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v_eq))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, _, b = cv2.split(lab)
        b = clahe.apply(b)
        _, yellow_mask = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)

        white_mask = cv2.inRange(hsv_eq, (0, 0, 200), (255, 50, 255))

        return yellow_mask, white_mask

    def detect_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def compute_metrics(self, left_fit, right_fit, img_height, car_x=400):
        y_eval = img_height - 1
        plot_y = np.linspace(0, y_eval, num=img_height)
        left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        center_x = (left_x + right_x) / 2

        left_a, right_a = 2 * left_fit[0], 2 * right_fit[0]
        left_curvature = ((1 + (left_a * y_eval + left_fit[1]) ** 2) ** 1.5) / max(abs(left_a), 1e-5)
        right_curvature = ((1 + (right_a * y_eval + right_fit[1]) ** 2) ** 1.5) / max(abs(right_a), 1e-5)
        curvature = (left_curvature + right_curvature) / 2

        lane_center = center_x[-1]
        lane_offset = (car_x - lane_center) * 0.26 / 800
        center_fit = np.polyfit(plot_y, center_x, 2)
        slope = 2 * center_fit[0] * y_eval + center_fit[1]
        steering_angle = np.arctan(-slope) * 180 / np.pi
        
        car_y = 599
        A, B, C = center_fit
        slope1 = 2 * A * car_y + B
        theta_desejado = np.arctan(-slope1)
        theta_carro = 0
        theta_erro = theta_desejado - theta_carro

        return curvature, lane_offset, steering_angle, theta_erro

detector = LaneDetector()

while True:
    success, image = vidcap.read()
    if not success:
        break

    frame = cv2.resize(image, (800, 600))
    original_frame = frame.copy()

    tl, bl, tr, br = (250, 470), (160, 599), (525, 470), (620, 599)
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 600], [800, 0], [800, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    transformed_frame = cv2.warpPerspective(frame, matrix, (800, 600))

    yellow_mask, white_mask = detector.adaptive_masks(transformed_frame)
    edge_mask = detector.detect_edges(transformed_frame)
    hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h, l_s, l_v = [cv2.getTrackbarPos(f"L - {c}", "Trackbars") for c in ["H", "S", "V"]]
    u_h, u_s, u_v = [cv2.getTrackbarPos(f"U - {c}", "Trackbars") for c in ["H", "S", "V"]]
    mask_hsv = cv2.inRange(hsv, (l_h, l_s, l_v), (u_h, u_s, u_v))
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edge_mask)
    combined_mask = cv2.bitwise_or(combined_mask, mask_hsv)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Exclusão da área central entre 300 e 420
    exclusion_mask = np.ones_like(combined_mask, dtype=np.uint8)
    exclusion_mask[:, 300:420] = 0  # Ajustado para 300:420
    combined_mask = combined_mask * exclusion_mask

    lines = cv2.HoughLinesP(edge_mask, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=50)
    left_base, right_base = None, None
    if lines is not None:
        left_x, right_x = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10: continue
            slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
            x_bottom = x1 + (600 - y1) * (x2 - x1) / (y2 - y1) if y2 - y1 != 0 else x1
            if slope < 0 and x_bottom < 300:  # Ajustado de 370 para 300
                left_x.append(x_bottom)
            elif slope > 0 and x_bottom > 420:  # Ajustado de 430 para 420
                right_x.append(x_bottom)
        left_base = int(np.mean(left_x)) if left_x else np.argmax(np.sum(combined_mask[300:, :300], axis=0))  # Ajustado de :370 para :300
        right_base = int(np.mean(right_x)) if right_x else np.argmax(np.sum(combined_mask[300:, 420:], axis=0)) + 420  # Ajustado de 430: para 420:
    else:
        histogram_left = np.sum(combined_mask[300:, :300], axis=0)  # Ajustado de :370 para :300
        histogram_right = np.sum(combined_mask[300:, 420:], axis=0)  # Ajustado de 430: para 420:
        left_base = np.argmax(histogram_left) if histogram_left.size > 0 else 200
        right_base = np.argmax(histogram_right) + 420 if histogram_right.size > 0 else 600  # Ajustado de +430 para +420

    y_current = 580
    left_points, right_points = [], []
    sliding_img = combined_mask.copy()
    for _ in range(15):
        left_pred = detector.kalman_left.predict() if detector.prev_left_points else np.array([left_base, y_current])
        right_pred = detector.kalman_right.predict() if detector.prev_right_points else np.array([right_base, y_current])

        win_left = combined_mask[y_current - 40:y_current, max(0, left_base - 120):left_base + 120]
        contours, _ = cv2.findContours(win_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and np.sum(win_left) > 1000:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] > 30:
                cx = int(M["m10"] / M["m00"]) + max(0, left_base - 120)
                cy = int(M["m01"] / M["m00"]) + (y_current - 40)
                left_points.append((cx, cy))
                left_base = cx
                detector.left_miss_count = 0
            else:
                detector.left_miss_count += 1
        else:
            detector.left_miss_count += 1
            if detector.left_miss_count < 5 and detector.prev_left_points:
                left_points.append((int(left_pred[0]), int(y_current - 20)))
                left_base = int(left_pred[0])

        win_right = combined_mask[y_current - 40:y_current, max(0, right_base - 120):right_base + 120]
        contours, _ = cv2.findContours(win_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and np.sum(win_right) > 1000:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] > 30:
                cx = int(M["m10"] / M["m00"]) + max(0, right_base - 120)
                cy = int(M["m01"] / M["m00"]) + (y_current - 40)
                right_points.append((cx, cy))
                right_base = cx
                detector.right_miss_count = 0
            else:
                detector.right_miss_count += 1
        else:
            detector.right_miss_count += 1
            if detector.right_miss_count < 5 and detector.prev_right_points:
                right_points.append((int(right_pred[0]), int(y_current - 20)))
                right_base = int(right_pred[0])

        cv2.rectangle(sliding_img, (left_base - 120, y_current), (left_base + 120, y_current - 40), 255, 2)
        cv2.rectangle(sliding_img, (right_base - 120, y_current), (right_base + 120, y_current - 40), 255, 2)
        y_current -= 40

    if len(left_points) > detector.MIN_LANE_POINTS:
        measurement = np.array([[np.mean([p[0] for p in left_points])],
                               [np.mean([p[1] for p in left_points])]], dtype=np.float32)
        detector.kalman_left.correct(measurement)
        detector.prev_left_points.append(left_points)
    elif detector.prev_left_points:
        prediction = detector.kalman_left.predict()
        left_points = [(int(prediction[0]), int(y)) for y in np.linspace(580, 0, 15)]

    if len(right_points) > detector.MIN_LANE_POINTS:
        measurement = np.array([[np.mean([p[0] for p in right_points])],
                               [np.mean([p[1] for p in right_points])]], dtype=np.float32)
        detector.kalman_right.correct(measurement)
        detector.prev_right_points.append(right_points)
    elif detector.prev_right_points:
        prediction = detector.kalman_right.predict()
        right_points = [(int(prediction[0]), int(y)) for y in np.linspace(580, 0, 15)]

    try:
        left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
        right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)
    except (np.linalg.LinAlgError, TypeError):
        if detector.prev_left_points and detector.prev_right_points:
            left_fit = np.polyfit([p[1] for p in detector.prev_left_points[-1]], 
                                 [p[0] for p in detector.prev_left_points[-1]], 2)
            right_fit = np.polyfit([p[1] for p in detector.prev_right_points[-1]], 
                                  [p[0] for p in detector.prev_right_points[-1]], 2)
        else:
            continue

    ploty = np.linspace(0, 599, 600)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    warp_zero = np.zeros_like(combined_mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, inv_matrix, (800, 600))
    result = cv2.addWeighted(original_frame, 1, newwarp, 0.3, 0)

    curvature, lane_offset, steering_angle, theta_erro = detector.compute_metrics(left_fit, right_fit, 600)

    cv2.putText(result, f"Offset: {lane_offset:.2f} m", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result, f"Angle error: {np.degrees(theta_erro):.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Lane Detection - Sliding Windows", sliding_img)
    cv2.imshow("Lane Detection", result)
    cv2.imshow("Mask", combined_mask)

    if cv2.waitKey(20) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()