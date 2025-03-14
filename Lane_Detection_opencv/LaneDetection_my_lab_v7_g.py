import cv2
import numpy as np
import warnings

# Ignorar avisos de RankWarning do np.polyfit
warnings.filterwarnings("ignore", category=np.RankWarning)

vidcap = cv2.VideoCapture("../output.mp4")
# vidcap = cv2.VideoCapture("../LaneVideo.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
# Orange - output.mp4
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 35, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# white - LaneVideo.mp4
# cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Variáveis para armazenar os pontos do frame anterior
prevLx = []
prevRx = []

#Code to make output video

# output_filename = 'lane_detection_OpenCV.mp4'
# output_frames_per_second = 20.0 
    
# # Read first frame to obtain real size
# ret, frame = vidcap.read()
# if not ret:
# 	print("Erro ao abrir o vídeo ou ler o primeiro frame!")
# 	exit()

# # height, width = frame.shape[:2]
# output_size = (800, 600)

# # Create a VideoWriter object so we can save the video output
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# result = cv2.VideoWriter(output_filename,  
# 						fourcc, 
# 						output_frames_per_second, 
# 						output_size) 



while True:
    success, image = vidcap.read()
    if not success:
        break

    # Resize o frame para 800x600
    frame = cv2.resize(image, (800, 600))
    original_frame = frame.copy()

    # Define point-s to perspective transform - output.mp4
    tl = (40, 200)
    bl = (20, 450)
    tr = (550, 200)
    br = (788, 450)
    
	# Define point-s to perspective transform - LaneVideo.mp4
    # tl = (350, 450)
    # bl = (150, 599)
    # tr = (428, 450)
    # br = (650, 599)

    # Desenhar os pontos na imagem
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Criar a transformação de perspectiva
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 600], [800, 0], [800, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (800, 600))

    # --- Data Augmentation Avançado ---
    # Ajuste de brilho e contraste dinâmico
    alpha = 1.2  # Contraste
    beta = 10    # Brilho
    transformed_frame_aug = cv2.convertScaleAbs(transformed_frame, alpha=alpha, beta=beta)

    # Conversão para HSV e equalização adaptativa (CLAHE) no canal V
    hsv_transformed = cv2.cvtColor(transformed_frame_aug, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_transformed)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_transformed = cv2.merge((h, s, v_eq))

    # Limiarização no frame equalizado
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed, lower, upper)

    # Suavização e morfologia para conectar lacunas
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Opcional: Visualizar o frame equalizado
    # debug_frame = cv2.cvtColor(hsv_transformed, cv2.COLOR_HSV2BGR)
    # cv2.imshow("Equalized Frame", debug_frame)

    # Histograma para definir a base das faixas
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # --- Sliding Window Melhorado ---
    y = 592  # Posição y inicial
    left_points = []
    right_points = []
    msk = mask.copy()


    while y > 0:
        # Faixa esquerda (área de busca aumentada para ±75)
        img_left = mask[y-40:y, max(0, left_base-75):left_base+75]
        contours, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_left = False
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filtro de área mínima
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    abs_x = max(0, left_base - 75) + cx
                    abs_y = (y - 40) + cy
                    # Validação: aceitar apenas se não desviar muito do anterior
                    if not prevLx or abs(abs_x - prevLx[-1][0]) < 100:
                        left_points.append((abs_x, abs_y))
                        left_base = abs_x
                        found_left = True
                        break
        if not found_left and prevLx:
            left_base = prevLx[-1][0]

        # Faixa direita (área de busca aumentada para ±75)
        img_right = mask[y-40:y, right_base-75:min(800, right_base+75)]
        contours, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_right = False
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filtro de área mínima
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    abs_x = right_base - 75 + cx
                    abs_y = (y - 40) + cy
                    # Validação: aceitar apenas se não desviar muito do anterior
                    if not prevRx or abs(abs_x - prevRx[-1][0]) < 100:
                        right_points.append((abs_x, abs_y))
                        right_base = abs_x
                        found_right = True
                        break
        if not found_right and prevRx:
            right_base = prevRx[-1][0]

        # Desenhar janelas
        cv2.rectangle(msk, (left_base-75, y), (left_base+75, y-40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base-75, y), (right_base+75, y-40), (255, 255, 255), 2)
        y -= 40

    # Atualizar pontos anteriores
    prevLx = left_points.copy() if left_points else prevLx
    prevRx = right_points.copy() if right_points else prevRx

    # Verificar pontos suficientes
    if len(left_points) < 3 or len(right_points) < 3:
        if prevLx and prevRx:
            left_points = prevLx
            right_points = prevRx
        else:
            continue

    # Ajuste polinomial
    left_x = np.array([p[0] for p in left_points])
    left_y = np.array([p[1] for p in left_points])
    right_x = np.array([p[0] for p in right_points])
    right_y = np.array([p[1] for p in right_points])

    try:
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
    except np.linalg.LinAlgError:
        continue

    # Gerar pontos para desenho
    plot_y = np.linspace(0, 599, num=600)
    left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
    center_fit_x = (left_fit_x + right_fit_x) / 2

    # Posição do carro
    car_x = 400
    car_y = 599

    # Cálculo da direção
    center_fit = np.polyfit(plot_y, center_fit_x, 2)
    A, B, C = center_fit
    slope = 2 * A * car_y + B
    theta_desejado = np.arctan(-slope)
    theta_carro = 0
    theta_erro = theta_desejado - theta_carro

    # Cálculo da curvatura
    y_eval = 599
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])
    curvature = (left_curvature + right_curvature) / 2

    # Cálculo do offset
    lane_center = (left_base + right_base) / 2
    lane_offset = (car_x - lane_center) * 0.26 / 800  # Ajustado para 3.7m (largura típica de faixa)

    # Cálculo do ângulo de direção
    steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi

    # Sobreposição da faixa
    left_points_sorted = sorted(left_points, key=lambda p: p[1], reverse=True)
    right_points_sorted = sorted(right_points, key=lambda p: p[1], reverse=True)
    top_left = left_points_sorted[0]
    bottom_left = left_points_sorted[-1]
    top_right = right_points_sorted[0]
    bottom_right = right_points_sorted[-1]

    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    quad_points_birdseye = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.float32)
    quad_points_birdseye = quad_points_birdseye.reshape(-1, 1, 2)
    quad_points_original = cv2.perspectiveTransform(quad_points_birdseye, inv_matrix)

    result_frame = original_frame.copy()
    overlay = result_frame.copy()
    cv2.fillPoly(overlay, [np.int32(quad_points_original)], (0, 255, 0))
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

    # Exibir informações
    # cv2.putText(result_frame, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(result_frame, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Angle error: {np.degrees(theta_erro):.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    #Conert msk to 3 channel - to make video slinding windows
    # msk_bgr = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)

    # Exibir janelas
    cv2.imshow("Original", original_frame)
    # cv2.imshow("Points", frame)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow("Lane Detection", result_frame)
    # result.write(msk_bgr)
    # result.write(result_frame)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
# result.release()
cv2.destroyAllWindows()