import cv2
import numpy as np
import warnings

#Ignorar avisos de RankWarning do np.polyfit
warnings.filterwarnings("ignore", category=np.RankWarning)

#Configurações de vídeo
fps = 30
width, height = 800, 600

# alterar <IP_RECEIVER> para o IP da máquina que receberá o stream.
gst_str = (
    "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "    
    "rtph264pay config-interval=1 pt=96 ! udpsink host=localhost port=5000"
)

# Cria o VideoWriter com o pipeline GStreamer
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)

vidcap = cv2.VideoCapture("output.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 35, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Variáveis para armazenar os pontos do frame anterior (caso algum frame não tenha pontos suficientes)
prevLx = []
prevRx = []

while True:
    success, image = vidcap.read()
    if not success:
        break

    # Redimensiona o frame para 800x600
    frame = cv2.resize(image, (800, 600))
    original_frame = frame.copy()

    # Definir os pontos para transformação de perspectiva (pontos na imagem original)
    tl = (30, 200)
    bl = (20, 450)
    tr = (550, 200)
    br = (788, 450)

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

    # Conversão para HSV e limiarização (thresholding) para detecção de faixa
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Histograma para definir a base das faixas
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # ------------------- Sliding Window -------------------
    y = 592  # Posição y inicial (parte inferior da imagem)
    left_points = []   # Lista para armazenar os pontos da faixa esquerda (x, y)
    right_points = []  # Lista para armazenar os pontos da faixa direita (x, y)

    msk = mask.copy()

    while y > 0:
        # Processamento para a faixa esquerda
        img_left = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_left = False
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                abs_x = left_base - 50 + cx
                abs_y = (y - 40) + cy  # Posição real em y
                left_points.append((abs_x, abs_y))
                left_base = abs_x  # Atualiza a base para a próxima janela
                found_left = True
        # Se nenhum contorno for encontrado, use o último valor válido
        if not found_left and prevLx:
            left_base = prevLx[-1][0]

        # Processamento para a faixa direita
        img_right = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found_right = False
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                abs_x = right_base - 50 + cx
                abs_y = (y - 40) + cy  # Posição real em y
                right_points.append((abs_x, abs_y))
                right_base = abs_x  # Atualiza a base para a próxima janela
                found_right = True
        if not found_right and prevRx:
            right_base = prevRx[-1][0]

        # Desenhar as janelas deslizantes para visualização
        cv2.rectangle(msk, (left_base-50, y), (left_base+50, y-40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base-50, y), (right_base+50, y-40), (255, 255, 255), 2)
        y -= 40

    # Atualizar os pontos anteriores (caso o frame atual não tenha pontos suficientes)
    prevLx = left_points.copy() if left_points else prevLx
    prevRx = right_points.copy() if right_points else prevRx

    # Verificar se há pontos suficientes para o ajuste polinomial
    if len(left_points) < 3 or len(right_points) < 3:
        if prevLx and prevRx:
            left_points = prevLx
            right_points = prevRx
        else:
            continue  # Pula o frame se não houver dados suficientes

    # Extrair as coordenadas para o ajuste polinomial
    left_x = np.array([p[0] for p in left_points])
    left_y = np.array([p[1] for p in left_points])
    right_x = np.array([p[0] for p in right_points])
    right_y = np.array([p[1] for p in right_points])

    # Ajustar polinômios de segundo grau para cada faixa
    try:
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
    except np.linalg.LinAlgError:
        continue  # Pula o frame se o ajuste falhar
    
	# # Cálculo da posição central
    # y_eval = 592  # Posição do carro na imagem
    # left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
    # right_slope = 2 * right_fit[0] * y_eval + right_fit[1]

    # center_slope = (left_slope + right_slope) / 2
    # theta_trajetoria = np.arctan(center_slope)

	# # Assumindo que o carro está alinhado no início
    # theta_carro = 0  # Ou use dados de um sensor IMU

	# # Cálculo do erro da trajetória
    # theta_erro = theta_trajetoria - theta_carro
    # print(f"Erro da trajetória: {theta_erro} rad")
    


	# Correção: Gerar plot_y de acordo com a altura real da imagem (600)
    plot_y = np.linspace(0, 599, num=600)
    left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
    center_fit_x = (left_fit_x + right_fit_x) / 2

    # Posição do carro corrigida (centro inferior da imagem 800x600)
    car_x = 400
    car_y = 599  # Último pixel válido na vertical

    # Cálculo da direção da trajetória usando derivada (correto)
    # Coeficientes do polinômio central (assumindo center_fit como [A, B, C])
    if len(center_fit_x) >= 3:
        # Ajuste para obter coeficientes do centro
        center_fit = np.polyfit(plot_y, center_fit_x, 2)
        A, B, C = center_fit
        
        # Derivada dy/dx no ponto do carro (y=599)
        slope = 2 * A * car_y + B  # dx/dy
        theta_desejado = np.arctan(slope)
        
        # Ângulo do carro (assumindo alinhamento com a estrada)
        theta_carro = 0  # Em radianos (alinhado com eixo y negativo)
        
        # Erro de trajetória (diferença angular)
        theta_erro = theta_desejado - theta_carro

        # print(f"Theta desejado: {np.degrees(theta_desejado):.2f}°")
        # print(f"Erro de trajetória: {np.degrees(theta_erro):.2f}°\n")
    

	

    # Cálculo da curvatura das faixas
    y_eval = 599  # Ponto de avaliação (parte inferior da imagem)
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])
    curvature = (left_curvature + right_curvature) / 2

    # Cálculo do offset 
    lane_center = (left_base + right_base) / 2
    car_position = 400  # Assumindo que o carro está centralizado na imagem
    lane_offset = (car_position - lane_center) * 0.2 / 800  # Conversão de pixels para metros

    # Cálculo do ângulo de direção
    steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi

    # Cálculo do ponto final para desenhar a linha (para visualização do ângulo)
    line_length = 100
    end_x = int(400 + line_length * np.sin(np.radians(steering_angle)))
    end_y = int(600 - line_length * np.cos(np.radians(steering_angle)))

    # Definir os pontos do quadrilátero que representa a faixa para sobreposição
    # Ordena os pontos de cada lado pela coordenada y (maior para menor)
    left_points_sorted = sorted(left_points, key=lambda p: p[1], reverse=True)
    right_points_sorted = sorted(right_points, key=lambda p: p[1], reverse=True)

    top_left = left_points_sorted[0]
    bottom_left = left_points_sorted[-1]
    top_right = right_points_sorted[0]
    bottom_right = right_points_sorted[-1]

    # Transformação inversa para mapear o quadrilátero de volta para a perspectiva original
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    quad_points_birdseye = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.float32)
    quad_points_birdseye = quad_points_birdseye.reshape(-1, 1, 2)
    quad_points_original = cv2.perspectiveTransform(quad_points_birdseye, inv_matrix)

    # Sobreposição da faixa detectada no frame original
    result_frame = original_frame.copy()
    overlay = result_frame.copy()
    cv2.fillPoly(overlay, [np.int32(quad_points_original)], (0, 255, 0))
    alpha = 0.3  # Opacidade da sobreposição
    cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

    # Exibir informações na tela
    cv2.putText(result_frame, f'Curvature: {curvature:.2f} m', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, f'Offset: {lane_offset:.2f} m', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, f'Angle: {steering_angle:.2f} deg', (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Angle error: {np.degrees(theta_erro):.2f}", (30, 150),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    

	# Draw a straight line in the center of the frame pointing with the current angle
    cv2.line(result_frame, (400, 600), (end_x, end_y), (255, 0, 0), 2)
    
    # Envia o frame para o pipeline
    out.write(result_frame)

    # Exibir janelas com os resultados
    # cv2.imshow("Original", original_frame)
    # cv2.imshow("Bird's Eye View", transformed_frame)
    # cv2.imshow("Lane Detection - Image Thresholding", mask)
    # cv2.imshow("Lane Detection - Sliding Windows", msk)
    # cv2.imshow("Lane Detection", result_frame)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()



# theta_carro = np.arctan2(y2 - y1, x2 - x1)