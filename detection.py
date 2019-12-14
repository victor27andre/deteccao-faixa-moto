import numpy as np
import cv2
import time

# identificar o nome do arquivo do vídeo a ser analisado
# recomendado colocar um video de 720p
cap = cv2.VideoCapture('testvideocut3.mp4')

# percorrer até que todo o arquivo de vídeo seja reproduzido
while(cap.isOpened()):

    # leia o quadro de vídeo e mostre na tela
    ret, frame = cap.read()
    
    #cv2.imshow("Original Scene", frame)
    
    #print(len(frame),(len(frame[0])))
    #break
    # recorte da seção do quadro de vídeo de interesse e mostre na tela
    snip = frame[300:1080,200:1000]
    
    
    #cv2.imshow("Snip",snip)

    # crie uma máscara de polígono (trapézio) para selecionar a região de interesse
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[450,150], [0, 400], [800, 400], [500, 170]], dtype=np.int32)
    
    cv2.fillConvexPoly(mask, pts, 255)
    #cv2.imshow("Mask", mask)

    # aplicar máscara e mostrar imagem mascarada na tela
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    #cv2.imshow("Region of Interest", masked)

    # converter para escala de cinza e depois preto/branco para imagem binária
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # controla a intensidade 
    thresh = 70
    
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Black/White", frame)

    # desfocar a imagem para ajudar na detecção de borda
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    cv2.imshow("Blurred", blurred)

    # identificar bordas e mostrar na tela
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Edged", edged)

    # realiza a transformação da técnica Hough para identificar as linhas da pista
    lines = cv2.HoughLines(edged, 1, np.pi / 80, 50)

    # definir matrizes para faixas esquerda e direita
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []


    # verifique se cv2.HoughLines encontrou pelo menos uma linha
    if lines is not None:

        # # percorre todas as linhas encontradas pelo cv2.HoughLines
        for i in range(0, len(lines)):

            # avaliar cada linha de 'linhas' de saída do cv2.HoughLines
            for rho, theta in lines[i]:

                # coleta pista da esquerda
                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)

                    # # plot all lane lines for DEMO PURPOSES ONLY
                    # a = np.cos(theta); b = np.sin(theta)
                    # x0 = a * rho; y0 = b * rho
                    # x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                    # x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    #
                    # cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # coleta pista da direita
                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)

                    # # plot all lane lines for DEMO PURPOSES ONLY
                     #a = np.cos(theta); b = np.sin(theta)
                     #x0 = a * rho; y0 = b * rho
                     #x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                     #x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    
                     #cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # estatística para identificar as dimensões medianas da faixa
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    # linha mediana da reta até o topo da tela
    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        offset1 = 200; offset2 = 800
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))

        cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/4:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 600; offset2 = 1000
        x3 = int(x0 - offset1 * (-b)); y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b)); y4 = int(y0 - offset2 * (a))

        cv2.line(snip, (x3, y3), (x4, y4), (255, 0, 0), 6)



    # sobreposição de contorno de pista semi-transparente no original
    if left_theta > np.pi/4 and right_theta > np.pi/4:
    #if left_theta > np.pi:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        # (1) crie uma cópia do original:
        overlay = snip.copy()
        # (2) crie uma cópia do original
        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))
        # (3)  crie uma cópia do original
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)

    cv2.imshow("Lined", snip)


    # pressione a tecla 'q' para interromper o vídeo
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# limpar tudo uma vez terminado
cap.release()
cv2.destroyAllWindows()

