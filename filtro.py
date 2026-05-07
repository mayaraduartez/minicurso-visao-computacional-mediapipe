import cv2 #opencv acessar webcam
import mediapipe as mp #deteccao de rosto e manipular modelos
import numpy as np #manipular imagens
from mediapipe.tasks import python #modulo python do mediapipe pra usar esses modelos
from mediapipe.tasks.python import vision  # modulo para deteccao de rosto
from mediapipe.tasks.python.vision import drawing_utils #modulo para desenhar os pontos de referencia do rosto
import math #blibioteca para calcular a distancia entre os olhos
from mediapipe import Image as MediaPipeImage #importa a classe Image do mediapipe para criar objetos de imagem a partir dos frames da webcam


#inicializa o modelo de detecção facial do mediapipe usando as opções definidas abaixo
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')

#configura as opções para o modelo de detecção facial, especificando o número máximo de faces a serem detectadas e os limiares de confiança para a detecção e rastreamento facial
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# carrega o modelo de detecção facial usando as opções definidas acima
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

#função para calcular a distância entre dois pontos (usada para calcular o raio das lentes dos óculos)
def distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# loop principal
while True:
    sucesso, imagem = cap.read()
    if not sucesso:
        break

    imagem = cv2.flip(imagem, 1) #espelha a imagem para que a mao direita apareca do lado direito da tela e a mao esquerda do lado esquerdo da tela

    #pega as dimensões da imagem para calcular a posição dos pontos de referência do rosto em pixels
    h, w = imagem.shape

    rgb_frame = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) #converte a imagem de BGR (formato usado pelo OpenCV) para RGB (formato usado pelo mediapipe)

    mp_image = mp.Image(image_format=1, data=rgb_frame) #cria um objeto de imagem do mediapipe usando a imagem convertida para RGB

    deteccao = detector.detect(mp_image)

    if deteccao.face_landmarks:
        face = deteccao.face_landmarks[0] # pega o primeiro rosto detectado
        lm = face # cria um apelido chamado lm para os landmarks do rosto

        #pontos aproximados dos olhos 
         # ponto 33 do rosto usado como referencia de um olho
         # como o media pipe entrega x e y, multiplicamos para trasformar em posição real na largura da imagem
        olho_esquerdo = (int(lm[33].x * w), int(lm[33].y * h))
        olho_direito = (int(lm[263].x * w), int(lm[263].y * h))
        nariz = (int(lm[1].x * w), int(lm[1].y * h))

        #calcula a distancia entre os olhos para determinar o tamanho dos óculos, e multiplica por um fator para ajustar o tamanho das lentes
        raio = int(distancia(olho_esquerdo, olho_direito) * 0.40)

        # desenhar lentes 
        # imagem que vai ser desenhada, centro da lente, raio da lente, cor da lente (0,0,0) para preto, espessura da linha (2)
        cv2.circle(imagem, olho_esquerdo, raio, (0, 0, 0), 2)
        cv2.circle(imagem, olho_direito, raio, (0, 0, 0), 2)
        
        # cria um ponto médio entre os olhos para desenhar a ponte dos óculos, usando a posição dos olhos e o raio das lentes para calcular a posição do ponto médio
        ponto_meio_esq = (olho_esquerdo[0] + raio, olho_esquerdo[1])
        ponto_meio_dir = (olho_direito[0] - raio, olho_direito[1])

        #desenha a ponte que liga as duas lentes
        cv2.line(imagem, ponto_meio_esq, ponto_meio_dir, (0, 0, 0), 4)

        #hastes
        # comeca na lateral externa da lente esquerda 
        # vai 60 pixels para a esquerda e 20 pixels para cima para criar um efeito de haste inclinada
        cv2.line(imagem, (olho_esquerdo[0] - raio, olho_esquerdo[1]), 
            (olho_esquerdo[0] - raio - 60, olho_esquerdo[1] - 20), (0, 0, 0), 4)
        
        # mesma coisa para a haste direita
        cv2.line(imagem, (olho_direito[0] + raio, olho_direito[1]), 
            (olho_direito[0] + raio + 60, olho_direito[1] - 20), (0, 0, 0), 4)
        
        # nariz desenha um circulo vermelho no nariz
        cv2.circle(imagem, nariz, 20, (0, 0, 255), cv2.FILLED)

    # abre a janela e mostra a imagem de retorno
    cv2.imshow('Filtro de Óculos', imagem )
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()