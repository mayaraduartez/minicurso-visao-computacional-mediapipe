import cv2 #opencv acessar webcam
import mediapipe as mp #deteccao de rosto e manipular modelos
import numpy as np #manipular imagens
from mediapipe.tasks import python #modulo python do mediapipe pra usar esses modelos
from mediapipe.tasks.python import vision  # modulo para deteccao de rosto
from mediapipe.tasks.python.vision import drawing_utils #modulo para desenhar os pontos de referencia do rosto

#funcao de desenho
def desenho_imagem(imagem, deteccoes):
    copy_imagem = np.copy(imagem) #copia imagem

    if not deteccoes.face_landmarks: #se nao tem deteccoes de rosto, retorna a imagem original
        return copy_imagem
    
    for face_landmarks in deteccoes.face_landmarks: # para cada rosto detectado, desenha os pontos de referencia do rosto usando o modulo drawing_utils

        drawing_utils.draw_landmarks(
            image = copy_imagem, #imagem a ser usada
            landmark_list = face_landmarks, #lista de pontos de referencia do rosto
            connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,   #conexoes entre os pontos de referencia do rosto
            landmark_drawing_spec = drawing_utils.DrawingSpec( #configurar como os pontos serao desenhados
                color=(0,255,0), # cor verde
                thickness=1,  #espessura da linha
                circle_radius=1 #espressura dos pontos
                )
        )

        drawing_utils.draw_landmarks( # desenho do olho esquerdo e direito
            image = copy_imagem, #imagem a ser usada
            landmark_list = face_landmarks, 
            connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS, # conexoes entre os pontos de referencia do olho esquerdo
            landmark_drawing_spec = None, #nao desenha os pontos de referencia do olho, apenas as conexoes entre eles
            connection_drawing_spec = drawing_utils.DrawingSpec( # configurar como as conexoes entre os pontos de referencia do olho serao desenhadas
                color=(0,255,0), # verde
                thickness=1, #espessura da linha
                circle_radius=1 #espessura do ponto
                )
        )

        drawing_utils.draw_landmarks( # desenho do olho esquerdo e direito
            image = copy_imagem, #imagem a ser usada
            landmark_list = face_landmarks, 
            connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS, # conexoes entre os pontos de referencia do olho direito
            landmark_drawing_spec = None, #nao desenha os pontos de referencia do olho, apenas as conexoes entre eles
            connection_drawing_spec = drawing_utils.DrawingSpec( # configurar como as conexoes entre os pontos de referencia do olho serao desenhadas
                color=(0,255,0), # verde
                thickness=1, #espessura da linha
                circle_radius=1 #espessura do ponto
                )
        )

    return copy_imagem #retorna a imagem com os pontos de referencia do rosto desenhados

capture = cv2.VideoCapture(0) #cria um objeto de captura de video usando a webcam (indice 0)

base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task") #configura as opcoes para o modelo de deteccao de rosto, especificando o caminho para o arquivo do modelo

#comeca a configuracao do detector de rosto, usa as bases de cima
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    num_faces=2,
    min_face_detection_confidence=0.5, #confiança mínima para considerar uma detecção de rosto válida (0.5 ou 50%)
    min_face_presence_confidence=0.5 #confiança mínima para considerar que um rosto está presente na imagem (0.5 ou 50%
    )


detector = vision.FaceLandmarker.create_from_options(options) # cria o detector de rosto usando as opções configuradas acima

# loop para capturar video da webcam, processar cada frame e exibir o resultado com os pontos de referencia do rosto desenhados
while True:
    sucesso, imagem = capture.read()

    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break

    # inverte a imagem horizontalmente para criar um efeito de espelho, o que é comum em aplicativos de webcam para que o movimento do usuário corresponda ao que ele vê na tela
    frame = cv2.flip(imagem, 1)

    # converte a imagem de BGR (formato padrão do OpenCV) para RGB (formato esperado pelo modelo de detecção de rosto do MediaPipe)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # cria um objeto Image do MediaPipe a partir da imagem RGB, especificando o formato da imagem como SRGB (Standard RGB) e passando os dados da imagem RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)

    #realiza a detecção de rosto na imagem usando o método detect do detector, que retorna as detecções de rosto encontradas na imagem
    deteccoes = detector.detect(mp_image)

    #desenha os pontos de referencia do rosto na imagem usando a função desenho_imagem, passando a imagem RGB e as detecções de rosto como argumentos, e armazena o resultado em imagem_desenhada
    imagem_desenhada = desenho_imagem(mp_image.numpy_view(), deteccoes)

    #converte a imagem desenhada de volta para o formato BGR (formato padrão do OpenCV) para exibição usando cv2.imshow
    mp_image = cv2.cvtColor(imagem_desenhada, cv2.COLOR_RGB2BGR)

    #exibe a imagem com os pontos de referencia do rosto desenhados em uma janela chamada "Face Landmarker"
    cv2.imshow("Face Landmarker", mp_image)

    #verifica se a tecla "Esc" (código 27) foi pressionada para sair do loop e encerrar o programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

#libera os recursos da webcam e fecha todas as janelas do OpenCV
capture.release()
cv2.destroyAllWindows()
