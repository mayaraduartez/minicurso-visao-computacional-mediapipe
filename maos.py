import cv2 #opencv acessar webcam
import mediapipe as mp #deteccao de rosto e manipular modelos
import numpy as np #manipular imagens
from mediapipe.tasks import python #modulo python do mediapipe pra usar esses modelos
from mediapipe.tasks.python import vision  # modulo para deteccao de rosto
from mediapipe.tasks.python.vision import drawing_utils #modulo para desenhar os pontos de referencia do rosto

def desenho_maos(imagem, deteccoes):
    copy_imagem = np.copy(imagem)

    if not deteccoes.hand_landmarks: #se nao tem deteccoes de mao, retorna a imagem original
        return copy_imagem
    
    for hand_landmarks in deteccoes.hand_landmarks: 
        # para cada mao detectada, desenha os pontos de referencia da mao usando o modulo drawing_utils

        drawing_utils.draw_landmarks(
            image = copy_imagem, #imagem a ser usada
            landmark_list = hand_landmarks, #lista de pontos de referencia da mao
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS,   #conexoes entre os pontos de referencia da mao
            landmark_drawing_spec = drawing_utils.DrawingSpec( #configurar como os pontos serao desenhados
                color=(0,255,0), # cor verde
                thickness=2,  #espessura da linha
                circle_radius=3 #espressura dos pontos
                ),
            connection_drawing_spec = drawing_utils.DrawingSpec( # configurar como as conexoes entre os pontos de referencia da mao serao desenhadas
                color=(255,255,255), # branco
                thickness=2, #espessura da linha
            )
        )

    return copy_imagem #retorna a imagem com os pontos de referencia da mao desenhados

capture = cv2.VideoCapture(0) 

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task") #configura as opcoes para o modelo de deteccao de mao, especificando o caminho para o arquivo do modelo

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2 #configura o modelo para detectar no maximo 2 maos
)

detector_maos = vision.HandLandmarker.create_from_options(options) #cria o objeto de deteccao de mao usando as opcoes configuradas acima

while True:
    sucesso, imagem = capture.read() #le um frame da webcam e armazena na variavel imagem

    if not sucesso: #se nao conseguiu ler um frame, continua para a proxima iteracao do loop
        break

    imagem = cv2.flip(imagem, 1) #espelha a imagem para que a mao direita apareca do lado direito da tela e a mao esquerda do lado esquerdo da tela

    rgb_frame = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) #converte a imagem de BGR (formato usado pelo OpenCV) para RGB (formato usado pelo mediapipe)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) #cria um objeto de imagem do mediapipe usando a imagem convertida para RGB

    deteccoes = detector_maos.detect(mp_image) #realiza a deteccao de mao na imagem usando o objeto de deteccao de mao criado acima 

    imagem_com_desenho = desenho_maos(mp_image.numpy_view(), deteccoes) #desenha os pontos de referencia da mao na imagem usando a funcao de desenho definida acima

    imagem_com_desenho = cv2.cvtColor(imagem_com_desenho, cv2.COLOR_RGB2BGR) #converte a imagem de volta para BGR para exibir usando o OpenCV

    cv2.imshow("Deteccao de Mao", imagem_com_desenho) #exibe a imagem com os pontos de referencia da mao desenhados em uma janela chamada "Deteccao de Mao"

    if cv2.waitKey(1) & 0xFF == 27: #espera por 1 milissegundo e verifica se a tecla 'q' foi pressionada para sair do loop
        break

capture.release() #libera a captura de video da webcam
cv2.destroyAllWindows() #fecha todas as janelas abertas pelo OpenCV