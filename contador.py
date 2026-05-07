import cv2 #opencv acessar webcam
import mediapipe as mp #deteccao de rosto e manipular modelos
import numpy as np #manipular imagens
from mediapipe.tasks import python #modulo python do mediapipe pra usar esses modelos
from mediapipe.tasks.python import vision  # modulo para deteccao de rosto
from mediapipe.tasks.python.vision import drawing_utils #modulo para desenhar os pontos de referencia do rosto

#funcao que conta, recebe os pontos da mao e a info se é direita ou esquerda
def contador(hand_landmarks, handedness):
    dedos = 0

    #lista com os pontos das ponta dos dedos
    pontos = [4, 8, 12, 16, 20]

    #percorre os dedos ignorando o polegar, porque o polegar se comporta diferente dos outros dedos
    for ponta in pontos[1:]:
        # verifica se a popnta do dedo está acima da articulacao do dedo 
        if hand_landmarks[ponta].y < hand_landmarks[ponta - 2].y:
            dedos += 1

    #descobre qual mao é 
    mao = handedness[0].category_name

    #se direita, usa uma regra do polegar
    if mao == "Right":
        if hand_landmarks[4].x > hand_landmarks[3].x:
            dedos += 1
    else:
        if hand_landmarks[4].x < hand_landmarks[3].x:
            dedos += 1

    return dedos


def desenho_maos(imagem, deteccoes):
    copy_imagem = np.copy(imagem)
    total_dedos = 0

    if not deteccoes.hand_landmarks: #se nao tem deteccoes de mao, retorna a imagem original
        return copy_imagem, total_dedos
    
    #para cada mao detectada, pega os pontos e a info se é D ou E
    for hand_landmarks, handedness in zip(
            deteccoes.hand_landmarks,
            deteccoes.handedness
        ):
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

        total_dedos += contador(hand_landmarks, handedness)

    return copy_imagem, total_dedos #retorna a imagem com os pontos de referencia da mao desenhados

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

    imagem_com_desenho, total_dedos = desenho_maos(mp_image.numpy_view(), deteccoes) #desenha os pontos de referencia da mao na imagem usando a funcao de desenho definida acima

    imagem_com_desenho = cv2.cvtColor(imagem_com_desenho, cv2.COLOR_RGB2BGR) #converte a imagem de volta para BGR para exibir usando o OpenCV

    
    cv2.putText(imagem_com_desenho, f"Total de Dedos: {total_dedos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #exibe o total de dedos contados na imagem

    cv2.imshow("Deteccao de Mao", imagem_com_desenho) #exibe a imagem com os pontos de referencia da mao desenhados em uma janela chamada "Deteccao de Mao"

    if cv2.waitKey(1) & 0xFF == 27: #espera por 1 milissegundo e verifica se a tecla 'q' foi pressionada para sair do loop
        break

capture.release() #libera a captura de video da webcam
cv2.destroyAllWindows() #fecha todas as janelas abertas pelo OpenCV