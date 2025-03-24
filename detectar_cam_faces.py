import face_recognition
import cv2

# Carrega a imagem conhecida e calcula o encoding do rosto
imagem_conhecida = face_recognition.load_image_file("./image.jpg")
encoding_conhecido = face_recognition.face_encodings(imagem_conhecida)[0]

# Inicializa a captura de vídeo (webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Captura um frame da webcam
    ret, frame = video_capture.read()

    # Reduz o tamanho do frame para acelerar o processamento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converte o frame de BGR (formato do OpenCV) para RGB (formato usado pelo face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detecta as localizações dos rostos no frame reduzido
    locais_dos_rostos = face_recognition.face_locations(rgb_small_frame, model="hog")

    # Calcula os encodings dos rostos detectados
    encodings_dos_rostos = []
    for face_location in locais_dos_rostos:
        # Se face_location não for uma tupla, converte para (top, right, bottom, left)
        if not isinstance(face_location, tuple):
            face_location = (
                face_location.top(),
                face_location.right(),
                face_location.bottom(),
                face_location.left()
            )
        encoding = face_recognition.face_encodings(rgb_small_frame, [face_location], num_jitters=0)[0]
        encodings_dos_rostos.append(encoding)

    # Compara cada encoding detectado com o encoding conhecido
    nomes = []
    for encoding in encodings_dos_rostos:
        resultados = face_recognition.compare_faces([encoding_conhecido], encoding)
        nome = "Desconhecido"
        if True in resultados:
            nome = "Meu Rosto"
        nomes.append(nome)

    # Exibe os resultados desenhando retângulos e labels no frame original
    for (top, right, bottom, left), nome in zip(locais_dos_rostos, nomes):
        # Ajusta as coordenadas de volta para o tamanho original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Desenha um retângulo para o label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Escreve o nome abaixo do rosto
        cv2.putText(frame, nome, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Mostra o frame com as detecções e o nome
    cv2.imshow("Reconhecimento Facial", frame)

    # Encerra se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha as janelas
video_capture.release()
cv2.destroyAllWindows()
