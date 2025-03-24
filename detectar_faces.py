import face_recognition
import cv2

imagem = face_recognition.load_image_file("./image.jpg")


locais_dos_rostos = face_recognition.face_locations(imagem)
print(f"Foram encontrados {len(locais_dos_rostos)} rostos nesta imagem.")


imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)


for (top, right, bottom, left) in locais_dos_rostos:
    cv2.rectangle(imagem_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow("Rostos Detectados", imagem_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
