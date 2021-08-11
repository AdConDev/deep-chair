import cv2

# Conexión con cámara
camara = cv2.VideoCapture(0)
valido, foto = camara.read()

if valido == True:
    cv2.imwrite('prueba.jpg', foto)
    print('Foto lista')
else:
    print('Error al iniciar cámara')

camara.release()
