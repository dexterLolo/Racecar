
import cv2


modelo_cascada = cv2.CascadeClassifier('<a href="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml" target="_blank" rel="noopener" data-mce-href="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml">haarcascade_frontalcatface.xml</a>')
imagen_rostro = 'cara.jpeg'

imagen = cv2.imread(imagen_rostro)

rostros = modelo_cascada.detectMultiScale(imagen, 
                                        scaleFactor=1.1, 
                                        minNeighbors=3, 
                                        minSize=(30, 30), 
                                        flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in rostros:
	cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imwrite('cara_detectadas.jpg', imagen)