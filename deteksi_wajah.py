import cv2
import matplotlib.pyplot as plt

# gambar yang akan digunakan
image_path = 'foto.jpeg'  # Ganti dengan path gambar Anda

# model Haar Cascade yang telah dilatih untuk pendeteksian wajah
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# gambar dari file
image = cv2.imread(image_path)

# Mengonversi gambar ke skala abu-abu
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mendeteksi wajah dalam gambar
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

# Menandai wajah yang terdeteksi dengan kotak
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 5)

# Menampilkan hasil gambar dengan wajah yang terdeteksi menggunakan matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Faces')
plt.show()

# Menyimpan hasil gambar
cv2.imwrite('hasil_deteksi_wajah.jpg', image)
