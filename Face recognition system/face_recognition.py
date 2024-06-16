import cv2

# LBPHFaceRecognizer_create() metodunu kullanarak yüz tanıma modelini oluşturun
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Eğitilmiş modeli dosyadan okuyun
recognizer.read('trainer/trainer.yml')

# Önceden oluşturulmuş modeli yüklemek için kullanılan dosyanın yolu
cascadePath = "haarcascade_frontalface_default.xml"

# OpenCV tarafından sağlanan eğitilmiş modeli kullanmak için haarcascade sınıfını oluşturun
faceCascade = cv2.CascadeClassifier(cascadePath);

# Metin yazısı için kullanılacak font
font = cv2.FONT_HERSHEY_SIMPLEX

# Video yakalama işlemini başlatın ve başlangıç yapın
cam = cv2.VideoCapture(0)

while True:
    # Video çerçevesini okuyun
    ret, im = cam.read()

    # Yakalanan çerçeveyi gri tonlamaya dönüştürün
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Çerçevedeki tüm yüzleri alın
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # Her yüz için döngü
    for (x, y, w, h) in faces:
        # Yüzün etrafına dikdörtgen çizin
        cv2.rectangle(im, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)

        # Yüzün hangi ID'ye ait olduğunu belirleyin
        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # ID mevcut ise
        if confidence < 100:
            # ID'yi belirli isme atayın
            if Id == 1:
                Id = "Jenifer"
            elif Id == 2:
                Id = "Jacky"
            elif Id == 3:
                Id = "Emily"

        # Tanımlanan kişiyi ekrana yazın
        cv2.putText(im, str(Id), (x, y-40), font, 2, (255, 255, 255), 3)

    # Sınırlı dikdörtgenlerle birlikte video çerçevesini gösterin
    cv2.imshow('im', im)

    # 'q' tuşuna basıldığında döngüyü sonlandırın
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakın
cam.release()

# Tüm pencereleri kapatın
cv2.destroyAllWindows()

