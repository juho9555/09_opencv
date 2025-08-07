import cv2

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)
while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    rate = 15
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0),2)
            # 얼굴부분 가져오기
            face = img[y:y+h, x:x+w]

            small = cv2.resize(face, (w//rate, h//rate)) # 1/rate 비율로 축소
            # 원래 크기로 확대
            mosaic = cv2.resize(small, (w,h), interpolation=cv2.INTER_AREA)  
            img[y:y+h, x:x+w] = mosaic
        cv2.imshow('face mosaic', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()

