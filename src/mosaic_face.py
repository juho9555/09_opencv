import cv2

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
frontal_face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

profile_face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)
while cap.isOpened():    
    ret, frame = cap.read()  # 프레임 읽기
    rate = 15
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_w = frame.shape[1]
        
        # faces_all을 리스트 형식으로 만들기
        faces_all = []

        # 정면 얼굴 검출    
        faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        for (x, y, w, h) in faces:
            faces_all.append((x, y, w, h)) # 좌표 변환

        # 왼쪽 측면 얼굴 검출
        profiles = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in profiles:
            faces_all.append((x, y, w, h)) # 좌표를 원본 이미지 기준으로 변환

        # 오른쪽 측면 얼굴 검출
        gray_flipped = cv2.flip(gray, 1)
        profiles_flipped = profile_face_cascade.detectMultiScale(gray_flipped, 1.3, 5)
        for (x, y, w, h) in profiles_flipped:
            x_original = frame_w - (x + w)
            faces_all.append((x_original, y, w, h)) # 좌표 변환

        for(x,y,w,h) in faces_all:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0),2)
            # 얼굴부분 가져오기
            face = frame[y:y+h, x:x+w]

            small = cv2.resize(face, (w//rate, h//rate)) # 1/rate 비율로 축소
            # 원래 크기로 확대
            mosaic = cv2.resize(small, (w,h), interpolation=cv2.INTER_AREA)  
            frame[y:y+h, x:x+w] = mosaic
        cv2.imshow('face mosaic', frame)
    else:
        break
    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()

