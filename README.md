# OpenCV

<details>
<summary>Haarcarcade</summary>
    
## 웹캠을 이용한 얼굴 검출기
```
import cv2

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(0)
while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0),2)
            roi = gray[y:y+h, x:x+w]

        cv2.imshow('face detect', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
```
- haarcascade_frontalface_default.xml(정면 얼굴 인식) 파일을 data폴더안에 위치시킴
- gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 그레이스케일로 변환하여 얼굴 검출을 더 정확하게 함
- for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0),2)
            roi = gray[y:y+h, x:x+w] = ROI영역 설정
- 
