# 2. 눈 랜드마크 검출하기
import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

Left_eye = list(range(36, 42)) # 왼쪽 눈
Right_eye = list(range(42, 48)) # 오른쪽 눈

cap = cv2.VideoCapture(0) # 웹캠 인식

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('no frame.');break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for rect in faces:
        # 눈만 검출
        shape = predictor(gray, rect)
        for i in Left_eye + Right_eye:
            part = shape.part(i)
            cv2.circle(frame, (part.x, part.y), 2, (0, 255, 255), -1)
        
        cv2.imshow('Eye landmark', frame)

    if cv2.waitKey(1) == 27: # ESC키로 종료
            break
            
cap.release()
cv2.destroyAllWindows()