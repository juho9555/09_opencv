# 1. 랜드마크 검출기 기반
import cv2
import dlib

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0) # 웹캠 인식

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('no frame.');break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출
    faces = detector(gray)

    for rect in faces:
        
        # 얼굴 랜드마크 검출
        shape = predictor(gray, rect)
        

cv2.imshow("~~ landmark", frame)
cv2.waitKey(0)