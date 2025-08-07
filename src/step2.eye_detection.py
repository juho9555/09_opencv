# 2. 눈 랜드마크 검출하기
import cv2
import dlib
import numpy as np

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

        # 왼쪽 눈 좌표 추출
        Left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in Left_eye])

        # 오른쪽 눈 좌표 추출
        Right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in Right_eye])

        # 눈 윤곽선 그리기
        cv2.polylines(frame, [Left_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [Right_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)

        # 왼쪽 눈 바운딩 박스 그리기
        lx, ly = np.min(Left_eye_pts, axis=0)
        rx, ry = np.max(Left_eye_pts, axis=0)
        cv2.rectangle(frame, (lx, ly), (rx, ry), (0, 255, 0), 1)

        # 오른쪽 눈 바운딩 박스 그리기
        lx, ly = np.min(Right_eye_pts, axis=0)
        rx, ry = np.max(Right_eye_pts, axis=0)
        cv2.rectangle(frame, (lx, ly), (rx, ry), (0, 255, 0), 1)

        # 왼쪽 눈 중심점 계산 및 표시
        center_left = np.mean(Left_eye_pts, axis=0).astype(int)
        cv2.circle(frame, tuple(center_left), 2, (255, 0, 0), -1) # 파란 점

        # 오른쪽 눈 중심점 계산 및 표시
        center_right = np.mean(Right_eye_pts, axis=0).astype(int)
        cv2.circle(frame, tuple(center_right), 2, (255, 0, 0), -1) # 파란 점


        cv2.imshow('Eye center point', frame)

    if cv2.waitKey(1) == 27: # ESC키로 종료
            break
            
cap.release()
cv2.destroyAllWindows()