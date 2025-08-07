# 3-1. EAR 계산 및 실시간 화면 표시
import cv2
import dlib
import numpy as np

# EAR 계산 함수
def eye_aspect_ratio(eye):
    # 세로 거리
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # 가로 거리
    C = np.linalg.norm(eye[0] - eye[3])

    EAR = (A + B) / (2.0 * C)
    return EAR

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

        # EAR 게산
        left_eye = eye_aspect_ratio(Left_eye_pts)
        right_eye = eye_aspect_ratio(Right_eye_pts)
        EAR = (left_eye + right_eye) / 2.0

        # EAR 값 표시
        cv2.putText(frame, f'EAR: {EAR:.3f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    cv2.imshow('Eye center point', frame)

    if cv2.waitKey(1) == 27: # ESC키로 종료
            break
            
cap.release()
cv2.destroyAllWindows()