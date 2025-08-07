import numpy as np

def eye_aspect_ratio(eye):
    # 세로 거리
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # 가로 거리
    C = np.linalg.norm(eye[0] - eye[3])

    EAR = (A + B) / (2.0 * C)
    return EAR