# 3-2. EAR값을 그래프로 실시간 표시하기

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# EAR 값 저장용 deque (최대 100개)
EAR_history = deque(maxlen=100)

plt.ion() # 인터랙티브 모드 켜기
fig, ax = plt.subplots() # 그래프(figure)과 좌표축(axes)생성
# 선 그리기
line, = ax.plot([], [], label='EAR') 
threshold_line = ax.axhline(y=0.25, color='r', linestyle='--', label='Threshold 0.25')
ax.set_ylim(0, 0.5)
ax.set_xlim(0, 100)
ax.set_xlabel('Frame')
ax.set_ylabel('EAR')
ax.legend()

    for rect in face:
        # EAR 게산
        # left_eye = eye_aspect_ratio(Left_eye_pts)
        # right_eye = eye_aspect_ratio(Right_eye_pts)
        # EAR = (left_eye + right_eye) / 2.0

        # EAR 값 표시
        # cv2.putText(frame, f'EAR: {EAR:.3f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 그래프 업데이트
        EAR_history.append(EAR)
        line.set_data(range(len(EAR_history)), list(EAR_history))
        ax.set_xlim(0, max(100, len(EAR_history)))
        fig.canvas.draw()
        fig.canvas.flush_events()

    cv2.imshow('Eye EAR', frame)

plt.ioff()
plt.show()