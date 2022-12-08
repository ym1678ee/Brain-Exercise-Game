# 원본 코드 https://google.github.io/mediapipe/solutions/hands.html


"""The 21 hand landmarks."""
# 손가락 위치 정의 참고 https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
#
# WRIST = 0
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4    엄지
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8  검지
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12  중지
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16  약지
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20  새끼


# 필요한 라이브러리
# pip install opencv-python mediapipe pillow numpy

import cv2 
import mediapipe as mp  
from PIL import ImageFont, ImageDraw, Image
import numpy as np      
import time

max_num_hands = 2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# For webcam input:
cap = cv2.VideoCapture(0)

level = 0
def next(level):
    level = level+1
    return level
if level == 0:
    text = "곰다리"
    print("레벨 0")
elif level == 1:
    print("레벨 1")     
elif level == 2:
    print("레벨 2")
elif level == 3:
    print("레벨 3")
elif level == 4:
    print("레벨 4")
elif level == 5:
    print("레벨 5")
elif level == 6:
    print("레벨 6")
elif level == 7: 
    print("레벨 7")
elif level == 8:
    print("레벨 8")
elif level == 9:
    print("레벨 9")
elif level == 10:
    print("레벨 10")
elif level == 11:
    print("레벨 11")
elif level == 12:
    print("레벨 11")
elif level == 13:
    print("레벨 13")

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

 
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")

      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    if results.multi_hand_landmarks != None:
      for hand in results.multi_handedness:
        #print(hand)
        #print(hand.classification)
        #print(hand.classification[0])
        handType=hand.classification[0].label
        index = hand.classification[0].index
        
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:


        # 엄지를 제외한 나머지 4개 손가락의 마디 위치 관계를 확인하여 플래그 변수를 설정합니다. 손가락을 일자로 편 상태인지 확인합니다.
        thumb_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
              thumb_finger_state = 1

        index_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
              index_finger_state = 1

        middle_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
              middle_finger_state = 1

        ring_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
              ring_finger_state = 1

        pinky_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
              pinky_finger_state = 1


        # 손가락 위치 확인한 값을 사용하여 가위,바위,보 중 하나를 출력 해줍니다.
        font = ImageFont.truetype("fonts/gulim.ttc", 80)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        
        text = ""
        result = ""

        t = ["곰다리", "네 개", "새다리", "두 개", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
        t[0] = "곰다리"
        t[1] = "네 개"
        t[2] = "새다리"
        t[3] = "두 개"
        t[4] = "새다리"
        t[5] = "두 개"
        t[6] = "곰다리"
        t[7] = "네 개"
        t[8] = "곰다리"
        t[9] = "네 개"
        t[10] = "새다리"
        t[11] = "두 개"
        t[12] = "합쳐서"
        t[13] =  "여섯개"
        #text = t[level]
        if level == 0:
            text = t[level]
            #text = "곰다리"
            if thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 1:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                level = next(level)
        elif level == 2:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                level = next(level)
        elif level == 3:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 4:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                level = next(level)
        elif level == 5:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 6:
            text = t[level]
            if thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 7:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                level = next(level)
        elif level == 8:
            text = t[level]
            if thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 9:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                level = next(level)
        elif level == 10:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
                level = next(level)
        elif level == 11:
            text = t[level]
            if thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 12:
            text = t[level]
            if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                level = next(level)
        elif level == 13:
            text = t[level]
            if thumb_finger_state == 1 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                level = next(level)
        elif level == 14:
            text = "성공!!"
                



        

          
      


        w, h = font.getsize(text)
        x = 50
        y = 50
        draw.rectangle((x, y, x + w, y + h), fill='black')
        draw.text((x, y),  text, font=font, fill=(255, 255, 255))
        image = np.array(image)
        


        # 손가락 뼈대를 그려줍니다.
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
      
      

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()