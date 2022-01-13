import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HistoryManager():

    def __init__(self):
        self.history = []
        self.num_history = 30

        # graph init
        plt.figure(figsize=(8, 8))
        self.ax = plt.subplot(1, 1, 1)
        self.ax.set_xlim([0, self.num_history])
        self.ax.set_ylim([0, 0.15])

    def update(self, v):
        """
        座標情報を最大格納数分だけ格納
        """
        self.history.append(v)
        if len(self.history) > self.num_history:
            self.history.pop(0)

    def plot(self):
        """
        グラフ描画（リアルタイム）
        """
        self.ax.cla()
        self.ax.plot([v for v in self.history])
        self.ax.set_xlim([0, self.num_history])
        self.ax.set_ylim([0, 0.15])
        plt.pause(0.01)
        

history_manager = HistoryManager()


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Print hand world landmarks.
    if results.multi_hand_world_landmarks:
      for hand_world_landmarks in results.multi_hand_world_landmarks:

        x = hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        y = hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        z = hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
        dist = (x**2 + y**2 + z**2)**(0.5)
        print(
            f'x={x}, '
            f'y={y}, '
            f'z={z}, '
            f'dist={dist}'
        )
        history_manager.update(dist)
        history_manager.plot()


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()