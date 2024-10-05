import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
player1_score = 0
player2_score = 0
ball_position = None  # Position of the ball
ball_radius = 15  # Radius of the ball for detection

# Define the webcam feed
cap = cv2.VideoCapture(0)

# Function to detect hands
def detect_hands(frame):
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        return results

# Function to draw hands and determine palm positions
def draw_hands(frame, results):
    palms = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            palm_position = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            palms.append((int(palm_position.x * frame.shape[1]), int(palm_position.y * frame.shape[0])))
    return palms

# Function to detect the ball using color
def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the ball color
    lower_bound = np.array([30, 50, 50])  # Adjust based on your ball color
    upper_bound = np.array([60, 255, 255])  # Adjust based on your ball color

    # Create a mask based on the HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours of the detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > ball_radius:  # Only consider if the radius is significant
            return (int(x), int(y))
    return None

# Main game loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    results = detect_hands(frame)

    palms = draw_hands(frame, results)

    # Detect the ball
    ball_position = detect_ball(frame)

    # If the ball is detected
    if ball_position:
        for idx, palm in enumerate(palms):
            # Check if palm is close to the ball
            if palm and (ball_position[0] - ball_radius < palm[0] < ball_position[0] + ball_radius) and \
                    (ball_position[1] - ball_radius < palm[1] < ball_position[1] + ball_radius):
                # Update scores based on which palm is detected
                if idx == 0:  # Player 1 catches the ball
                    player1_score += 1
                    cv2.putText(frame, f'Player 1 Score: {player1_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif idx == 1:  # Player 2 catches the ball
                    player2_score += 1
                    cv2.putText(frame, f'Player 2 Score: {player2_score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Simulate passing the ball after catching
                # Move the ball off-screen (for demonstration)
                ball_position = (-100, -100)  # Reset ball position after catch
                break  # Exit loop after catching the ball to prevent double counting

    # Draw the ball if its position is valid
    if ball_position is not None and ball_position[0] != -100:  # Check if ball_position is not None
        cv2.circle(frame, ball_position, ball_radius, (0, 255, 0), -1)  # Draw the detected ball

    # Display the scores
    cv2.putText(frame, f'Player 1 Score: {player1_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Player 2 Score: {player2_score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Palm Detection Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
