import mediapipe as mp
import cv2
import numpy as np

def detect_ball(frame):

    # this is for detecting green balls
    lower_bound = np.array([30, 50, 50])  # Adjust based on your ball color
    upper_bound = np.array([60, 255, 255])  # Adjust based on your ball color

    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # create the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest_contour = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 10:
            return int(x), int(y), int(radius)
    return None

def detect_collision(ball_pos, palm_pos, threshold=200):
    distance = np.sqrt((ball_pos[0] - palm_pos[0]) ** 2 + (ball_pos[1] - palm_pos[1]) ** 2)
    return distance < threshold

def main():
    p1_score = 0
    p2_score = 0

    collision_state = False

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))

    size = (video_width, video_height)

    video = cv2.VideoWriter('project.mp4', -1, 30, size)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

        while cap.isOpened():

            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ball = detect_ball(frame)

            # stop copying
            img.flags.writeable = False
            results = hands.process(img)
            # allow copying
            img.flags.writeable = True
            # revert the image back
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # print(results)

            # checks to see if we have land marks
            if results.multi_hand_landmarks:
                # loop through land marks

                for num, hand in enumerate(results.multi_hand_landmarks):
                    # draw the marks
                    mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                    palm_position = hand.landmark[0]

                    # get the height and width of the shape
                    h, w, _ = frame.shape

                    palm_x = int(palm_position.x * w)
                    palm_y = int(palm_position.y * h)

                    # print(f"Hand {num + 1} Palm Position: X = {palm_x}, Y = {palm_y}")

                    if ball:
                        cv2.circle(img, (ball[0], ball[1]), ball[2]+25, (0, 255, 0), -1)  # Draw the detected ball
                        if detect_collision((ball[0], ball[1]), (palm_x, palm_y)):
                            if not collision_state:
                                if num == 0:
                                    p1_score += 1
                                    collision_state = True
                                else:
                                    p2_score += 1
                                    collision_state = True
                            else:
                                collision_state = False
                        
            # Display the scores
            cv2.putText(img, f'Player 1 Score: {p1_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Player 2 Score: {p2_score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            video.write(img)
            cv2.imshow("Hand Tracking", img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()