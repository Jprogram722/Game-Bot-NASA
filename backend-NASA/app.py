from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import uvicorn
import cv2
import mediapipe as mp
import base64

app = FastAPI()

# allow cors between the browser an the api
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def turn_on_camera() -> cv2.VideoCapture:
    """
    This function turns the camera on
    >>> turn_on_camera()
    >>> cv2.VideoCapture
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error")
        return None
    return cap

def getFrame():
    """
    This function returns the frames that were gather from the video from the webcam.
    by using the yeild instead of return the function won't stop after sending data
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            else:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)
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
                _, buffer = cv2.imencode(".jpg", img)
                img = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+img+b'\r\n')

@app.get("/api/video")
async def stream():
    # if not video_active:
    #     return Response("Feed Not Running", status_code=200)
    return StreamingResponse(
        getFrame(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8080, reload=True)