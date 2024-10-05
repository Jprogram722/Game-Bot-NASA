from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import uvicorn
import cv2
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
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            continue
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

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