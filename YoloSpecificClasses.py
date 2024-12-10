import cv2

from ultralytics import YOLO, solutions

model = YOLO("/home/tts/Desktop/TTS_CV/YoloModelPractice/best.pt")

cap = cv2.VideoCapture("/home/tts/Desktop/TTS_CV/EJ_Hand_Tracking/generated_videos/banana_uyu_left.mp4")

assert cap.isOpened(), "Error reading video file"
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cap.CAP_PROP_FPS))

#video writer
video_writer = cv2.VideoWriter("object_counting_output_3.avi", cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

#Init Object Counter
counter = solutions.ObjectCounter(
    show = True,
    model = "yolo11n.pt",
    classes =[0,1],
)

#Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

