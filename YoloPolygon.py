import cv2

from ultralytics import YOLO,solutions


model = YOLO("/home/tts/Desktop/TTS_CV/YoloModelPractice/best.pt")

cap = cv2.VideoCapture("/home/tts/Desktop/TTS_CV/EJ_Hand_Tracking/generated_videos/banana_uyu_left.mp4")

assert cap.isOpened(), "Error reading video file"

w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))

#Define region points
region_points = [(20,400),(1080,404),(1080,360),(20,360),(20,400)]

#Video writer
video_writer = cv2.VideoWriter("object_counting_output_2.avi",cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))

#init object counter
counter = solutions.ObjectCounter(
    show = True,
    region = region_points,
    model = "yolo11n.pt",
)

#Proces video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully complted.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)


cap.release()
video_writer.release()
cv2.destroyAllWindows()


    