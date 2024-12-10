import cv2

from ultralytics import solutions

def count_objects_in_region(video_path, output_video_path,model_path):
    """count objects in a specific region within a video"""
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    region_points = [(20,400),(1080,404),(1080,360),(20,360)]

    counter = solutions.ObjectCounter(show = True, region = region_points, model = model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("/home/tts/Desktop/TTS_CV/EJ_Hand_Tracking/generated_videos/1_new_fridge_left_15fps.mp4", "output_count_specific_classes.avi", "/home/tts/Desktop/TTS_CV/YoloModelPractice/best.pt")