import cv2

from ultralytics import solutions

def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """ Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"
    w,h,fps = (int(cap.get(x))for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    line_points = [(20,400),(1080,400)]

    counter = solutions.ObjectCounter(show =True,region = line_points,model=model_path,classes = classes_to_count)

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



count_specific_classes("/home/tts/Desktop/TTS_CV/EJ_Hand_Tracking/generated_videos/1_new_fridge_left_15fps.mp4", "output_count_specific_classes.avi", "/home/tts/Desktop/TTS_CV/YoloModelPractice/best.pt", [0, 2])