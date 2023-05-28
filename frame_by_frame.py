import cv2
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mediapipe as mp
import numpy as np
import tempfile
import glob
from yolov5.detect import run
from run_model import convert_mp4_to_jpg


def get_file_path(path):
    directory = os.path.dirname(path)
    return directory

def get_file_name(path):
    file_name = os.path.basename(path)
    return file_name

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points in degrees.

    Args:
        a (tuple): Coordinates of point a (ax, ay).
        b (tuple): Coordinates of point b (bx, by).
        c (tuple): Coordinates of point c (cx, cy).

    Returns:
        float: Angle between the line segments ab and bc in degrees.
    """
    # Calculate the angle using arctan2 and convert it to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize the angle to be within [0, 180] degrees
    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def process_frame_media_pipe(frame, frame_path):
    """
    Process a frame using MediaPipe to detect pose landmarks and calculate angles.

    Args:
        frame (numpy.ndarray): Input frame in BGR format.
        frame_path (str): Path to save the processed frame.

    Returns:
        numpy.ndarray: Processed frame with annotated angles and landmarks.
    """

    # Convert the frame to RGB
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)

        # Do something with the results
        if results.pose_landmarks:
            # Access pose landmarks and perform further analysis
            landmarks = results.pose_landmarks.landmark

            # Get landmark indices for elbow and knee
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate elbow angle
            elbow_angle = calculate_angle(
                (left_shoulder.x, left_shoulder.y),
                (left_elbow.x, left_elbow.y),
                (left_wrist.x, left_wrist.y)
            )

            # Calculate knee angle
            knee_angle = calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )

            # Draw angles on the frame
            cv2.putText(frame, f"Elbow Angle: {elbow_angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Knee Angle: {knee_angle:.2f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        file_name = os.path.basename(frame_path)
        image_path = os.path.join('/Users/igautam/Documents/GitHub/basketball-shot-analysis/temp',file_name)
        print(image_path)
        cv2.imwrite(image_path, frame)
    return frame,image_path

def process_frame_YOLOV5(image,frame_path):

    name = "temp"
    run(weights='./weights/basket_rim.pt', source=get_file_path(frame_path), conf_thres=0.1, project='/Users/igautam/Documents/GitHub/basketball-shot-analysis/', name=name)
    #image_files = glob.glob(image_path + "*.jpg")
    #image = cv2.imread(image_files)
    name = get_file_name(frame_path)
    
    shutil.rmtree('/Users/igautam/Documents/GitHub/basketball-shot-analysis/temp')
    return image

if __name__ == '__main__':

    # CHANGE HERE
    input_video_path = "./input/new2.mp4"
    image_folder_path = "./input/input_images"

    # Call the function to convert the MP4 file to JPG images
    image_paths = convert_mp4_to_jpg(input_video_path, image_folder_path)

    for filename in os.listdir(image_folder_path):
        # Check if the file has an image extension (e.g., .jpg, .png)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder_path, filename)
            image = cv2.imread(image_path)
            img,path_temp = process_frame_media_pipe(image,image_path)
            img = process_frame_YOLOV5(img,path_temp)
            cv2.imshow(img)
        break






