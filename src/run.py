import mediapipe as mp
from yolov5.detect import run
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import os
import scipy.optimize as optimize

def ball_under_basket(bball, rim, threshold):
    ball_x = (bball[0] + bball[2]) / 2
    rim_x = (rim[0] + rim[2]) / 2
    ball_y = (bball[1] + bball[3]) / 2
    rim_y = (rim[1] + rim[3]) / 2
    print(abs(ball_x - rim_x), ball_y - rim_y)
    if abs(ball_x - rim_x) < threshold and ball_y - rim_y > 0:
        return True
    else:
        return False

def get_tangent_angle(a, b):
    a_x_mean = (a[0] + a[2]) / 2
    a_y_mean = (a[1] + a[3]) / 2
    b_x_mean = (b[0] + b[2]) / 2
    b_y_mean = (b[1] + b[3]) / 2
    x = (a_x_mean, a_y_mean)
    y = (b_x_mean, b_y_mean)
    z = (b_x_mean, a_y_mean)
    ang = calculate_angle(x,y,z)
    if ang > 90:
        ang = abs(90-ang)
    return ang
    
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

def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1/2)

def delete_folder_contents(folder_path):
    # Iterate over all the files and subdirectories in the given folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # If the item is a file, delete it
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # If the item is a subdirectory, recursively call the function to delete its contents
            delete_folder_contents(file_path)
            # After deleting the subdirectory's contents, remove the subdirectory itself
            os.rmdir(file_path)

def get_angles_postions(frame):
    #write frame
    # cv2.imwrite('frame.jpg', frame)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # Process the frame with MediaPipe
    with mp_pose.Pose(min_detection_confidence=0.05, min_tracking_confidence=0.05) as pose:
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
            head = landmarks[mp_pose.PoseLandmark.NOSE]
            left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]


            # left_shoulder = [left_shoulder.x, left_shoulder.y]

            left_hand_x = left_hand.x
            left_hand_y = left_hand.y

            # Retrieve right hand coordinates
            right_hand_x = right_hand.x
            right_hand_y = right_hand.y

            # Store left and right hand coordinates in the respective lists
            left_hand_coordinates = [left_hand_x, left_hand_y]
            right_hand_coordinates = [right_hand_x, right_hand_y]

            head_x = head.x
            head_y = head.y


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
        else:
            return None

        # Draw pose landmarks on the frame
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imwrite('temp.jpg', frame)

    return frame,head_x,head_y,left_hand_coordinates,right_hand_coordinates,elbow_angle,knee_angle,[left_elbow.x, left_elbow.y],[left_knee.x, left_knee.y], left_shoulder

def release_start(boxes, classes, y_elbow):
    if 0 not in classes[0]:
        return False
    if classes[0][0] == 0:
        b_index = 0
    else:
        b_index = 1
    y_box = boxes[b_index][3]
    print(y_box, y_elbow)
    if y_box < y_elbow:
        return True
    else:
        return False
    
def ball_near_body(boxes, classes, right_hand_coordinates, left_hand_coordinates, distance_threshold):
    if 0 not in classes[0]:
        return False
    if classes[0][0] == 0:
        b_index = 0
    else:
        b_index = 1
    box = boxes[b_index]
    if distance(box, right_hand_coordinates) < distance_threshold or distance(box, left_hand_coordinates) < distance_threshold:
        return True
    else:
        return False
    
def find_suitable_ball(bball_coords):
    if len(bball_coords) == 0 or len(bball_coords) == 1:
        return None
    else:
        i = len(bball_coords) - 2
        while i >= 0:
            if bball_coords[i] is not None and bball_coords[i][0] is not None:
                return bball_coords[i]
            i -= 1
    
def release_end(boxes, classes, left_hand_coordinates, distance_threshold):
    if 0 not in classes[0]:
        return False
    if classes[0][0] == 0:
        b_index = 0
    else:
        b_index = 1
    box = boxes[b_index]
    if distance(box, left_hand_coordinates) > distance_threshold: #or distance(box, left_hand_coordinates) > distance_threshold:
        return True
    else:
        return False

def trajectory_fit(shot_tracking, height, width, folder_path):

    if os.path.exists(folder_path):
        file_list = os.listdir(folder_path)
        if len(file_list) > 0:
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)

    print("SHOT TRACKING", len(shot_tracking))
    counter = 1
    for shot in shot_tracking:
        # if shot == len(shot_tracking) - 1:
        #     break
        print("RELEASE FRAMES:", shot_tracking[shot]['release_frames'], len(shot_tracking[shot]['bball']),
              len(shot_tracking[shot]["release_tracking"]))
        balls = shot_tracking[shot]['bball']
        rims = shot_tracking[shot]['rim']
        release_tracking = shot_tracking[shot]['release_tracking']
        print("RELEASE TRACKING:", release_tracking)
        trace = np.full((height, width, 3), 255, np.uint8)
        x_data = []
        y_data = []
        for i in range(len(balls)):
            box = balls[i]
            if box is not None and release_tracking[i] != False:
                x_min, y_min, x_max, y_max = box
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                radius = min((x_max - x_min) // 2, (y_max - y_min) // 2)
                cv2.circle(trace, (center_x, center_y), radius, (0, 0, 255), 2)  
                x_data.append(center_x)
                y_data.append(center_y)

        # Perform curve fitting
        def curve_func(x, a, b, c):
            return a * x**2 + b * x + c

        if(len(x_data) > 0 and len(y_data) > 0):

            params, _ = optimize.curve_fit(curve_func, x_data, y_data)

            # Generate curve points
            curve_x = np.linspace(min(x_data), max(x_data), 100)
            curve_y = curve_func(curve_x, *params)

            # Plot curve
            for i in range(len(curve_x) - 1):
                cv2.line(trace, (int(curve_x[i]), int(curve_y[i])), (int(curve_x[i + 1]), int(curve_y[i + 1])), (0, 0, 255), 2)

        for box in rims:
            if box is not None:
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(trace, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                break

        text = shot_tracking[shot]['result']
        org = (10, 30)  # Coordinates of the bottom-left corner of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)  # Blue color for the text
        thickness = 2
        cv2.putText(trace, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        file_name = "trace_with_balls_{}.jpg".format(counter)
        cv2.imwrite(folder_path + file_name, trace)
        counter = counter + 1

def getVideoStreams(video_path):

    shooting_time = []
    release_angle = []
    make_or_miss = []
    knee_angles = []
    elbow_angles = []

    output_file = '../output/output_video.mp4'

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 10

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    coords_tracking = {
        "bball": [],
        "rim": [],
        "distances": []
    }

    shot_tracking = {
        1: {
            "bball": [],
            "rim": [],
            "result": None,
            "release_frames": 0,
            "release_tracking": []
        }
    }

    shot_number = 1
    coords_tracking["bball"] = []

    # Load the video
    cap = cv2.VideoCapture(video_path)
    skip_count = 0

    runs_folder = "../runs/"

    delete_folder_contents(runs_folder)
    delete_folder_contents('../output/traces/')

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    output_video = cv2.VideoWriter(output_file, fourcc, fps/2, (frame_width,frame_height))

    release_started = False
    tracking_shot = False

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        skip_count += 1
        if(skip_count < 4):
            continue
        skip_count = 0
        # cv2.imwrite('temp.jpg', frame)
        out = get_angles_postions(frame)
        if out is not None:
            frame,head_x,head_y,left_hand_coordinates,right_hand_coordinates,elbow_angle,knee_angle,elbow_coo,knee_coo, left_shoulder = out
        img,boxes,scores,classes, height, width = detect_API(frame, 'temp.jpg',[])
        if 0 not in classes[0]:
            coords_tracking["bball"].append(None)
            shot_tracking[shot_number]["bball"].append(None)
            #shot_tracking[shot_number]["release_tracking"].append(False)
        elif classes[0][0] == 0:
            b_index = 0
            coords_tracking["bball"].append(boxes[0])
            shot_tracking[shot_number]["bball"].append(boxes[0])
        else:
            b_index = 1
            coords_tracking["bball"].append(boxes[1])
            shot_tracking[shot_number]["bball"].append(boxes[1])

        if 2 not in classes[0]:
            coords_tracking["rim"].append(None)
            shot_tracking[shot_number]["rim"].append(None)
        elif classes[0][0] == 2:
            b_index = 0
            coords_tracking["rim"].append(boxes[0])
            shot_tracking[shot_number]["rim"].append(boxes[0])
        else:
            b_index = 1
            coords_tracking["rim"].append(boxes[1])
            shot_tracking[shot_number]["rim"].append(boxes[1])

        if coords_tracking["rim"][-1] is not None and coords_tracking["bball"][-1] is not None:
            coords_tracking["distances"].append(distance(coords_tracking["rim"][-1], coords_tracking["bball"][-1]))


        left_hand_coordinates[0] *= width
        left_hand_coordinates[1] *= height
        right_hand_coordinates[0] *= width
        right_hand_coordinates[1] *= height
        if not release_started:
            shot_tracking[shot_number]["release_tracking"].append(False)
            release_started = release_start(boxes, classes, left_shoulder.y*height)
            if release_started and ball_near_body(boxes, classes, right_hand_coordinates, left_hand_coordinates, 100):
                cv2.putText(frame, "RELEASE STARTED", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                release_ended = False
                knee_angle_min = 366
                elbow_angle_min = 366
            else:
                release_started = False
        elif release_started and not release_ended:
            shot_tracking[shot_number]["release_frames"] = shot_tracking[shot_number]["release_frames"] + 1
            shot_tracking[shot_number]["release_tracking"].append(True)
            release_ended = release_end(boxes, classes, right_hand_coordinates, 120)
            if knee_angle > 90:
                knee_angle_min = min(knee_angle_min, knee_angle)
            if elbow_angle > 90:
                elbow_angle_min = min(elbow_angle_min, elbow_angle)
            if release_ended:
                cv2.putText(frame, "RELEASE ENDED", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                release_ended = True
                tracking_shot = True
                if len(coords_tracking["bball"]) > 1:
                    angle = get_tangent_angle(coords_tracking["bball"][-1], find_suitable_ball(coords_tracking["bball"]))
                    release_angle.append(angle)
                    cv2.putText(frame, "ANGLE SHOT: {}".format(round(angle,2)), (int(width/4), int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        elif release_started and release_ended and tracking_shot:
            if coords_tracking["rim"][-1] is not None:
                rimm = find_suitable_ball(coords_tracking["rim"])
            else:
                rimm = find_suitable_ball(coords_tracking["rim"])

            if coords_tracking["bball"][-1] is not None and rimm is not None and ball_under_basket(coords_tracking["bball"][-1], rimm, 100):
                    make_or_miss.append('Make')
                    cv2.putText(frame, "SCORE", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                    shot_tracking[shot_number]["result"] = "Make"
                    shot_tracking[shot_number]["release_tracking"].append(False)
                    shot_number = shot_number + 1
                    shot_tracking[shot_number] = {
                                                    "bball": [],
                                                    "rim": [],
                                                    "result": None,
                                                    "release_frames": 0,
                                                    "release_tracking": []
                                                }
                    tracking_shot = False
                    knee_angles.append(knee_angle_min)
                    elbow_angles.append(elbow_angle_min)
            elif coords_tracking["distances"][-1] > coords_tracking["distances"][-2] and coords_tracking["bball"][-1] is not None and coords_tracking["rim"][-1] is not None:
                shot_tracking[shot_number]["release_tracking"].append(False)
                # cv2.putText(frame, "BALL MOVING AWAY", (int(width/4), int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                tracking_shot = False
                if ball_under_basket(coords_tracking["bball"][-1], coords_tracking["rim"][-1], 70):
                    make_or_miss.append('Make')
                    cv2.putText(frame, "SCORE", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                    shot_tracking[shot_number]["result"] = "Make"
                    shot_number = shot_number + 1
                    shot_tracking[shot_number] = {
                                                    "bball": [],
                                                    "rim": [],
                                                    "result": None,
                                                    "release_frames": 0,
                                                    "release_tracking": []
                                                }
                    knee_angles.append(knee_angle_min)
                    elbow_angles.append(elbow_angle)
                else:
                    make_or_miss.append('Miss')
                    cv2.putText(frame, "MISS", (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                    shot_tracking[shot_number]["result"] = "Miss"
                    shot_number = shot_number + 1
                    shot_tracking[shot_number] = {
                                                    "bball": [],
                                                    "rim": [],
                                                    "result": None,
                                                    "release_frames": 0,
                                                    "release_tracking": []
                                                }
                    knee_angles.append(knee_angle_min)
                    elbow_angles.append(elbow_angle)
            else:
                shot_tracking[shot_number]["release_frames"] = shot_tracking[shot_number]["release_frames"] + 1
                shot_tracking[shot_number]["release_tracking"].append(True)
        else:
            shot_tracking[shot_number]["release_tracking"].append(False)

        if release_started and release_ended and ball_near_body(boxes, classes, right_hand_coordinates, left_hand_coordinates, 50):
            shot_tracking[shot_number]["release_tracking"].append(False)
            release_started = False
            release_ended = False
            tracking_shot = False
            coords_tracking["bball"] = []
            coords_tracking["rim"] = []
            coords_tracking["distances"] = []
            cv2.putText(frame, "RESET", (int(width/1.5), int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
        output_video.write(frame)
    trajectory_fit(shot_tracking,height,width,'../output/traces/')
    os.remove("temp.jpg")
    # delete_folder_contents(runs_folder)
    output_video.release()
    for i in range(len(release_angle)):
        shooting_time.append(shot_tracking[i+1]['release_frames'])
    elbow_angles = [x-90 for x in elbow_angles]
    return shooting_time[0:len(make_or_miss)],release_angle[0:len(make_or_miss)], make_or_miss, knee_angles, elbow_angles, fps

def detect_API(img, img_path,response):

    height, width = img.shape[:2]
        
    runs_folder = "../runs/"
    name = "run"

    boxes, scores, classes, n = run(weights='../weights/basket_rim.pt', source=img_path, conf_thres=0.1, project=runs_folder, name=name, classes=[0,2])

    scores = scores[0]
    classes = classes[0]
    boxes = boxes[::-1]

    if len(scores[0]) != 0:
        max_0 = 0
        max_2 = 0
        for i in range(len(scores[0])):
            if(classes[0][i] == 0 and scores[0][i] > max_0):
                max_0 = scores[0][i]
            if(classes[0][i] == 2 and scores[0][i] > max_2):
                max_2 = scores[0][i]
        if max_0 == 0:
            index_2 = np.where(scores[0] == max_2)[0][0]
            scores = np.array([[scores[0][index_2]]])
            classes = np.array([[classes[0][index_2]]])
            boxes = np.array([boxes[index_2]])
        elif max_2 == 0:
            index_0 = np.where(scores[0] == max_0)[0][0]
            scores = np.array([[scores[0][index_0]]])
            classes = np.array([[classes[0][index_0]]])
            boxes = np.array([boxes[index_0]])
        else:
            index_0 = np.where(scores[0] == max_0)[0][0]
            index_2 = np.where(scores[0] == max_2)[0][0]
            scores = np.array([[scores[0][index_0],scores[0][index_2]]])
            classes = np.array([[classes[0][index_0],classes[0][index_2]]])
            boxes = np.array([boxes[index_0],boxes[index_2]])

    for i, box in enumerate(boxes):
        if (scores[0][i] > 0.055):
            ymin = int(box[1])
            xmin = int(box[0])
            ymax = int(box[3])
            xmax = int(box[2])
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))

            if(int(classes[0][i]) == 0):  # basketball
                cv2.circle(img=img, center=(xCoor, yCoor), radius=25,color=(255, 0, 0), thickness=-1)
                # cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
            if(int(classes[0][i]) == 2):  # Rim
                cv2.rectangle(img, (xmin, ymax),(xmax, ymin), (48, 124, 255), 10)
                # cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 2)


    return img,boxes,scores,classes, height, width

if __name__ == '__main__':
    print(getVideoStreams('/Users/simonzouki/Documents/Northwestern/Q3/CV/project/basketball-shot-analysis/input/IMG_9136.mp4'))