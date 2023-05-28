import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
#from .config import shooting_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from .utils import detect_shot, detect_image, detect_API, tensorflow_init
from statistics import mean
import mediapipe as mp
from yolov5.detect import run
#tf.disable_v2_behavior()


shooting_result = {
    'attempts': 0,
    'made': 0,
    'miss': 0,
    'avg_elbow_angle': 0,
    'avg_knee_angle': 0,
    'avg_release_angle': 0,
    'avg_ballInHand_time': 0
}

def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)

def tensorflow_init():
    MODEL_NAME = 'weights/inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    return detection_graph, image_tensor, boxes, scores, classes, num_detections

def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def trajectory_fit(balls, height, width, shotJudgement, fig):
    x = [ball[0] for ball in balls]
    y = [height - ball[1] for ball in balls]

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]   
    except:
        print("fitting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = [(a * (x_val ** 2)) + (b * x_val) + c for x_val in x_pos]

    if(shotJudgement == "MISS"):
        plt.plot(x, y, 'ro', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5, figure=fig)
    else:
        plt.plot(x, y, 'go', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5, figure=fig)

def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1/2)


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


def getAngleFromDatum(frame,frame_path):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            head = landmarks[mp_pose.PoseLandmark.NOSE]
            left_hand = landmarks[mp_pose.PoseLandmark.LEFT_HAND]
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_HAND]

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

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imwrite('temp.jpg', frame)

    return frame,head_x,head_y,left_hand_coordinates,right_hand_coordinates,elbow_angle,knee_angle,[left_elbow.x, left_elbow.y],[left_knee.x, left_knee.y]

def detect_shot(frame, trace, width, height, image_tensor, boxes, scores, classes, num_detections, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose):
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
        # print("ball in hand")

    # getting openpose keypoints


    try:
        frame,headX,headY,left_hand_coordinates,right_hand_coordinates,elbowAngle,kneeAngle,elbowCoord,kneeCoord = getAngleFromDatum(frame,'/Users/igautam/Documents/GitHub/basketball-shot-analysis/output')
        handX = left_hand_coordinates[0]
        handY = left_hand_coordinates[1]
    except:
        print("Something went wrong with Media Pipe")
        headX = 0
        headY = 0
        handX = 0
        handY = 0
        elbowAngle = 0
        kneeAngle = 0
        elbowCoord = np.array([0, 0])
        kneeCoord = np.array([0, 0])
    
    #frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    cv2.imwrite('temp.jpg', frame)
    '''
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    '''

    frame,boxes,scores,classes,num_detections = detect_API(frame,[])
    
    cv2.imwrite('temp3.jpg', frame)

    if(shot_result['release_displayFrames']):
        cv2.putText(frame, 'Release: ' + str(during_shooting['release_angle_list'][-1]) + ' deg',
                    (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] + 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 255), 3)


    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            # Basketball (not head)
            if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):
                print("Shooting Pose")
                # recording shooting pose
                if(distance([xCoor, yCoor], [handX, handY]) < 120):
                    shooting_pose['ball_in_hand'] = True
                    shooting_pose['knee_angle'] = min(
                        shooting_pose['knee_angle'], kneeAngle)
                    shooting_pose['elbow_angle'] = min(
                        shooting_pose['elbow_angle'], elbowAngle)
                else:
                    shooting_pose['ball_in_hand'] = False

                # During Shooting
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #calculating release angle
                    if(len(during_shooting['balls_during_shooting']) == 2):
                        first_shooting_point = during_shooting['balls_during_shooting'][0]
                        release_angle = calculateAngle(np.array(during_shooting['balls_during_shooting'][1]), np.array(
                            first_shooting_point), np.array([first_shooting_point[0] + 1, first_shooting_point[1]]))
                        if(release_angle > 90):
                            release_angle = 180 - release_angle
                        during_shooting['release_angle_list'].append(
                            release_angle)
                        during_shooting['release_point'] = first_shooting_point
                        shot_result['release_displayFrames'] = 30
                        print("release angle:", release_angle)

                    #draw purple circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)

                # Not shooting
                elif(ymin >= (previous['hoop_height'] - 30) and (distance([xCoor, yCoor], previous['ball']) < 100)):
                    # the moment when ball go below basket
                    print(" Not Shooting Pose")

                    if(during_shooting['isShooting']):
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):  # shot
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # draw green trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], False, color=(
                                82, 168, 50), thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(82, 168, 50), thickness=-1)
                        else:  # miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # draw red trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], color=(
                                0, 0, 255), isClosed=False, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(0, 0, 255), thickness=-1)

                        # reset all variables
                        trajectory_fit(
                            during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False
                        shooting_pose['ballInHand_frames_list'].append(
                            shooting_pose['ballInHand_frames'])
                        print("ball in hand frames: ",
                              shooting_pose['ballInHand_frames'])
                        shooting_pose['ballInHand_frames'] = 0

                        print("elbow angle: ", shooting_pose['elbow_angle'])
                        print("knee angle: ", shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle_list'].append(
                            shooting_pose['elbow_angle'])
                        shooting_pose['knee_angle_list'].append(
                            shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370

                    #draw blue circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)

                previous['ball'][0] = xCoor
                previous['ball'][1] = yCoor

            if(classes[0][i] == 2):  # Rim
                print("RIM")

                # cover previous hoop with white rectangle
                cv2.rectangle(
                    trace, (previous['hoop'][0], previous['hoop'][1]), (previous['hoop'][2], previous['hoop'][3]), (255, 255, 255), 5)
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)

                #display judgement after shot
                if(shot_result['displayFrames']):
                    if(shot_result['judgement'] == "MISS"):
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
                    else:
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (82, 168, 50), 8)

                previous['hoop'][0] = xmin
                previous['hoop'][1] = ymax
                previous['hoop'][2] = xmax
                previous['hoop'][3] = ymin
                previous['hoop_height'] = max(ymin, previous['hoop_height'])

    combined = np.concatenate((frame, trace), axis=1)
    return frame,combined, trace


def detect_image(img):
    response = []
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})
        valid_detections = 0

        for i, box in enumerate(boxes[0]):
            # print("detect")
            if (scores[0][i] > 0.5):
                valid_detections += 1
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    cv2.circle(img=img, center=(xCoor, yCoor), radius=25,
                               color=(255, 0, 0), thickness=-1)
                    cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                    print("add basketball")
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    cv2.rectangle(img, (xmin, ymax),
                                  (xmax, ymin), (48, 124, 255), 10)
                    cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    print("add hoop")
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
        
        if(valid_detections < 2):
            for i in range(2):
                response.append({
                    'class': 'Not Found',
                    'detection_detail': {
                        'confidence': 0.0,
                        'center_coordinate': {'x': 0, 'y': 0},
                        'box_boundary': {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
                    }
                })
            
    return img

def detect_API(img,response):

    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})

        for i, box in enumerate(boxes[0]):
            if (scores[0][i] > 0.5):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    cv2.circle(img=img, center=(xCoor, yCoor), radius=25,color=(255, 0, 0), thickness=-1)
                    cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                    print("add basketball")
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    cv2.rectangle(img, (xmin, ymax),(xmax, ymin), (48, 124, 255), 10)
                    cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    print("add hoop")
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })

    return img,boxes,scores,classes,num_detections

def getVideoStreams(video_path):
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
    output_file = 'output_video.mp4'

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    fig = plt.figure()
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370,
        'knee_angle': 370,
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

    # Load the video
    cap = cv2.VideoCapture(video_path)
    skip_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well, such as 'XVID'
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width,frame_height))

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        skip_count += 1
        if(skip_count < 4):
            continue
        skip_count = 0
        frame_3,detection, trace = detect_shot(frame, trace, width, height,image_tensor, boxes, scores, classes,
        num_detections, previous, during_shooting, shot_result, fig, None, None, shooting_pose)
        detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
        frame = cv2.imencode('.jpg', detection)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        output_video.write(frame_3)

        # Release the video capture and close windows

    # getting average shooting angle

    shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
    shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
    shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
    shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (4 / fps), 2)

    print("avg", shooting_result['avg_elbow_angle'])
    print("avg", shooting_result['avg_knee_angle'])
    print("avg", shooting_result['avg_release_angle'])
    print("avg", shooting_result['avg_ballInHand_time'])
    
    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0, top=height)
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg")
    fig.savefig(trajectory_path)
    fig.clear()
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def get_image(image_path, img_name, response):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    filename = img_name
    detection = detect_image(image, response)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))

def detectionAPI(response, image_path):
    image = cv2.imread(image_path)
    detect_API(response, image)

# getVideoStreams("/Users/igautam/Documents/GitHub/basketball-shot-analysis/static/two_score_three_miss.mp4")
# #img = cv2.imread("temp.jpg")
# #detect_API([],img)

print(detect_API(cv2.imread("temp.jpg", [])))