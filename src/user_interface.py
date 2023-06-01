import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from run import getVideoStreams
import os
import subprocess
from moviepy.editor import VideoFileClip
import glob
    
def convert_to_mp4(input_path, output_path):
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec='libx264')

def process_video(video_path):
    shooting_time, release_angle, make_or_miss, knee_angles, elbow_angles, fps = getVideoStreams(video_path)
    return shooting_time, release_angle, make_or_miss, knee_angles, elbow_angles, fps

# Function to display the processed video
def display_video(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        st.image(frame, channels="BGR")

# Function to display a chart
def display_chart():
    data = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [10, 20, 30]})
    fig, ax = plt.subplots()
    ax.bar(data['Category'], data['Values'])
    st.pyplot(fig)

# Function to display a table
def display_table(shooting_time, release_angle, make_or_miss, knee_angles, elbow_angles, fps):
    shooting_time = [i/fps for i in shooting_time]
    # Create a sample DataFrame
    shooting_time = np.round(shooting_time,3)
    release_angle = np.round(release_angle,3)
    knee_angles = np.round(knee_angles,3)
    elbow_angles = np.round(elbow_angles,3)

    data = {
        'Shooting Time (Sec)': shooting_time,
        'Release Angle (Deg)': release_angle,
        'Make or Miss': make_or_miss,
        'Knee Angle (Deg)': knee_angles,
        'Elbow Angle (Deg)': elbow_angles
    }

    print(len(release_angle),len(make_or_miss))
    df = pd.DataFrame(data)
    make_count = make_or_miss.count('Make')
    total_shots = len(make_or_miss)
    make_percentage = (make_count / total_shots) * 100
    mean_shooting_time = np.mean(shooting_time)
    mean_release_angle = np.mean(release_angle)
    mean_knee_angle = np.mean(knee_angles)
    mean_elbow_angle = np.mean(elbow_angles)
    average_row = [str(np.round(mean_shooting_time,2))+' sec',str(np.round(mean_release_angle,2)) + " Deg", str(np.round(make_percentage)) + '% Shooting Percent', str(np.round(mean_knee_angle,2)) + ' Deg', str(np.round(mean_elbow_angle,2)) + ' Deg']
    #average_row = ['Avg ' + str(np.round(mean_release_angle,2)) + ' Deg',str(make_percentage) + '% Shooting Percent']
    new_row_df = pd.DataFrame([average_row], columns=df.columns)
    df_with_new_row = pd.concat([df, new_row_df], ignore_index=True)
    indices = []
    for i in range(len(release_angle)):
        indices.append('Shot '+str(i+1))
    indices.append('Averages')
    df_with_new_row.index = indices
    st.table(df_with_new_row)

# Function to display a table
def display_table_comparison(metrics_A, metrics_B):
    # Create a sample DataFrame
    metrics_A[0] = [i/metrics_A[5] for i in metrics_A[0]]
    metrics_B[0] = [i/metrics_B[5] for i in metrics_B[0]]


    make_count_A = metrics_A[2].count('Make')
    total_shots_A = len(metrics_A[2])
    make_percentage_A = (make_count_A / total_shots_A) * 100
    avg_shooting_time_A = np.round(np.mean(metrics_A[0]),2)
    avg_releasing_angle_A = np.round(np.mean(metrics_A[1]),2)
    avg_knee_angle_A = np.round(np.mean(metrics_A[3]),2)
    avg_elbow_angle_A = np.round(np.mean(metrics_A[4]),2)

    make_count_B = metrics_B[2].count('Make')
    total_shots_B = len(metrics_B[2])
    make_percentage_B = (make_count_B / total_shots_B) * 100
    avg_shooting_time_B = np.round(np.mean(metrics_B[0]),2)
    avg_releasing_angle_B = np.round(np.mean(metrics_B[1]),2)
    avg_knee_angle_B = np.round(np.mean(metrics_B[3]),2)
    avg_elbow_angle_B = np.round(np.mean(metrics_B[4]),2)

    data = {
        'Average Shooting Time (Sec)': [avg_shooting_time_A, avg_shooting_time_B],
        'Average Release Angle (Deg)': [avg_releasing_angle_A, avg_releasing_angle_B],
        'Shooting%': [make_percentage_A, make_percentage_B],
        'Average Knee Angle (Deg)': [avg_knee_angle_A, avg_knee_angle_B],
        'Average Elbow Angle (Deg)': [avg_elbow_angle_A, avg_elbow_angle_B]
    }
    df = pd.DataFrame(data)
    df.index = ['Player','Comparison']
    st.table(df)

# Main function to run the Streamlit app
def main():
    st.title('Shooting Analysis Application')

    # Create tabs
    tabs = ['Shooting Analysis', 'Player Comparison']
    selected_tab = st.sidebar.selectbox('Select a tab', tabs)

    if selected_tab == 'Shooting Analysis':
        # Upload video file
        video_file = st.file_uploader('Upload your shooting video', type=['mp4', 'mov'])

        if video_file is not None:
            # Save uploaded video file
            video_path = '../input/uploaded_video.mp4'
            if os.path.exists(video_path):
                os.remove(video_path)
            with open(video_path, 'wb') as f:
                f.write(video_file.read())

            # Perform video postprocessing
            with st.spinner('Processing video...'):
                shooting_time, release_angle, make_or_miss, knee_angles, elbow_angles, fps = process_video(video_path)

            # Display processed video
            #convert_to_mp4('../output/output_video.mp4','../output/output_video2.mp4')
            input_path = '../output/output_video.mp4'

            st.subheader('Processed Video')
            st.video(input_path)

            st.subheader('Table')
            display_table(shooting_time, release_angle, make_or_miss, knee_angles, elbow_angles, fps)

        if(video_file is not None):
            st.subheader("Shot Traces")
            folder_path = '../output/traces'
            image_pattern = '*.jpg'
            image_files = glob.glob(f"{folder_path}/{image_pattern}")
            #current_index = st.session_state.get('current_index', 0)

            # Display frame for the current photo
            #frame = st.empty()
            #frame.image(image_files[current_index], use_column_width=True)

            # Create a button to iterate through the images
            #if st.button('Next'):
            #    current_index = (current_index + 1) % len(image_files)
            #    frame.image(image_files[current_index], use_column_width=True)

            #st.session_state['current_index'] = current_index


            #image_urls = [f"file://{file_path}" for file_path in image_files]
            #print(image_urls)
            # Initialize index to keep track of the current image
            for url in image_files:
                st.image(url,use_column_width=True)

            
            
    elif selected_tab == 'Player Comparison':
        # Upload video file
        video_player_file = st.file_uploader('Upload your shooting video', type=['mp4', 'mov'])
        video_comparison_file = st.file_uploader('Upload the comparison video', type=['mp4', 'mov'])

        if video_player_file is not None and video_comparison_file is not None:
            # Save uploaded video file
            video_player_path = '../output/video_player_file.mp4'
            with open(video_player_path, 'wb') as f:
                f.write(video_player_file.read())
            
            # Save uploaded video file
            video_comparison_path = '../output/video_comparison_file.mp4'
            with open(video_comparison_path, 'wb') as f:
                f.write(video_comparison_file.read())

            # Perform video postprocessing
            with st.spinner('Processing video...'):
                shooting_time_p, release_angle_p, make_or_miss_p, knee_angles_p, elbow_angles_p, fps_p = process_video(video_player_path)
                convert_to_mp4('../output/output_video.mp4','../output/output_comparison/output_video_player.mp4')
                shooting_time_c, release_angle_c, make_or_miss_c, knee_angles_c, elbow_angles_c, fps_c = process_video(video_comparison_path)
                convert_to_mp4('../output/output_video.mp4','../output/output_comparison/output_video_comparison.mp4')


            # Display two videos side by side
            col1, col2 = st.columns(2)

            # Video 1
            with col1:
                st.subheader('Processed Video 1')
                st.video('../output/output_comparison/output_video_player.mp4')

            # Video 2
            with col2:
                st.subheader('Processed Video 2')
                st.video('../output/output_comparison/output_video_comparison.mp4')

            st.subheader('Comparison Table')
            display_table_comparison(
                                     [shooting_time_p, release_angle_p, make_or_miss_p, knee_angles_p, elbow_angles_p, fps_p],
                                     [shooting_time_c, release_angle_c, make_or_miss_c, knee_angles_c, elbow_angles_c, fps_c]
                                     )
if __name__ == '__main__':
    main()
