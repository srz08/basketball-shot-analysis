import cv2
import os
from yolov5.detect import run
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def convert_mp4_to_jpg(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    success = True
    images = []

    # Read frames from the video and save them as JPG images
    while success:
        success, frame = video.read()
        if success:
            output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            images.append(output_path)
            frame_count += 1

    # Release the video capture object
    video.release()

    return images

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


def get_files(directory):
    files = os.listdir(directory)
    files.sort()
    return files

def generate_video(base_img, input_directory, output_video_path):
    w = base_img.shape[1]
    h = base_img.shape[0]
    video = cv2.VideoWriter('{}.mp4'.format(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), 40, (w, h))

    files = get_files(input_directory)
    for file in files:
        print(file)
        img_name = "{}".format(file)
        new_img = cv2.imread(input_directory + '/{}'.format(img_name))
        video.write(new_img)

    cv2.destroyAllWindows()
    video.release()

# CHANGE HERE
input_video_path = "./input/new2.mp4"
image_folder_path = "./input/input_images"

# Call the function to convert the MP4 file to JPG images
image_paths = convert_mp4_to_jpg(input_video_path, image_folder_path)


runs_folder = "./runs/"
name = "run"

delete_folder_contents(runs_folder)

run(weights='./weights/basket_rim.pt', source='./input/input_images', conf_thres=0.1, project=runs_folder, name=name)


delete_folder_contents(image_folder_path)

base_img = cv2.imread('./runs/run/frame_0000.jpg')
generate_video(base_img, "./runs/run", "./output/new_output")