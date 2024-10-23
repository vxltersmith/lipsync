import os
import cv2



def read_video_as_numpy(folder_path, video_name):
    # Construct the full video path
    video_path = os.path.join(folder_path, video_name)
    # Open the video using cv2
    cap = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video: {video_path}")
    frames = []  # List to hold the frames as numpy arrays
    # Read frames one by one
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or failed to read a frame
            break
        # Append frame to the list (as numpy array)
        frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames
