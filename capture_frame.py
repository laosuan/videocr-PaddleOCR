import cv2

def capture_frame(video_path, time_str, output_path, crop_x=None, crop_y=None, crop_width=None, crop_height=None):
    """
    Capture a frame from a video at a specific time and optionally crop it.
    
    Args:
        video_path (str): Path to the video file
        time_str (str): Time in format 'HH:MM:SS'
        output_path (str): Path to save the captured frame
        crop_x, crop_y, crop_width, crop_height: Crop parameters
    """
    # Parse the time string
    h, m, s = map(int, time_str.split(':'))
    seconds = h * 3600 + m * 60 + s
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get fps to calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(seconds * fps)
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame at position {frame_number} ({time_str})")
        cap.release()
        return False
    
    # Crop the frame if specified
    if all(param is not None for param in [crop_x, crop_y, crop_width, crop_height]):
        crop_y_end = crop_y + crop_height
        crop_x_end = crop_x + crop_width
        
        # Make sure we're within image bounds
        h, w = frame.shape[:2]
        crop_y = max(0, min(crop_y, h-1))
        crop_y_end = max(0, min(crop_y_end, h))
        crop_x = max(0, min(crop_x, w-1))
        crop_x_end = max(0, min(crop_x_end, w))
        
        # Crop
        frame = frame[crop_y:crop_y_end, crop_x:crop_x_end]
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    print(f"Frame saved to {output_path}")
    
    # Release the video
    cap.release()
    return True

if __name__ == '__main__':
    video_path = 'Strangest_Animal_Fact__Why_Do_Animals_Eat_Their_Babies__Filial_Cannibalism__Dr._Binocs_Show.mp4'
    time_str = '00:00:20'
    output_path = 'frame_at_20min.jpg'
    crop_x = 0
    crop_y = 900
    crop_width = 2000
    crop_height = 300
    
    capture_frame(video_path, time_str, output_path, crop_x, crop_y, crop_width, crop_height) 