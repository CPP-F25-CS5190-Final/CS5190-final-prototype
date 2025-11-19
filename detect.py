from yolov8 import Yolov8
import cv2
import numpy as np
import argparse

# Define a constant for the default resolution
DEFAULT_RESOLUTION = [1280, 720]

def parse_arguments() -> argparse.Namespace:
    # Use 'description' only for the overall program description
    parser = argparse.ArgumentParser(description='YoloV8 Fire Detection')
    
    parser.add_argument(
        "--webcam-resolution",
        type=int,
        nargs=2,
        # **FIX 1: Set the default resolution correctly**
        default=DEFAULT_RESOLUTION,
        help=f"Webcam resolution (width height). Default: {DEFAULT_RESOLUTION[0]} {DEFAULT_RESOLUTION[1]}"
    )
    parser.add_argument(
        '--model',  
        default=r".\model\fire_detection.onnx", 
        type=str,
        help="Path to the YOLOv8 ONNX model file."
    )
    parser.add_argument(
        "--source", 
        default="0", 
        type=str,
        help="Video source (e.g., '0' for webcam, or a path to a video file)."
    )
    args = parser.parse_args()
    return args

def main():

    # Get information
    args = parse_arguments()
    path = args.source
    model = args.model
    # **FIX 2: Get the desired resolution from arguments**
    webcam_width, webcam_height = args.webcam_resolution

    # Set up the Yolo model
    fire_model = Yolov8()
    fire_model.set_up_model(model)
    fire_model.model.warmup(imgsz=(1 , 3, *fire_model.imgsz))
    
    # Set up video capture
    cap = cv2.VideoCapture(path)
    
    # **FIX 3: Apply the desired resolution if using a webcam ('0')**
    # This might not work for all webcams/drivers, but it's the standard way.
    if path == "0":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

    # **Improvement: Get the actual frame size for display aspect ratio**
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # Reopen the video source if it ends (useful for video files)
            # This is correct for looping video/restarting webcam connection
            cap = cv2.VideoCapture(args.source)
            # **Minor Fix: Re-apply resolution properties if the camera was re-opened**
            if args.source == "0":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
            continue
            
        results = fire_model.inference(frame)
        
        for result in results:
            x1, y1, x2, y2, conf, cls = result
            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put text (class name)
            cv2.putText(frame, str(fire_model.names[int(cls)]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # **Improvement: Resize for display using a size that maintains the aspect ratio**
        # E.g., for a 16:9 input, a 640x360 display might be better than 640x640.
        # However, if 640x640 is desired, keep the original line.
        # Keeping your original line to maintain the intended display size:
        frame_display = cv2.resize(frame, (640, 640)) 

        cv2.imshow("frame", frame_display)

        # press Esc to close the window
        if (cv2.waitKey(1) == 27): # Changed waitKey to 1 for smoother video processing
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()