import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import os
from datetime import datetime
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wave_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import pyodbc
        logging.info("All required packages are installed")
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {str(e)}")
        return False

def is_blue_background(frame, x, y, w, h, threshold=0.2):
    """Check if the region has predominantly blue background with improved color detection"""
    # Extract the region around the contour with padding
    pad = 30  # Increased padding to better analyze surrounding area
    y_start = max(0, y-pad)
    y_end = min(frame.shape[0], y+h+pad)
    x_start = max(0, x-pad)
    x_end = min(frame.shape[1], x+w+pad)
    
    region = frame[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        return False
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Define broader blue color range in HSV
    lower_blue = np.array([90, 50, 50])   # Adjusted for more permissive blue detection
    upper_blue = np.array([140, 255, 255])
    
    # Create mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Calculate percentage of blue pixels
    blue_percentage = np.sum(blue_mask > 0) / blue_mask.size
    
    return blue_percentage > threshold

def analyze_frame(frame):
    """Analyze a single frame using enhanced wave detection logic"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding instead of global thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,  # Block size
            5    # C constant
        )
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        
        # Morphological operations with adjusted kernel size
        kernel = np.ones((7, 7), np.uint8)  # Increased kernel size
        binary = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with hierarchy to better handle nested contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store all wave measurements
        wave_measurements = []
        annotated_frame = frame.copy()
        
        min_contour_area = 1000  # Minimum area to be considered a wave
        max_contour_area = frame.shape[0] * frame.shape[1] * 0.5  # Max 50% of frame
        
        for i, contour in enumerate(contours):
            # Filter out very small or very large contours
            area = cv2.contourArea(contour)
            if area < min_contour_area or area > max_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio check - waves are typically taller than they are wide
            aspect_ratio = h / w
            if aspect_ratio < 0.5:  # Skip if too wide relative to height
                continue
            
            # Check if this contour has a blue background
            if not is_blue_background(frame, x, y, w, h):
                continue
            
            # Calculate wave metrics
            height_pixels = h
            
            # Convert to meters (assuming calibration of 100 pixels = 1 meter)
            # This should be calibrated based on known reference objects
            pixel_to_meter_ratio = 1.0 / 100
            height_meters = height_pixels * pixel_to_meter_ratio
            
            # Draw thicker rectangle and measurements on the frame
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Increased thickness
            
            # Add background rectangle for text
            text = f"{height_meters:.2f}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (x, y-text_size[1]-10), 
                         (x+text_size[0]+10, y), 
                         (0, 0, 0), 
                         -1)  # Filled rectangle
            
            # Draw text with outline for better visibility
            cv2.putText(annotated_frame, text,
                       (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
            
            # Store measurement with additional metrics
            wave_measurements.append({
                'height_meters': height_meters,
                'height_pixels': height_pixels,
                'x': x,
                'y': y,
                'width': w,
                'area_pixels': area,
                'aspect_ratio': aspect_ratio
            })
        
        # Add wave count to frame
        count_text = f"Waves: {len(wave_measurements)}"
        cv2.putText(annotated_frame, count_text,
                   (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.0,
                   (0, 255, 0),
                   2)
        
        return wave_measurements, annotated_frame
        
    except Exception as e:
        logging.error(f"Error in analyze_frame: {str(e)}")
        raise

def process_video(video_path, output_path=None, location_id=1, weather_conditions="Clear", frame_interval=30):
    """Process each frame of the video and store results in SQL Server"""
    # SQL Server connection
    conn_str = 'DRIVER={SQL Server};SERVER=PARKER1\\SQLEXPRESS;DATABASE=wave;Trusted_Connection=yes;'
    
    # Check if input video exists
    if not os.path.exists(video_path):
        logging.error(f"Input video file not found: {video_path}")
        return False
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video file {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        
        # Create video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Only process every nth frame
            if frame_number % frame_interval != 0:
                continue
            
            logging.info(f"Processing frame {frame_number}/{total_frames} ({(frame_number/total_frames)*100:.1f}%)")
            
            wave_measurements, annotated_frame = analyze_frame(frame)
            
            # Save the annotated frame to output video if specified
            if output_path:
                out.write(annotated_frame)
            
            # Save frame as image for database reference
            frame_filename = f"frame_{frame_number}.jpg"
            cv2.imwrite(frame_filename, annotated_frame)
            
            # Store measurements in database
            for wave in wave_measurements:
                cursor.execute('''
                    INSERT INTO WaveMeasurements 
                    (CaptureDateTime, WaveHeightMeters, ImagePath, LocationID, WeatherConditions,
                     HeightPixels, XPosition, YPosition, Width, Area, AspectRatio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(), 
                    wave['height_meters'], 
                    frame_filename, 
                    location_id, 
                    weather_conditions,
                    wave['height_pixels'],
                    wave['x'],
                    wave['y'],
                    wave['width'],
                    wave['area_pixels'],
                    wave['aspect_ratio']
                ))
                conn.commit()
            
            logging.info(f"Found {len(wave_measurements)} waves in frame {frame_number}")
        
        cap.release()
        if output_path:
            out.release()
        
        logging.info(f"Processing complete. Processed {frame_number} frames.")
        return True
        
    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def steal_frames(video_path, output_dir="data", max_frames=800):
    """
    Extract frames from a video file and save them to the output directory
    with wave analysis and SQL database storage
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # SQL Server connection
    conn_str = 'DRIVER={SQL Server};SERVER=PARKER1\\SQLEXPRESS;DATABASE=wave;Trusted_Connection=yes;'
    
    # Check if input video exists
    if not os.path.exists(video_path):
        logging.error(f"Input video file not found: {video_path}")
        return False
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video file {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} total frames")
        logging.info(f"Extracting first {max_frames} frames...")
        
        # Process frames
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"End of video reached at frame {frame_count}")
                break
            
            # Save original frame as image
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Process frame to find waves
            try:
                # Use the analyze_frame function to detect waves and draw boxes
                wave_measurements, annotated_frame = analyze_frame(frame)
                
                # Save annotated frame with boxes
                annotated_path = os.path.join(output_dir, f"annotated_{frame_count:04d}.jpg")
                cv2.imwrite(annotated_path, annotated_frame)
                
                # Store measurements in database
                if wave_measurements:
                    for wave in wave_measurements:
                        cursor.execute('''
                            INSERT INTO WaveMeasurements 
                            (CaptureDateTime, WaveHeightMeters, ImagePath, LocationID, WeatherConditions)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (datetime.now(), wave['height_meters'], annotated_path, 1, "Clear"))
                        conn.commit()
                
                logging.info(f"Frame {frame_count}: Found {len(wave_measurements)} waves")
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {str(e)}")
            
            frame_count += 1
        
        cap.release()
        logging.info(f"Extracted {frame_count} frames to {output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Error during frame extraction: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        logging.error("Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process video frames for wave analysis')
    parser.add_argument('--mode', choices=['process', 'steal'], default='process',
                       help='Mode: process (full video) or steal (extract frames)')
    parser.add_argument('--video', default="1550080-uhd_3840_2160_30fps.mp4",
                       help='Input video file path')
    parser.add_argument('--output', default="annotated_wave_video.mp4",
                       help='Output video file path (for process mode)')
    parser.add_argument('--frames', type=int, default=800,
                       help='Number of frames to extract (for steal mode)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Frame interval for processing')
    parser.add_argument('--outdir', default="data",
                       help='Output directory for extracted frames')
    
    args = parser.parse_args()
    
    if args.mode == 'process':
        logging.info(f"Starting wave analysis on video: {args.video}")
        if process_video(args.video, args.output, frame_interval=args.interval):
            logging.info("Video processing completed successfully")
        else:
            logging.error("Video processing failed")
            sys.exit(1)
    else:  # steal mode
        logging.info(f"Starting frame extraction from video: {args.video}")
        if steal_frames(args.video, args.outdir, max_frames=args.frames):
            logging.info("Frame extraction completed successfully")
        else:
            logging.error("Frame extraction failed")
            sys.exit(1) 