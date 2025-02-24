import argparse
from src.inference import GazeTracker

def main():
    parser = argparse.ArgumentParser(description="Run GazeTracker on a video file.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--start_time", type=float, default=0.0, help="Optional start time in seconds (default: 0.0).")
    
    args = parser.parse_args()
    
    gaze_tracker = GazeTracker()
    gaze_tracker.process_video(args.video_path, args.output_path, start_time=args.start_time)

if __name__ == "__main__":
    main()
