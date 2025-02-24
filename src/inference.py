import torch
import cv2
from PIL import Image , ImageDraw
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm

class GazeTracker:

  def __init__(self, cuda = True):
    self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {self.device}")

    # Load Gazelle model
    self.model, self.transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    self.model.eval()
    self.model.to(self.device)

    # Colors for visualization
    self.colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']

  def process_frame(self, frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    width , height = image.size

    # detect face
    face_detection = RetinaFace.detect_faces(frame_rgb)

    # extract bounding box
    bboxes = [face_detection[key]['facial_area'] for key in face_detection.keys()]

    if not bboxes:
      return frame

    # norm the bboxes
    norm_bboxes = [[np.array(bbox)/ np.array([width, height, width, height]) for bbox in bboxes]]

    # prepare the input
    img_tensor =  self.transform(image).unsqueeze(0).to(self.device)
    input_data = {
        "images":img_tensor,
        "bboxes": norm_bboxes
    }

    # get model prediction
    with torch.no_grad():
      pred = self.model(input_data)

    #visualize result
    result_image = self.visualize_all(
        image,
        pred['heatmap'][0],
        norm_bboxes[0],
        pred['inout'][0] if pred['inout'] is not None else None
    )

    #convert back to rgb using opencv
    result = np.array(result_image)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

  def visualize_all(self,image,heatmaps,bboxes,inout_scores,inout_thresh = 0.5):

    overlay_image = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width , height = image.size

    for i in range(len(bboxes)):
      bbox = bboxes[i]
      xmin, ymin, xmax, ymax = bbox
      color = self.colors[i%len(self.colors)]

      # draw face bbox

      draw.rectangle(
          [xmin * width, ymin * height, xmax * width, ymax * height],
                outline=color,
                width=int(min(width, height) * 0.01)
      )

      if inout_scores is not None:
        inout_score = inout_scores[i]

        #draw in frame score
        text = f"in-frame: {inout_score:.2f}"
        test_y = ymax * height + int(height * 0.01)

        draw.text(
            (xmin*width, test_y),
            text,
            fill=color,
            stroke_width=1,
            font = None,
            font_size = 2
        )
        #  draw gaze direction if looking in-frame
        if inout_score > inout_thresh:
          heatmap = heatmaps[i]
          heatmap_np = heatmap.detach().cpu().numpy()
          max_index = np.unravel_index(np.argmax(heatmap_np),heatmap_np.shape)

          # calculate gaze target
          gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
          gaze_target_y = max_index[0] / heatmap_np.shape[0] * height

          # calculate face center
          bbox_center_x = ((xmin + xmax)/2)*width
          bbox_center_y = ((ymin + ymax)/2)*height

          # draw gaze target and line

          draw.ellipse(
              [(gaze_target_x-5,gaze_target_y-5),(gaze_target_x+5,gaze_target_y+5)],
              fill=color,width = int(0.005*min(width,height))
          )
          draw.line(
              [(bbox_center_x,bbox_center_y),(gaze_target_x,gaze_target_y)],
              fill=color,width = int(0.005*min(width,height))
          )
    return overlay_image.convert("RGB")

  def process_video(self,video_path,output_path,start_time = 0,duration = None):

    # oepn a video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
      raise ValueError("Could not open video")

    #video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate start and end frames

    start_frame = int(start_time * fps)
    if duration:
      end_frame = start_frame +  int(duration * fps)
    else:
      end_frame = total_frames

    #set up video writer

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path,fourcc,fps,(width,height))

    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

    try:
      with tqdm(total = end_frame - start_frame,desc = "Processing") as pbar:
        frame_count = start_frame
        while cap.isOpened() and frame_count < end_frame:
          ret, frame = cap.read()
          if not ret:
            break

          # process frame
          processed_frame = self.process_frame(frame)
          out.write(processed_frame)
          frame_count += 1
          pbar.update(1)
    finally:
      cap.release()
      out.release()
      cv2.destroyAllWindows()