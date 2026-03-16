Perception Task Report

Methodology:
Distance estimation was performed using the Pin-hole Camera Model. I utilized the Similarity of Triangles principle, which relates the known real-world height of the object (H=30cm) and the camera's focal length (f=1000mm) to the object's height in pixels (h) detected by the YOLOv11 model. The depth d was calculated using the formula d=(H×f)/h.

Assumptions: 
1. The traffic cones are standing perfectly upright on a level ground plane.
2. The YOLO bounding boxes are tightly fit to the vertical extent of the cones.