Specifically created to detect vehicles in Terai regions of Nepal
The model used here was custom trained in roboflow using 1000 images from a midblock section in Bharatpur, Nepal.
Utilizes yolov8 for detection and deepsort for tracking. 

Features:
Detects vehicle
Classifies vehicles
Counts the total number of vehicles according to their directions
Calculates average speed and individual vehicle speed

Installation:
Clone the repo:
git clone https://github.com/kaydotehdot/vehicle-tracking-yolo-deepsort.git
cd vehicle-tracking-yolo-deepsort

Change FRAME_WIDTH, FRAME_HEIGHT
You can change the model in line 122.
