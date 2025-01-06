import cv2
import sys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


FRAME_WIDTH = 2304
FRAME_HEIGHT = 1296
CLASS_NAMES = ["auto", "bus", "car", "cycle", "motorcycle", "truck", "van"]
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
    6: (128, 128, 128),
}


temp_line_points = []
done_this_line = False

def draw_line_mouse_callback(event, x, y, flags, param):

    global temp_line_points, done_this_line

    if event == cv2.EVENT_LBUTTONDOWN:
        temp_line_points.append((x, y))
        if len(temp_line_points) == 2:
            done_this_line = True

def select_line(window_name, cap, label="Line"):

    global temp_line_points, done_this_line
    temp_line_points = []
    done_this_line = False
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(window_name, draw_line_mouse_callback)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #rewind in case video playback ends

    print(f"\nDrawing '{label}' line: Insert starting and ending point of the line.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if len(temp_line_points) == 1:
            cv2.circle(frame, temp_line_points[0], 5, (0, 255, 0), -1)

        if len(temp_line_points) == 2:
            cv2.line(frame, temp_line_points[0], temp_line_points[1], (0, 255, 0), 2)
            cv2.circle(frame, temp_line_points[0], 5, (0, 255, 0), -1)
            cv2.circle(frame, temp_line_points[1], 5, (0, 255, 0), -1)
            cv2.putText(frame, label, (temp_line_points[0][0], temp_line_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20)
        if key == 27:
            print(f"Cancelled drawing '{label}' line.")
            temp_line_points = []
            break

        if done_this_line:
            break

    cv2.destroyWindow(window_name)

    if len(temp_line_points) < 2:
        print(f"Not enough points for '{label}'")
        return None

    return (temp_line_points[0], temp_line_points[1])

def ccw(A, B, C):

    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def lines_intersect(p1, p2, p3, p4):

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def main():
    video_path = input("Enter video path (or press Enter for 'traffic.mp4'): ")
    if not video_path:
        video_path = "traffic.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        sys.exit(1)


    eb_start_line = select_line("Draw EB START", cap, "EB START")
    eb_end_line   = select_line("Draw EB END",   cap, "EB END")
    wb_start_line = select_line("Draw WB START", cap, "WB START")
    wb_end_line   = select_line("Draw WB END",   cap, "WB END")


    if not all([eb_start_line, eb_end_line, wb_start_line, wb_end_line]):
        print("Not all lines were drawn successfully. Exiting.")
        sys.exit(1)

    try:
        distance_eb_m = float(input("\nEnter distance (m) between EB lines (EB Start -> EB End): "))
        distance_wb_m = float(input("Enter distance (m) between WB lines (WB Start -> WB End): "))
    except ValueError:
        print("[ERROR] Invalid distance input. Exiting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    try:
        model = YOLO("runs/detect/train3/weights/best.pt")
    except FileNotFoundError:
        print("YOLO model file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the YOLO model: {e}")
        sys.exit(1)

    deepsort = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
    )

    old_positions = {}


    eb_start_times = {}
    wb_start_times = {}
    eb_end_times = {}
    wb_end_times = {}
    vehicle_data = []
    eb_count = 0
    wb_count = 0
    eb_speeds = []
    wb_speeds = []


    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", FRAME_WIDTH, FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.04  # Default to ~25 FPS if fps is 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_msec / 1000.0

        if eb_start_line:
            cv2.line(frame, eb_start_line[0], eb_start_line[1], (0, 255, 0), 2)
        if eb_end_line:
            cv2.line(frame, eb_end_line[0], eb_end_line[1], (0, 255, 0), 2)
        if wb_start_line:
            cv2.line(frame, wb_start_line[0], wb_start_line[1], (0, 255, 0), 2)
        if wb_end_line:
            cv2.line(frame, wb_end_line[0], wb_end_line[1], (0, 255, 0), 2)

        results = model(frame, verbose=False, conf=0.5)
        detections = results[0].boxes

        dets_to_deep_sort = []
        for det in detections:
            try:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                conf = det.conf[0].item()
                cls = int(det.cls[0].item())
            except IndexError:
                print("Detected object does not have enough attributes. Skipping.")
                continue

            if cls >= len(CLASS_NAMES):
                print(f"Detected class index {cls} exceeds defined CLASS_NAMES. Skipping.")
                continue

            dets_to_deep_sort.append(([x1, y1, x2 - x1, y2 - y1], conf, CLASS_NAMES[cls]))


        tracks = deepsort.update_tracks(dets_to_deep_sort, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_name = track.get_det_class()
            if cls_name not in CLASS_NAMES:
                continue
            cls_idx = CLASS_NAMES.index(cls_name)


            class_label = cls_name
            color = CLASS_COLORS.get(cls_idx, (0, 255, 255))

            # Get bounding box
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"{class_label}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"ID: {track_id}", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Retrieve previous position
            prev_info = old_positions.get(track_id, None)
            if prev_info is not None:
                old_x, old_y, old_t = prev_info
                old_pt = (old_x, old_y)
                new_pt = bbox_center

                # Check for Eastbound line crossings
                if eb_start_line and lines_intersect(old_pt, new_pt, eb_start_line[0], eb_start_line[1]):
                    if track_id not in eb_start_times:
                        eb_start_times[track_id] = current_sec

                        eb_count += 1

                if eb_end_line and lines_intersect(old_pt, new_pt, eb_end_line[0], eb_end_line[1]):
                    if track_id in eb_start_times and track_id not in eb_end_times:
                        eb_end_times[track_id] = current_sec
                        dt = eb_end_times[track_id] - eb_start_times[track_id]
                        if dt > 0.5:
                            speed_m_s = distance_eb_m / dt
                            speed_km_h = speed_m_s * 3.6
                            eb_speeds.append(speed_km_h)
                            direction = "EB"

                            # Append to vehicle_data list
                            vehicle_data.append({
                                "Vehicle SN": track_id,
                                "Vehicle Class": class_label,
                                "Vehicle Speed (km/h)": round(speed_km_h, 2),
                                "Direction": direction
                            })

                        else:
                            print(f"EB crossing too fast, dt={dt:.3f}s")
                        del eb_start_times[track_id]


                if wb_start_line and lines_intersect(old_pt, new_pt, wb_start_line[0], wb_start_line[1]):
                    if track_id not in wb_start_times:
                        wb_start_times[track_id] = current_sec

                        wb_count += 1

                if wb_end_line and lines_intersect(old_pt, new_pt, wb_end_line[0], wb_end_line[1]):
                    if track_id in wb_start_times and track_id not in wb_end_times:
                        wb_end_times[track_id] = current_sec
                        dt = wb_end_times[track_id] - wb_start_times[track_id]
                        if dt > 0.5:
                            speed_m_s = distance_wb_m / dt
                            speed_km_h = speed_m_s * 3.6
                            wb_speeds.append(speed_km_h)

                            direction = "WB"

                            vehicle_data.append({
                                "Vehicle SN": track_id,
                                "Vehicle Class": class_label,
                                "Vehicle Speed (km/h)": round(speed_km_h, 2),
                                "Direction": direction
                            })

                        else:
                            print(f"WB crossing too fast, dt={dt:.3f}s")
                        del wb_start_times[track_id]

            old_positions[track_id] = (bbox_center[0], bbox_center[1], current_sec)




        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # 7) Final stats
    print("\n=== Final Results ===")
    print(f"Eastbound Count: {eb_count}")
    if eb_speeds:
        avg_eb = sum(eb_speeds) / len(eb_speeds)
        print(f"Avg Eastbound Speed: {avg_eb:.2f} km/h")
    else:
        print("No EB vehicles counted.")

    print(f"Westbound Count: {wb_count}")
    if wb_speeds:
        avg_wb = sum(wb_speeds) / len(wb_speeds)
        print(f"Avg Westbound Speed: {avg_wb:.2f} km/h")
    else:
        print("No WB vehicles counted.")


if __name__ == "__main__":
    main()
