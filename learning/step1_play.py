import cv2  # OpenCV: reads/writes video and draws on images

VIDEO_IN = "data/clips/practice.mp4"
VIDEO_OUT = "outputs/step1.mp4"

# 1. Open the video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}. Are you running this from the minorProject folder?")

# 2. Read its properties (OpenCV returns them as floats)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{width}x{height}, {fps:.1f} fps, {total} frames ({total / fps:.1f} s)")

# 3. Prepare the output video: same fps and size as the input
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# 4. The frame loop: read -> work on the frame -> write/show -> repeat
frame_no = 0
while True:
    ok, frame = cap.read()      # frame is a NumPy array: (height, width, 3)
    if not ok:                  # ok is False when the video has ended
        break
    frame_no += 1

    if frame_no == 1:           # look inside the very first frame once
        print("frame.shape:", frame.shape)
        print("centre pixel (B, G, R):", frame[height // 2, width // 2])
        crop = frame[height // 4 : height // 2, width // 4 : width // 2]   # [rows, cols] = [y, x]
        cv2.imwrite("outputs/crop.jpg", crop)

    # draw on the frame (colours are B, G, R)
    cv2.putText(frame, f"frame {frame_no}/{total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.rectangle(frame, (width // 3, height // 3), (2 * width // 3, 2 * height // 3), (0, 0, 255), 4)

    writer.write(frame)

    preview = cv2.resize(frame, (960, int(960 * height / width)))   # phone videos are too big for the screen
    cv2.imshow("step 1 - press q to quit", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 5. Clean up
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"saved {VIDEO_OUT} and outputs/crop.jpg")