import cv2
import numpy as np
import face_recognition
import os
import socket
import time
from datetime import datetime

# ---- Config ----
TRAINING_PATH = 'Training_images'
ATTENDANCE_FILE = 'Attendance.csv'
FACE_MATCH_TOLERANCE = 0.6   # lower = stricter match
FRAME_RESIZE_SCALE = 0.25    # smaller = faster but less accurate detection
UNKNOWN_LOG_COOLDOWN_SEC = 30  # avoid flooding the CSV with repeated "Unknown" rows

_last_unknown_log_time = 0  # tracks last time an Unknown row was written


def get_local_ip():
    """Best-effort local network IP (not a public/internet IP)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def load_training_images(path):
    images = []
    class_names = []
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Training folder '{path}' not found.")

    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {filename} (skipping)")
            continue
        images.append(img)
        class_names.append(os.path.splitext(filename)[0])
    return images, class_names


def find_encodings(images, class_names):
    """Returns (encodings, matching_names) - skips images with no detectable face."""
    encode_list = []
    valid_names = []

    for img, name in zip(images, class_names):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if not encodings:
            print(f"[WARN] No face detected in training image for '{name}' (skipping)")
            continue
        encode_list.append(encodings[0])
        valid_names.append(name)

    return encode_list, valid_names


def ensure_attendance_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('Name, Time, IP\n')


def mark_attendance(name, ip, path):
    """Logs a known person once (skips if already logged)."""
    with open(path, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0].strip() for line in my_data_list]

        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'{name},{dt_string},{ip}\n')


def log_unknown(ip, path):
    """Logs an unrecognized face, throttled by UNKNOWN_LOG_COOLDOWN_SEC."""
    global _last_unknown_log_time
    now_ts = time.time()
    if now_ts - _last_unknown_log_time < UNKNOWN_LOG_COOLDOWN_SEC:
        return
    _last_unknown_log_time = now_ts

    with open(path, 'a') as f:
        now = datetime.now()
        dt_string = now.strftime('%H:%M:%S')
        f.write(f'Unknown,{dt_string},{ip}\n')


def main():
    images, class_names = load_training_images(TRAINING_PATH)
    print(f"Loaded {len(images)} training images: {class_names}")

    encode_list_known, class_names = find_encodings(images, class_names)
    print(f"Encoding complete. {len(encode_list_known)} usable face(s).")

    if not encode_list_known:
        print("[ERROR] No usable face encodings found. Check your training images.")
        return

    ensure_attendance_file(ATTENDANCE_FILE)
    local_ip = get_local_ip()
    print(f"Local IP for attendance logging: {local_ip}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("[WARN] Failed to read frame from webcam.")
                continue

            img_small = cv2.resize(img, (0, 0), None, FRAME_RESIZE_SCALE, FRAME_RESIZE_SCALE)
            img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            faces_cur_frame = face_recognition.face_locations(img_small)
            encodes_cur_frame = face_recognition.face_encodings(img_small, faces_cur_frame)

            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(
                    encode_list_known, encode_face, tolerance=FACE_MATCH_TOLERANCE
                )
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_dis)

                scale = int(1 / FRAME_RESIZE_SCALE)
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * scale, x2 * scale, y2 * scale, x1 * scale

                if matches[match_index]:
                    name = class_names[match_index].upper()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    mark_attendance(name, local_ip, ATTENDANCE_FILE)
                else:
                    print("[WARNING] Face detected but not recognized (not in Training_images).")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, 'UNKNOWN', (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    log_unknown(local_ip, ATTENDANCE_FILE)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()