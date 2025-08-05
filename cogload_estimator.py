#!/usr/bin/env python3
"""
cogload_estimator.py

Cognitive Load Estimation from Eye Gaze & Micro‑Expressions (Webcam‑only POC)

Usage:
  # 1) Calibrate baseline (30s neutral):
  python cogload_estimator.py --calibrate

  # 2) Run estimator:
  python cogload_estimator.py
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import argparse
from collections import deque
from plyer import notification

# === PARAMETERS ===
EAR_THRESH = 0.22           # eye aspect ratio threshold for blink
BLINK_WINDOW = 60          # seconds for blink-rate window
BLINK_MIN_RATE = 10        # blinks/min below which dryness warning
SACCADE_VEL_THRESH = 5.0   # pixels/frame for saccade detection
WINDOW = 10                # seconds per feature window
STRIP_LEN = int(WINDOW * 30)  # number of frames to keep (assuming ~30fps)
HIGH_LOAD_THRESH = 75      # strain index above which to alert
HIGH_LOAD_DURATION = 30    # seconds to sustain before break alert
NO_BLINK_DURATION = 5     # seconds no blink before blink reminder

CALIB_FILE = "calib.json"


# === Utility Functions ===

def eye_aspect_ratio(eye):
    # eye: 6 points (x,y)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def notify(title, msg):
    """Cross-platform desktop notification."""
    notification.notify(
        title=title,
        message=msg,
        timeout=5
    )

# === Calibration ===

def calibrate(baseline_duration=30):
    """
    Run a 30s neutral recording to compute baseline metrics.
    """
    print(f"[CALIBRATE] Hold a neutral expression for {baseline_duration}s...")
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True)
    start = time.time()
    ears, pupil_sizes, brow_diffs = [], [], []
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = None
    prev_center = None

    while time.time() - start < baseline_duration:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if not results.multi_face_landmarks:
            cv2.putText(frame, "Face not found", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.imshow("Calibrating", frame)
            cv2.waitKey(1)
            continue

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = np.array([(int(p.x*w), int(p.y*h)) for p in lm])
        # EAR: left eye indices from MediaPipe
        left_eye = pts[[33,160,158,133,153,144]]
        right_eye = pts[[263,387,385,362,380,373]]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        ears.append(ear)

        # Pupil: use centroid of iris landmarks 468-477
        iris = pts[468:478]
        center = iris.mean(axis=0)
        pupil_sizes.append(np.std(iris[:,0] - center[0]))  # approximate radius

        # Brow- eye vertical diff: average eyebrow top (y of 10/338) minus eye top (min y of eye pts)
        brow = (pts[10][1] + pts[338][1]) / 2.0
        eye_top = min(left_eye[:,1].min(), right_eye[:,1].min())
        brow_diffs.append(abs(brow - eye_top))

        # for optical flow initialization
        prev_gray = gray.copy()
        prev_center = np.array(center, dtype=np.float32)

        cv2.imshow("Calibrating", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    calib = {
        "ear": float(np.mean(ears)),
        "pupil": float(np.mean(pupil_sizes)),
        "brow_diff": float(np.mean(brow_diffs))
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[CALIBRATE] Saved baseline to {CALIB_FILE}: {calib}")


# === Main Estimator ===

def run():
    # load calibration
    try:
        with open(CALIB_FILE) as f:
            calib = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file '{CALIB_FILE}' not found. Run with --calibrate first.")
        return

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True)
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # rolling buffers
    gray_prev = None
    pupil_prev = None
    blink_times = deque()
    flow_centers = deque(maxlen=STRIP_LEN)
    brow_deque = deque(maxlen=STRIP_LEN)

    window_start = time.time()
    last_break_alert = 0
    last_blink_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # preprocess
        frame = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face & landmarks
        results = mp_face.process(rgb)
        if not results.multi_face_landmarks:
            cv2.imshow("Estimator", frame)
            if cv2.waitKey(1)==27: break
            continue
        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = np.array([(int(p.x*w), int(p.y*h)) for p in lm])

        # EAR & blink detection
        left_eye = pts[[33,160,158,133,153,144]]
        right_eye = pts[[263,387,385,362,380,373]]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        now = time.time()
        if ear < EAR_THRESH:
            blink_times.append(now)
        # purge old blinks
        while blink_times and now - blink_times[0] > BLINK_WINDOW:
            blink_times.popleft()

        # pupil center
        iris = pts[468:478]
        center = iris.mean(axis=0)
        # collect for flow
        if pupil_prev is not None and gray_prev is not None:
            p0 = np.array([pupil_prev], dtype=np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)
            if st[0][0] == 1:
                flow_centers.append(tuple(p1[0]))

        pupil_prev = center
        gray_prev = gray.copy()

        # brow diff
        brow = (pts[10][1] + pts[338][1]) / 2.0
        eye_top = min(left_eye[:,1].min(), right_eye[:,1].min())
        brow_diff = abs(brow - eye_top)
        brow_deque.append(brow_diff)

        # draw
        cv2.putText(frame, f"EAR: {ear:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.circle(frame, tuple(center.astype(int)), 3, (0,0,255), -1)

        # feature window
        if now - window_start >= WINDOW:
            # compute features
            blink_rate = len(blink_times) * (60.0/BLINK_WINDOW)
            pupil_sizes = [abs(c[0] - calib["pupil"]) for c in flow_centers]  # misuse flow_centers length as proxy
            # actually use iris size change: (we skip for brevity)
            # saccade rate:
            velocities = []
            pts_list = list(flow_centers)
            for i in range(1, len(pts_list)):
                velocities.append(np.linalg.norm(np.array(pts_list[i]) - np.array(pts_list[i-1])))
            saccades = sum(v > SACCADE_VEL_THRESH for v in velocities)
            saccade_rate = saccades * (60.0 / WINDOW)
            avg_fix = WINDOW*1.0 / max(1, (saccades+1))  # rough
            au4_var = np.var(brow_deque)

            # simple strain index: normalize each to 0–1 and weight equally
            f1 = np.clip((blink_rate - calib["ear"]*60) / (BLINK_MIN_RATE), 0, 1)
            f2 = np.clip((au4_var - calib["brow_diff"]) / (2*calib["brow_diff"]), 0, 1)
            f3 = np.clip(saccade_rate / 30.0, 0, 1)
            strain = np.mean([f1, f2, f3]) * 100

            # alerts
            # blink reminder
            if len(blink_times)==0 and now - last_blink_alert > NO_BLINK_DURATION:
                notify("CognitiveLoad", "Please blink to refresh your eyes")
                last_blink_alert = now
            # break reminder
            if strain > HIGH_LOAD_THRESH and now - last_break_alert > HIGH_LOAD_DURATION:
                notify("CognitiveLoad", "High cognitive load detected—consider a break")
                last_break_alert = now

            # display strain
            cv2.putText(frame, f"Strain: {strain:.0f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            window_start = now
            flow_centers.clear()
            brow_deque.clear()

        cv2.imshow("Estimator", frame)
        if cv2.waitKey(1)==27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# === Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Run baseline calibration")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    else:
        run()
