#!/usr/bin/env python3
"""
cogload_estimator.py

Cognitive Load Estimation from Eye Gaze & Micro‑Expressions (Webcam‑only POC)

Usage:
  # 1) Calibrate baseline (30s neutral):
  python cogload_estimator.py --calibrate

  # 2) Run estimator:
  python cogload_estimator.py [--fixation] [--microsacc] [--au] [--headpose]

  # 3) Smooth-pursuit calibration mode:
  python cogload_estimator.py --pursuit
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import argparse
import math
import subprocess
import csv
import pygame
from collections import deque
from plyer import notification

# === PARAMETERS ===
EAR_THRESH = 0.22           # eye aspect ratio threshold for blink
BLINK_WINDOW = 60           # seconds for blink-rate window
BLINK_MIN_RATE = 10         # blinks/min below which dryness warning
SACCADE_VEL_THRESH = 5.0    # pixels/frame for saccade detection
WINDOW = 10                 # seconds per feature window
STRIP_LEN = int(WINDOW * 30)  # number of frames to keep (~30fps)
HIGH_LOAD_THRESH = 75       # strain index above which to alert
HIGH_LOAD_DURATION = 30     # seconds to sustain before break alert
NO_BLINK_DURATION = 5       # seconds no blink before blink reminder

# Fixation
FIXATION_MIN_DUR = 0.2      # seconds
FIX_THRESH_PIX   = 15       # px

# Microsaccades
MICRO_MIN = 0.5             # px/frame
MICRO_MAX = 2.0             # px/frame

CALIB_FILE = "calib.json"

# === Utility Functions ===

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def notify(title, msg):
    notification.notify(title=title, message=msg, timeout=5)

# === Calibration ===

def calibrate(baseline_duration=30):
    print(f"[CALIBRATE] Hold a neutral expression for {baseline_duration}s...")
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True)
    start = time.time()
    ears, pupil_sizes, brow_diffs = [], [], []
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = None
    prev_center = None

    while time.time() - start < baseline_duration:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if not results.multi_face_landmarks:
            cv2.imshow("Calibrating", frame)
            cv2.waitKey(1); continue

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = np.array([(int(p.x*w), int(p.y*h)) for p in lm])

        # EAR
        left_eye = pts[[33,160,158,133,153,144]]
        right_eye = pts[[263,387,385,362,380,373]]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        ears.append(ear)

        # Pupil size proxy
        iris = pts[468:478]
        center = iris.mean(axis=0)
        pupil_sizes.append(np.std(iris[:,0] - center[0]))

        # Brow diff
        brow = (pts[10][1] + pts[338][1]) / 2.0
        eye_top = min(left_eye[:,1].min(), right_eye[:,1].min())
        brow_diffs.append(abs(brow - eye_top))

        prev_gray = gray.copy()
        prev_center = np.array(center, dtype=np.float32)

        cv2.imshow("Calibrating", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

    calib = {"ear": float(np.mean(ears)),
             "pupil": float(np.mean(pupil_sizes)),
             "brow_diff": float(np.mean(brow_diffs))}
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[CALIBRATE] Saved baseline to {CALIB_FILE}: {calib}")

# === Smooth Pursuit Mode ===

def pursuit_task(duration=15):
    pygame.init()
    screen = pygame.display.set_mode((640,480))
    clock = pygame.time.Clock()
    t0 = time.time()
    traj = []
    while time.time() - t0 < duration:
        t = time.time() - t0
        x = 320 + 200 * math.sin(2*math.pi*0.2*t)
        y = 240
        screen.fill((0,0,0))
        pygame.draw.circle(screen, (0,255,0), (int(x),int(y)), 10)
        pygame.display.flip()
        traj.append((time.time(), x, y))
        clock.tick(30)
    pygame.quit()
    # Save trajectory to CSV
    with open("pursuit_traj.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","x","y"])
        writer.writerows(traj)
    print("[PURSUIT] Trajectory saved to pursuit_traj.csv")

# === Main Estimator ===

def run(args):
    # load calibration
    try:
        with open(CALIB_FILE) as f:
            calib = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file '{CALIB_FILE}' not found.")
        return

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1, refine_landmarks=True)
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    gray_prev = None
    pupil_prev = None
    blink_times = deque()
    flow_centers = deque(maxlen=STRIP_LEN)
    brow_deque = deque(maxlen=STRIP_LEN)
    fixation_buffer = deque()

    window_start = time.time()
    last_break_alert = last_blink_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        while blink_times and now - blink_times[0] > BLINK_WINDOW:
            blink_times.popleft()

        # pupil center & optical flow
        iris = pts[468:478]
        center = iris.mean(axis=0)
        if pupil_prev is not None and gray_prev is not None:
            p0 = np.array([pupil_prev], dtype=np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)
            if st[0][0] == 1:
                flow_centers.append(tuple(p1[0]))
        pupil_prev, gray_prev = center, gray.copy()

        # brow diff
        brow = (pts[10][1] + pts[338][1]) / 2.0
        eye_top = min(left_eye[:,1].min(), right_eye[:,1].min())
        brow_diff = abs(brow - eye_top)
        brow_deque.append(brow_diff)

        # Feature additions
        # 1) Fixation detection
        if args.fixation:
            fixation_buffer.append((now, *center))
            while fixation_buffer and now - fixation_buffer[0][0] > FIXATION_MIN_DUR:
                fixation_buffer.popleft()
            if fixation_buffer and now - fixation_buffer[0][0] >= FIXATION_MIN_DUR:
                xs = [p[1] for p in fixation_buffer]
                ys = [p[2] for p in fixation_buffer]
                disp = (max(xs)-min(xs)) + (max(ys)-min(ys))
                if disp < FIX_THRESH_PIX:
                    cv2.putText(frame, "Fixation", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

        # draw EAR & pupil
        cv2.putText(frame, f"EAR: {ear:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.circle(frame, tuple(center.astype(int)), 3, (0,0,255), -1)

        # head pose
        if args.headpose:
            model_pts = np.array([
                (0.0,0.0,0.0), (0.0,-330.0,-65.0), (-225.0,170.0,-135.0),
                (225.0,170.0,-135.0), (-150.0,-150.0,-125.0), (150.0,-150.0,-125.0)
            ])
            img_pts = np.array([
                pts[1], pts[152], pts[263], pts[33], pts[287], pts[57]
            ], dtype="double")
            focal = w
            cam_mat = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]])
            dist = np.zeros((4,1))
            _, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_mat, dist)
            axis = np.float32([[50,0,0],[0,50,0],[0,0,50]])
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mat, dist)
            p1 = tuple(img_pts[0].ravel().astype(int))
            for p, col in zip(imgpts, [(0,0,255),(0,255,0),(255,0,0)]):
                cv2.line(frame, p1, tuple(p.ravel().astype(int)), col,2)

        # compute window-based features
        if now - window_start >= WINDOW:
            blink_rate = len(blink_times)*(60.0/BLINK_WINDOW)
            # saccade rate
            velocities = [
                np.linalg.norm(np.array(flow_centers[i]) - flow_centers[i-1])
                for i in range(1,len(flow_centers))
            ]
            saccades = sum(v > SACCADE_VEL_THRESH for v in velocities)
            saccade_rate = saccades*(60.0/WINDOW)
            au4_var = np.var(brow_deque)

            # microsaccades
            if args.microsacc:
                micro_count = sum(MICRO_MIN < v <= MICRO_MAX for v in velocities)
                if micro_count>0:
                    cv2.putText(frame, f"Microsaccades: {micro_count}", (10,120),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            # action-unit detection
            if args.au:
                cv2.imwrite("_tmp.jpg", frame)
                sub = subprocess.run(["FeatureExtraction","-f","_tmp.jpg","-aus"], capture_output=True)
                with open("_tmp_000.csv") as fcsv:
                    reader = csv.DictReader(fcsv)
                    row = next(reader)
                    au1 = float(row.get("AU01_r",0))
                cv2.putText(frame, f"AU01: {au1:.2f}", (10,150),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

            # strain index
            f1 = np.clip((blink_rate - calib["ear"]*60)/BLINK_MIN_RATE,0,1)
            f2 = np.clip((au4_var - calib["brow_diff"])/(2*calib["brow_diff"]),0,1)
            f3 = np.clip(saccade_rate/30.0,0,1)
            strain = np.mean([f1,f2,f3])*100

            # alerts
            if len(blink_times)==0 and now-last_blink_alert>NO_BLINK_DURATION:
                notify("CognitiveLoad","Please blink to refresh your eyes")
                last_blink_alert = now
            if strain>HIGH_LOAD_THRESH and now-last_break_alert>HIGH_LOAD_DURATION:
                notify("CognitiveLoad","High cognitive load detected—consider a break")
                last_break_alert = now

            cv2.putText(frame, f"Strain: {strain:.0f}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            window_start = now
            flow_centers.clear(); brow_deque.clear()

        cv2.imshow("Estimator", frame)
        if cv2.waitKey(1)==27: break

    cap.release()
    cv2.destroyAllWindows()

# === Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--fixation", action="store_true")
    parser.add_argument("--microsacc", action="store_true")
    parser.add_argument("--au", action="store_true")
    parser.add_argument("--headpose", action="store_true")
    parser.add_argument("--pursuit", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    elif args.pursuit:
        pursuit_task()
    else:
        run(args)
