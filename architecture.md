# Cognitive-Load Estimator Architecture

```mermaid
flowchart TD
  %% Calibration pipeline
  subgraph Calibration
    A1[Webcam Input]
    B1[FaceMesh Detection]
    C1[Compute Baseline Metrics\n(EAR · Pupil · Brow)]
    D1[Save calib.json]
    A1 --> B1 --> C1 --> D1
  end

  %% Estimation pipeline
  subgraph Estimation
    A2[Webcam Input]
    B2[FaceMesh Detection]
    E1[Blink Detection\n(EAR)]
    E2[Pupil Tracking\n(Optical Flow)]
    E3[Brow Diff\n(Variance)]
    F1[Sliding Window\n(Buffer Features)]
    G1[Fixation Detection]
    G2[Microsaccade Detection]
    G3[AU Extraction\n(OpenFace)]
    G4[Head-Pose Estimation]
    H1[Strain Index\nComputation]
    I1[Overlay & Alerts]
    A2 --> B2
    B2 --> E1
    B2 --> E2
    B2 --> E3
    E1 & E2 & E3 --> F1
    F1 --> G1
    F1 --> G2
    F1 --> G3
    F1 --> G4
    G1 & G2 & G3 & G4 --> H1
    H1 --> I1
  end

  %% Smooth-pursuit mode
  subgraph Smooth-Pursuit Mode
    P1[Launch Moving Dot]
    P2[Record Gaze Trajectory]
    P3[Save pursuit_traj.csv]
    P1 --> P2 --> P3
  end
