import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading

from streamlit_webrtc import VideoProcessorBase

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector

from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):

        # =====================================================
        # STATE
        # =====================================================

        self._lock = threading.Lock()

        self._latest_metrics = {
            "reps": 0,
            "pose_detected": False,
        }

        self._exercise_type = "Squats"

        # =====================================================
        # PROJECT ROOT
        # =====================================================

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_root = os.path.abspath(
            os.path.join(
                current_dir,
                "..",
                "..",
                ".."
            )
        )

        # =====================================================
        # MEDIAPIPE MODEL
        # =====================================================

        model_path = os.path.join(
            project_root,
            "ml_models",
            "pose_landmarker_full.task"
        )

        print(
            "========================================"
        )
        print(
            "MediaPipe Model Path:"
        )
        print(
            model_path
        )
        print(
            "========================================"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"""
❌ MediaPipe model file not found.

Expected location:

{model_path}

Your project should contain:

ai-real-time-gym-trainer/
├── Main App/
├── ml_models/
│   └── pose_landmarker_full.task
├── static/
└── requirements.txt
"""
            )

        # =====================================================
        # MEDIAPIPE OPTIONS
        # =====================================================

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.VIDEO,

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_segmentation_masks=False,
        )

        # =====================================================
        # CREATE LANDMARKER
        # =====================================================

        try:

            self._landmarker = (
                vision.PoseLandmarker.create_from_options(
                    options
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"""
❌ MediaPipe PoseLandmarker failed.

Model:

{model_path}

Error:

{type(e).__name__}: {e}
"""
            ) from e

        # =====================================================
        # DETECTORS
        # =====================================================

        self._detectors = {

            "Squats":
                SquatDetector(),

            "Push-ups":
                PushUpDetector(),

            "Biceps Curls (Dumbbell)":
                BicepsCurlDetector(),

            "Shoulder Press":
                ShoulderPressDetector(),

            "Lunges":
                LungesDetector(),
        }

        # =====================================================
        # TIMESTAMP
        # =====================================================

        self._frame_timestamps_ms = 0

    # =========================================================
    # METRICS
    # =========================================================

    def set_latest_metrics(
        self,
        metrics
    ):

        with self._lock:

            self._latest_metrics = (
                metrics.copy()
            )

    def get_latest_metrics(self):

        with self._lock:

            if not self._latest_metrics:

                return None

            return self._latest_metrics.copy()

    # =========================================================
    # EXERCISE
    # =========================================================

    def set_exercise(
        self,
        exercise_type
    ):

        with self._lock:

            if (
                exercise_type
                != self._exercise_type
            ):

                self._exercise_type = (
                    exercise_type
                )

                # Reset selected detector
                detector = self._detectors.get(
                    exercise_type
                )

                if detector and hasattr(
                    detector,
                    "reset"
                ):

                    detector.reset()

                self._latest_metrics = {
                    "reps": 0,
                    "pose_detected": False,
                }

                print(
                    f"Exercise changed to: "
                    f"{exercise_type}"
                )

    def get_exercise(self):

        with self._lock:

            return self._exercise_type

    # =========================================================
    # DRAW SKELETON
    # =========================================================

    def _draw_skeleton(
        self,
        img,
        landmarks
    ):

        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:

            if (
                start_idx >= len(landmarks)
                or
                end_idx >= len(landmarks)
            ):

                continue

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if (
                p1.visibility > 0.5
                and
                p2.visibility > 0.5
            ):

                cv2.line(

                    img,

                    (
                        int(p1.x * w),
                        int(p1.y * h)
                    ),

                    (
                        int(p2.x * w),
                        int(p2.y * h)
                    ),

                    (0, 255, 0),

                    3
                )

        for lm in landmarks:

            if lm.visibility > 0.5:

                cv2.circle(

                    img,

                    (
                        int(lm.x * w),
                        int(lm.y * h)
                    ),

                    5,

                    (255, 0, 0),

                    -1
                )

    # =========================================================
    # NO POSE
    # =========================================================

    def _draw_no_pose_warnings(
        self,
        img
    ):

        cv2.putText(

            img,

            "NO POSE DETECTED",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2,

            cv2.LINE_AA
        )

        cv2.putText(

            img,

            "PLEASE FACE THE CAMERA",

            (30, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2,

            cv2.LINE_AA
        )

    # =========================================================
    # OVERLAYS
    # =========================================================

    def _draw_overlays(
        self,
        img,
        metrics,
        ex_type
    ):

        h, _ = img.shape[:2]

        reps = metrics.get(
            "reps",
            0
        )

        cv2.putText(

            img,

            f"REPS: {reps}",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            3
        )

        if ex_type == "Squats":

            self._draw_squats_overlays(
                img,
                metrics
            )

        elif ex_type == "Push-ups":

            self._draw_pushup_overlays(
                img,
                metrics
            )

        elif ex_type == "Biceps Curls (Dumbbell)":

            self._draw_curl_overlays(
                img,
                metrics
            )

        elif ex_type == "Shoulder Press":

            self._draw_press_overlays(
                img,
                metrics
            )

        elif ex_type == "Lunges":

            self._draw_lunge_overlays(
                img,
                metrics
            )

    # =========================================================
    # SQUAT
    # =========================================================

    def _draw_squats_overlays(
        self,
        img,
        metrics
    ):

        h, _ = img.shape[:2]

        depth = metrics.get(
            "depth_status",
            "N/A"
        )

        knee = metrics.get(
            "knee_angle",
            0
        )

        cv2.putText(

            img,

            f"KNEE: {knee} | DEPTH: {depth}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

    # =========================================================
    # PUSHUP
    # =========================================================

    def _draw_pushup_overlays(
        self,
        img,
        metrics
    ):

        h, _ = img.shape[:2]

        body = metrics.get(
            "body_alignment",
            "N/A"
        )

        hip = metrics.get(
            "hip_status",
            "N/A"
        )

        elbow = metrics.get(
            "elbow_angle",
            0
        )

        cv2.putText(

            img,

            f"ELBOW: {elbow} | BODY: {body} | HIP: {hip}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )

    # =========================================================
    # BICEPS
    # =========================================================

    def _draw_curl_overlays(
        self,
        img,
        metrics
    ):

        h, _ = img.shape[:2]

        swing = metrics.get(
            "swing_status",
            "N/A"
        )

        elbow = metrics.get(
            "elbow_angle",
            0
        )

        cv2.putText(

            img,

            f"ELBOW: {elbow} | SWING: {swing}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

    # =========================================================
    # SHOULDER PRESS
    # =========================================================

    def _draw_press_overlays(
        self,
        img,
        metrics
    ):

        h, _ = img.shape[:2]

        extension = metrics.get(
            "extension_status",
            "N/A"
        )

        back = metrics.get(
            "back_arch_status",
            "N/A"
        )

        elbow = metrics.get(
            "elbow_angle",
            0
        )

        cv2.putText(

            img,

            f"ELBOW: {elbow}",

            (20, h - 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

        cv2.putText(

            img,

            f"EXT: {extension}",

            (20, h - 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

        cv2.putText(

            img,

            f"BACK: {back}",

            (20, h - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

    # =========================================================
    # LUNGES
    # =========================================================

    def _draw_lunge_overlays(
        self,
        img,
        metrics
    ):

        h, _ = img.shape[:2]

        balance = metrics.get(
            "balance_status",
            "N/A"
        )

        knee = metrics.get(
            "front_knee_angle",
            0
        )

        cv2.putText(

            img,

            f"KNEE: {knee} | BALANCE: {balance}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def recv(
        self,
        frame
    ):

        # =====================================================
        # FRAME
        # =====================================================

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera

        image = cv2.flip(
            image,
            1
        )

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        # =====================================================
        # RGB
        # =====================================================

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # =====================================================
        # MEDIAPIPE IMAGE
        # =====================================================

        mp_image = mp.Image(

            image_format=
                mp.ImageFormat.SRGB,

            data=rgb_image
        )

        # =====================================================
        # TIMESTAMP
        # =====================================================

        self._frame_timestamps_ms += 30

        # =====================================================
        # POSE DETECTION
        # =====================================================

        result = (
            self._landmarker.detect_for_video(
                mp_image,
                self._frame_timestamps_ms
            )
        )

        # =====================================================
        # POSE FOUND
        # =====================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Draw skeleton

            self._draw_skeleton(
                image,
                landmarks
            )

            # Current exercise

            ex_type = self.get_exercise()

            # Current detector

            detector = self._detectors.get(
                ex_type
            )

            if detector:

                try:

                    # =========================================
                    # PROCESS EXERCISE
                    # =========================================

                    metrics = detector.process(
                        landmarks
                    )

                    # =========================================
                    # POSE DETECTED
                    # =========================================

                    metrics["pose_detected"] = True

                    # =========================================
                    # DEBUG LOG
                    # =========================================

                    print(
                        f"EXERCISE={ex_type} | "
                        f"REPS={metrics.get('reps', 0)} | "
                        f"ELBOW={metrics.get('elbow_angle', 'N/A')} | "
                        f"EXTENSION={metrics.get('extension_status', 'N/A')}"
                    )

                    # =========================================
                    # DRAW
                    # =========================================

                    self._draw_overlays(

                        image,

                        metrics,

                        ex_type
                    )

                    # =========================================
                    # SAVE METRICS
                    # =========================================

                    self.set_latest_metrics(
                        metrics
                    )

                except Exception as e:

                    print(
                        "========================================"
                    )

                    print(
                        "DETECTOR ERROR"
                    )

                    print(
                        type(e).__name__,
                        str(e)
                    )

                    print(
                        "========================================"
                    )

        # =====================================================
        # NO POSE
        # =====================================================

        else:

            self._draw_no_pose_warnings(
                image
            )

            with self._lock:

                self._latest_metrics = {

                    "reps": (
                        self._latest_metrics.get(
                            "reps",
                            0
                        )
                    ),

                    "pose_detected":
                        False,
                }

        # =====================================================
        # RETURN FRAME
        # =====================================================

        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )