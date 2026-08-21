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
        self._latest_metrics = None
        self._exercise_type = "Squats"

        # =====================================================
        # PROJECT ROOT
        # =====================================================

        # Current file:
        #
        # ai-real-time-gym-trainer/
        # └── Main App/
        #     └── services/
        #         └── vision/
        #             └── exercise_video_processor.py
        #
        # We need to go:
        #
        # vision -> services -> Main App -> project root

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
        # MEDIAPIPE MODEL PATH
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

        # =====================================================
        # CHECK MODEL
        # =====================================================

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
        # MEDIAPIPE BASE OPTIONS
        # =====================================================

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        # =====================================================
        # POSE LANDMARKER OPTIONS
        # =====================================================

        options = vision.PoseLandmarkerOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.VIDEO,

            min_pose_detection_confidence=0.7,

            min_pose_presence_confidence=0.7,

            min_tracking_confidence=0.7,

            output_segmentation_masks=False,
        )

        # =====================================================
        # CREATE POSE LANDMARKER
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
        # EXERCISE DETECTORS
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
        # VIDEO TIMESTAMP
        # =====================================================

        self._frame_timestamps_ms = 0

    # =========================================================
    # METRICS
    # =========================================================

    def set_latest_metrics(self, metrics):

        with self._lock:

            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):

        with self._lock:

            if self._latest_metrics is None:

                return None

            return self._latest_metrics.copy()

    # =========================================================
    # EXERCISE
    # =========================================================

    def set_exercise(self, exercise_type):

        with self._lock:

            self._exercise_type = exercise_type

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

        # -----------------------------------------------------
        # SKELETON CONNECTIONS
        # -----------------------------------------------------

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
                p1.visibility > 0.7
                and
                p2.visibility > 0.7
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

                    4
                )

        # -----------------------------------------------------
        # LANDMARK POINTS
        # -----------------------------------------------------

        for lm in landmarks:

            if lm.visibility > 0.7:

                cv2.circle(

                    img,

                    (
                        int(lm.x * w),
                        int(lm.y * h)
                    ),

                    6,

                    (255, 0, 0),

                    -1
                )

    # =========================================================
    # NO POSE WARNING
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
    # SQUAT OVERLAY
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

        cv2.putText(

            img,

            f"DEPTH: {depth}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

    # =========================================================
    # PUSHUP OVERLAY
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

        cv2.putText(

            img,

            f"BODY: {body} | HIP: {hip}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

    # =========================================================
    # BICEPS CURL OVERLAY
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

        cv2.putText(

            img,

            f"SWING: {swing}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

    # =========================================================
    # SHOULDER PRESS OVERLAY
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

        cv2.putText(

            img,

            f"EXT: {extension} | BACK: {back}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

    # =========================================================
    # LUNGE OVERLAY
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

        cv2.putText(

            img,

            f"BALANCE: {balance}",

            (20, h - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

    # =========================================================
    # PROCESS VIDEO FRAME
    # =========================================================

    def recv(
        self,
        frame
    ):

        # -----------------------------------------------------
        # FRAME TO NUMPY
        # -----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )

        # -----------------------------------------------------
        # MIRROR CAMERA
        # -----------------------------------------------------

        image = cv2.flip(
            image,
            1
        )

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        # -----------------------------------------------------
        # BGR -> RGB
        # -----------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------------------------------
        # CREATE MEDIAPIPE IMAGE
        # -----------------------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_image
        )

        # -----------------------------------------------------
        # TIMESTAMP
        # -----------------------------------------------------

        self._frame_timestamps_ms += 30

        # -----------------------------------------------------
        # DETECT POSE
        # -----------------------------------------------------

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

            # Get selected exercise
            ex_type = self.get_exercise()

            # Get detector
            detector = self._detectors.get(
                ex_type
            )

            if detector:

                try:

                    metrics = detector.process(
                        landmarks
                    )

                    metrics["pose_detected"] = True

                    # Draw metrics
                    self._draw_overlays(
                        image,
                        metrics,
                        ex_type
                    )

                    # Save metrics
                    self.set_latest_metrics(
                        metrics
                    )

                except Exception as e:

                    print(
                        f"Detector error: {e}"
                    )

        # =====================================================
        # NO POSE
        # =====================================================

        else:

            self._draw_no_pose_warnings(
                image
            )

            with self._lock:

                if self._latest_metrics is not None:

                    self._latest_metrics[
                        "pose_detected"
                    ] = False

                else:

                    self._latest_metrics = {
                        "pose_detected": False
                    }

        # =====================================================
        # RETURN FRAME
        # =====================================================

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )