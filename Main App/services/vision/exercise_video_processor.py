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
        # MODEL PATH
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
            "MediaPipe Model:"
        )

        print(
            model_path
        )

        print(
            "========================================"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        # =====================================================
        # MEDIAPIPE
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

        self._landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

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

    def set_latest_metrics(self, metrics):

        if metrics is None:
            return

        with self._lock:

            self._latest_metrics = dict(
                metrics
            )

    def get_latest_metrics(self):

        with self._lock:

            if self._latest_metrics is None:
                return None

            return dict(
                self._latest_metrics
            )

    # =========================================================
    # EXERCISE
    # =========================================================

    def set_exercise(self, exercise_type):

        if not exercise_type:
            return

        with self._lock:

            if exercise_type != self._exercise_type:

                self._exercise_type = exercise_type

                # Reset selected detector
                detector = self._detectors.get(
                    exercise_type
                )

                if detector and hasattr(
                    detector,
                    "reset"
                ):
                    detector.reset()

                # Reset metrics
                self._latest_metrics = {
                    "reps": 0,
                    "pose_detected": False,
                }

    def get_exercise(self):

        with self._lock:

            return self._exercise_type

    # =========================================================
    # SKELETON
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
                p1.visibility >= 0.5
                and
                p2.visibility >= 0.5
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

            if lm.visibility >= 0.5:

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

            2
        )

        cv2.putText(

            img,

            "PLEASE FACE THE CAMERA",

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )

    # =========================================================
    # OVERLAY
    # =========================================================

    def _draw_overlays(
        self,
        img,
        metrics,
        exercise
    ):

        h, _ = img.shape[:2]

        reps = metrics.get(
            "reps",
            0
        )

        cv2.putText(

            img,

            f"REPS: {reps}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            3
        )

        # -----------------------------------------------------
        # SHOULDER PRESS
        # -----------------------------------------------------

        if exercise == "Shoulder Press":

            elbow = metrics.get(
                "elbow_angle",
                0
            )

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

                f"ELBOW: {elbow}",

                (20, h - 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            cv2.putText(

                img,

                f"EXT: {extension}",

                (20, h - 55),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )

            cv2.putText(

                img,

                f"BACK: {back}",

                (20, h - 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )

        # -----------------------------------------------------
        # SQUATS
        # -----------------------------------------------------

        elif exercise == "Squats":

            angle = metrics.get(
                "knee_angle",
                0
            )

            depth = metrics.get(
                "depth_status",
                "N/A"
            )

            cv2.putText(

                img,

                f"KNEE: {angle}",

                (20, h - 55),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            cv2.putText(

                img,

                f"DEPTH: {depth}",

                (20, h - 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )

    # =========================================================
    # RECEIVE FRAME
    # =========================================================

    def recv(
        self,
        frame
    ):

        try:

            # -------------------------------------------------
            # FRAME
            # -------------------------------------------------

            image = frame.to_ndarray(
                format="bgr24"
            )

            image = cv2.flip(
                image,
                1
            )

            image = np.asarray(
                image,
                dtype=np.uint8
            )

            # -------------------------------------------------
            # RGB
            # -------------------------------------------------

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb_image
            )

            # -------------------------------------------------
            # TIMESTAMP
            # -------------------------------------------------

            self._frame_timestamps_ms += 33

            # -------------------------------------------------
            # POSE
            # -------------------------------------------------

            result = (
                self._landmarker.detect_for_video(
                    mp_image,
                    self._frame_timestamps_ms
                )
            )

            # -------------------------------------------------
            # POSE FOUND
            # -------------------------------------------------

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                self._draw_skeleton(
                    image,
                    landmarks
                )

                exercise = self.get_exercise()

                detector = self._detectors.get(
                    exercise
                )

                if detector:

                    try:

                        metrics = detector.process(
                            landmarks
                        )

                        if metrics is None:
                            metrics = {}

                        metrics["pose_detected"] = True

                        self.set_latest_metrics(
                            metrics
                        )

                        self._draw_overlays(
                            image,
                            metrics,
                            exercise
                        )

                    except Exception as detector_error:

                        print(
                            "================================"
                        )

                        print(
                            "DETECTOR ERROR:"
                        )

                        print(
                            repr(detector_error)
                        )

                        print(
                            "================================"
                        )

                        # Keep previous metrics alive
                        previous = (
                            self.get_latest_metrics()
                            or {}
                        )

                        previous[
                            "pose_detected"
                        ] = True

                        self.set_latest_metrics(
                            previous
                        )

            # -------------------------------------------------
            # NO POSE
            # -------------------------------------------------

            else:

                self._draw_no_pose_warnings(
                    image
                )

                previous = (
                    self.get_latest_metrics()
                    or {}
                )

                previous[
                    "pose_detected"
                ] = False

                self.set_latest_metrics(
                    previous
                )

        except Exception as e:

            print(
                "================================"
            )

            print(
                "VIDEO PROCESSOR ERROR:"
            )

            print(
                repr(e)
            )

            print(
                "================================"
            )

        # -----------------------------------------------------
        # RETURN FRAME
        # -----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )