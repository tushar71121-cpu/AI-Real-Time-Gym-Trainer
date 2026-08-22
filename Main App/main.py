import os
import time
import pandas as pd
import streamlit as st

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import (
    load_css,
    inject_local_font,
    inject_webrtc_styles,
)
from services.persistence.exercise_repository import (
    init_db,
    get_users_exercises,
)

from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update


# =============================================================
# TURN CONFIGURATION
# =============================================================

def get_turn_credentials():

    username = os.environ.get("TURN_USERNAME", "")
    password = os.environ.get("TURN_PASSWORD", "")

    # Streamlit Cloud Secrets
    if not username:
        try:
            username = st.secrets.get("TURN_USERNAME", "")
        except Exception:
            pass

    if not password:
        try:
            password = st.secrets.get("TURN_PASSWORD", "")
        except Exception:
            pass

    return username, password


def get_rtc_configuration():

    username, password = get_turn_credentials()

    ice_servers = [
        {
            "urls": "stun:stun.relay.metered.ca:80"
        }
    ]

    # Add TURN only when credentials exist
    if username and password:

        ice_servers.extend([
            {
                "urls": "turn:in.relay.metered.ca:80",
                "username": username,
                "credential": password,
            },
            {
                "urls": "turn:in.relay.metered.ca:80?transport=tcp",
                "username": username,
                "credential": password,
            },
            {
                "urls": "turn:in.relay.metered.ca:443",
                "username": username,
                "credential": password,
            },
            {
                "urls": "turns:in.relay.metered.ca:443?transport=tcp",
                "username": username,
                "credential": password,
            },
        ])

    return {
        "iceServers": ice_servers
    }


# =============================================================
# MAIN
# =============================================================

def main():

    # =========================================================
    # PAGE CONFIG
    # =========================================================

    st.set_page_config(
        page_icon="🏋️‍♂️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered",
    )

    # =========================================================
    # STATIC FILES
    # =========================================================

    load_css(
        os.path.join(
            os.getcwd(),
            "static",
            "style.css",
        )
    )

    inject_local_font(
        os.path.join(
            os.getcwd(),
            "static",
            "AdobeClean.otf",
        ),
        "AdobeClean",
    )

    # =========================================================
    # DATABASE
    # =========================================================

    init_db()

    # =========================================================
    # LOGIN
    # =========================================================

    if not render_login_wall():
        return

    initial_session_defaults()

    # =========================================================
    # WORKOUT STATE
    # =========================================================

    workout_started = st.session_state.get(
        "workout_started",
        False,
    )

    # =========================================================
    # SIDEBAR
    # =========================================================

    with st.sidebar:

        st.title("🏋️‍♂️ Apna AI Coach")

        username = st.session_state.get("username")

        if username:
            st.caption(
                f"👤 Login as {username}"
            )

        st.divider()

        st.subheader("Workout Plan")

        # =====================================================
        # BEFORE WORKOUT
        # =====================================================

        if not workout_started:

            plan_exercise = st.selectbox(
                "Exercise",
                options=EXERCISE_OPTIONS,
                key="plan_exercise",
            )

            plan_sets = st.number_input(
                "Sets",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
                key="plan_sets",
            )

            plan_reps = st.number_input(
                "Reps per Set",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="plan_reps",
            )

            st.markdown("")

            start_session_button = st.button(
                "Start Workout",
                width="stretch",
                key="start_session_button",
            )

            if start_session_button:

                st.session_state.exercise_type = (
                    plan_exercise
                )

                st.session_state.target_sets = (
                    int(plan_sets)
                )

                st.session_state.reps_per_set = (
                    int(plan_reps)
                )

                # Reset workout counters
                st.session_state.reps = 0
                st.session_state.sets_completed = 0
                st.session_state.current_set_reps = 0

                st.session_state.workout_completed = False

                st.session_state.last_saved_sets_completed = 0

                st.session_state.last_notified_sets_completed = 0

                st.session_state.last_notified_workout_complete = False

                st.session_state.set_cycle_started_at = (
                    time.time()
                )

                st.session_state.workout_started = True

                # Clear old metrics
                st.session_state.knee_angle = 0
                st.session_state.back_angle = 0
                st.session_state.elbow_angle = 0
                st.session_state.front_knee_angle = 0
                st.session_state.torso_angle = 0

                st.session_state.depth_status = "N/A"
                st.session_state.body_alignment = "N/A"
                st.session_state.hip_status = "N/A"
                st.session_state.shoulder_status = "N/A"
                st.session_state.swing_status = "N/A"
                st.session_state.extension_status = "N/A"
                st.session_state.back_arch_status = "N/A"
                st.session_state.balance_status = "N/A"

                # Clear audio / feedback
                st.session_state.audio_to_play = None
                st.session_state.coach_feedback = None

                st.rerun()

        # =====================================================
        # DURING WORKOUT
        # =====================================================

        else:

            exercise = st.session_state.get(
                "exercise_type",
                "Unknown",
            )

            sets = st.session_state.get(
                "target_sets",
                0,
            )

            reps = st.session_state.get(
                "reps_per_set",
                0,
            )

            st.info(
                f"**{exercise}**\n\n"
                f"{sets} Sets / {reps} Reps"
            )

            end_session_button = st.button(
                "End Workout",
                key="end_session_button",
                width="stretch",
            )

            if end_session_button:

                st.session_state.workout_started = False

                st.session_state.audio_to_play = None
                st.session_state.coach_feedback = None

                st.rerun()

        # =====================================================
        # PROGRESS
        # =====================================================

        if workout_started:

            st.divider()

            total_reps = st.session_state.get(
                "reps",
                0,
            )

            current_set_reps = st.session_state.get(
                "current_set_reps",
                0,
            )

            reps_per_set = st.session_state.get(
                "reps_per_set",
                0,
            )

            sets_completed = st.session_state.get(
                "sets_completed",
                0,
            )

            target_sets = st.session_state.get(
                "target_sets",
                0,
            )

            st.subheader("Progress")

            st.metric(
                "Total Reps",
                total_reps,
            )

            st.metric(
                "Current Set Reps",
                f"{current_set_reps} / {reps_per_set}",
            )

            st.metric(
                "Sets Completed",
                f"{sets_completed} / {target_sets}",
            )

            st.divider()

            # =================================================
            # SQUATS
            # =================================================

            if exercise == "Squats":

                st.subheader("Squat Metrics")

                st.metric(
                    "Knee Angle",
                    f"{st.session_state.get('knee_angle', 0)}°",
                )

                st.metric(
                    "Back Angle",
                    f"{st.session_state.get('back_angle', 0)}°",
                )

                st.metric(
                    "Depth Status",
                    st.session_state.get(
                        "depth_status",
                        "N/A",
                    ),
                )

            # =================================================
            # PUSH UPS
            # =================================================

            elif exercise == "Push-ups":

                st.subheader("Push-up Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Body Alignment",
                    st.session_state.get(
                        "body_alignment",
                        "N/A",
                    ),
                )

                st.metric(
                    "Hip Position",
                    st.session_state.get(
                        "hip_status",
                        "N/A",
                    ),
                )

            # =================================================
            # BICEPS
            # =================================================

            elif exercise == "Biceps Curls (Dumbbell)":

                st.subheader("Curl Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Shoulder Stability",
                    st.session_state.get(
                        "shoulder_status",
                        "N/A",
                    ),
                )

                st.metric(
                    "Swing Detection",
                    st.session_state.get(
                        "swing_status",
                        "N/A",
                    ),
                )

            # =================================================
            # SHOULDER PRESS
            # =================================================

            elif exercise == "Shoulder Press":

                st.subheader("Shoulder Press Metrics")

                st.metric(
                    "Elbow Angle",
                    f"{st.session_state.get('elbow_angle', 0)}°",
                )

                st.metric(
                    "Arm Extension",
                    st.session_state.get(
                        "extension_status",
                        "N/A",
                    ),
                )

                st.metric(
                    "Back Arch",
                    st.session_state.get(
                        "back_arch_status",
                        "N/A",
                    ),
                )

            # =================================================
            # LUNGES
            # =================================================

            elif exercise == "Lunges":

                st.subheader("Lunge Metrics")

                st.metric(
                    "Front Knee Angle",
                    f"{st.session_state.get('front_knee_angle', 0)}°",
                )

                st.metric(
                    "Torso Angle",
                    f"{st.session_state.get('torso_angle', 0)}°",
                )

                st.metric(
                    "Balance Status",
                    st.session_state.get(
                        "balance_status",
                        "N/A",
                    ),
                )

    # =========================================================
    # MAIN TITLE
    # =========================================================

    st.title("AI Real-time GYM Coach")

    st.markdown(
        "#### Real-time pose detection with proactive AI coaching"
    )

    # =========================================================
    # COACH FEEDBACK
    # =========================================================

    if st.session_state.get("coach_feedback"):

        st.success(
            f"🤖 **Coach:** "
            f"{st.session_state.coach_feedback}"
        )

    # =========================================================
    # AUDIO
    # =========================================================

    audio_to_play = st.session_state.get(
        "audio_to_play"
    )

    if audio_to_play:

        try:

            from services.coaching.voice_pipeline import (
                autoplay_audio
            )

            autoplay_audio(
                audio_to_play
            )

            # Prevent repeated playback
            st.session_state.audio_to_play = None

        except Exception as e:

            print(
                f"Audio playback error: {e}"
            )

    # =========================================================
    # CAMERA
    # =========================================================

    if workout_started:

        context = webrtc_streamer(

            key="exercise-analysis",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=VideoProcessorClass,

            rtc_configuration=get_rtc_configuration(),

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

        # =====================================================
        # SYNC METRICS
        # =====================================================

        sync_metrics_update(
            context
        )

        # =====================================================
        # REFRESH METRICS
        # =====================================================

        if context and context.state.playing:

            time.sleep(0.25)

            st.rerun()

        inject_webrtc_styles()

    # =========================================================
    # WORKOUT HISTORY
    # =========================================================

    st.divider()

    st.markdown(
        "#### Workout History"
    )

    user_id = st.session_state.get(
        "user_id",
        0,
    )

    if isinstance(user_id, int):

        history_rows = get_users_exercises(
            user_id
        )

        arr = [
            {
                "Exercise": row["exercise_name"],
                "Reps": row["reps"],
                "Sets": row["sets"],
                "Time (sec)": row["time"],
                "Date": row["created_at"],
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:

            df["Date"] = pd.to_datetime(
                df["Date"]
            ).dt.date

            agg_df = (
                df.groupby(
                    [
                        "Exercise",
                        "Date",
                    ]
                )
                .agg(
                    {
                        "Reps": "sum",
                        "Sets": "sum",
                        "Time (sec)": "sum",
                    }
                )
                .reset_index()
            )

            agg_df.index += 1

            st.table(
                agg_df,
                border="horizontal",
            )

        else:

            st.info(
                "No workout history found."
            )


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":
    main()