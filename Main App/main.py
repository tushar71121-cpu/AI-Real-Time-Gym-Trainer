import os
import time
import pandas as pd
import streamlit as st

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

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from services.vision.exercise_video_processor import (
    VideoProcessorClass
)

from services.tracking.metrics import (
    sync_metrics_update
)


# =============================================================
# MAIN
# =============================================================

def main():

    # =========================================================
    # PAGE CONFIG
    # =========================================================

    st.set_page_config(
        page_title="AI Real-time GYM Coach",
        page_icon="🏋️‍♂️",
        layout="centered",
        initial_sidebar_state="expanded",
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

    font_path = os.path.join(
        os.getcwd(),
        "static",
        "AdobeClean.otf",
    )

    if os.path.exists(font_path):

        inject_local_font(
            font_path,
            "AdobeClean",
        )

    # =========================================================
    # DATABASE
    # =========================================================

    try:

        init_db()

    except Exception as e:

        st.error(
            f"Database initialization error: {e}"
        )

    # =========================================================
    # LOGIN
    # =========================================================

    if not render_login_wall():

        return

    # =========================================================
    # SESSION DEFAULTS
    # =========================================================

    initial_session_defaults()

    # =========================================================
    # IMPORTANT DEFAULTS
    # =========================================================

    if "workout_started" not in st.session_state:
        st.session_state.workout_started = False

    if "reps" not in st.session_state:
        st.session_state.reps = 0

    if "sets_completed" not in st.session_state:
        st.session_state.sets_completed = 0

    if "current_set_reps" not in st.session_state:
        st.session_state.current_set_reps = 0

    if "workout_completed" not in st.session_state:
        st.session_state.workout_completed = False

    if "last_saved_sets_completed" not in st.session_state:
        st.session_state.last_saved_sets_completed = 0

    if "set_cycle_started_at" not in st.session_state:
        st.session_state.set_cycle_started_at = time.time()

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

        st.title(
            "🏋️‍♂️ Apna AI Coach"
        )

        username = st.session_state.get(
            "username"
        )

        if username:

            st.caption(
                f"👤 Login as {username}"
            )

        st.divider()

        st.subheader(
            "Workout Plan"
        )

        # =====================================================
        # WORKOUT NOT STARTED
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
                max_value=100,
                value=10,
                step=1,
                key="plan_reps",
            )

            st.write("")

            start_button = st.button(
                "▶️ Start Workout",
                width="stretch",
                key="start_session_button",
            )

            if start_button:

                # ---------------------------------------------
                # SAVE WORKOUT SETTINGS
                # ---------------------------------------------

                st.session_state.exercise_type = (
                    plan_exercise
                )

                st.session_state.target_sets = int(
                    plan_sets
                )

                st.session_state.reps_per_set = int(
                    plan_reps
                )

                # ---------------------------------------------
                # RESET COUNTERS
                # ---------------------------------------------

                st.session_state.reps = 0

                st.session_state.sets_completed = 0

                st.session_state.current_set_reps = 0

                st.session_state.workout_completed = False

                st.session_state.last_saved_sets_completed = 0

                st.session_state.last_notified_workout_complete = False

                st.session_state.set_cycle_started_at = (
                    time.time()
                )

                # ---------------------------------------------
                # RESET OLD METRICS
                # ---------------------------------------------

                st.session_state.knee_angle = 0
                st.session_state.back_angle = 0
                st.session_state.depth_status = "N/A"

                st.session_state.elbow_angle = 0
                st.session_state.body_alignment = "N/A"
                st.session_state.hip_status = "N/A"

                st.session_state.shoulder_status = "N/A"
                st.session_state.swing_status = "N/A"

                st.session_state.extension_status = "N/A"
                st.session_state.back_arch_status = "N/A"

                st.session_state.front_knee_angle = 0
                st.session_state.torso_angle = 0
                st.session_state.balance_status = "N/A"

                # ---------------------------------------------
                # START
                # ---------------------------------------------

                st.session_state.workout_started = True

                st.rerun()

        # =====================================================
        # WORKOUT STARTED
        # =====================================================

        else:

            exercise = st.session_state.get(
                "exercise_type",
                "Unknown",
            )

            target_sets = st.session_state.get(
                "target_sets",
                0,
            )

            reps_per_set = st.session_state.get(
                "reps_per_set",
                0,
            )

            st.info(
                f"**{exercise}**\n\n"
                f"{target_sets} Sets × "
                f"{reps_per_set} Reps"
            )

            end_button = st.button(
                "⏹️ End Workout",
                width="stretch",
                key="end_session_button",
            )

            if end_button:

                st.session_state.workout_started = False

                st.rerun()

        # =====================================================
        # PROGRESS
        # =====================================================

        if workout_started:

            st.divider()

            st.subheader(
                "Progress"
            )

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

            # =================================================
            # EXERCISE METRICS
            # =================================================

            exercise = st.session_state.get(
                "exercise_type"
            )

            st.divider()

            # -------------------------------------------------
            # SQUATS
            # -------------------------------------------------

            if exercise == "Squats":

                st.subheader(
                    "Squat Metrics"
                )

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

            # -------------------------------------------------
            # PUSH UPS
            # -------------------------------------------------

            elif exercise == "Push-ups":

                st.subheader(
                    "Push-up Metrics"
                )

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

            # -------------------------------------------------
            # BICEPS
            # -------------------------------------------------

            elif exercise == "Biceps Curls (Dumbbell)":

                st.subheader(
                    "Biceps Curl Metrics"
                )

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

            # -------------------------------------------------
            # SHOULDER PRESS
            # -------------------------------------------------

            elif exercise == "Shoulder Press":

                st.subheader(
                    "Shoulder Press Metrics"
                )

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

            # -------------------------------------------------
            # LUNGES
            # -------------------------------------------------

            elif exercise == "Lunges":

                st.subheader(
                    "Lunge Metrics"
                )

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

    st.title(
        "AI Real-time GYM Coach"
    )

    st.markdown(
        "#### Real-time pose detection with proactive AI coaching"
    )

    # =========================================================
    # CAMERA
    # =========================================================

    if not workout_started:

        st.markdown(
            """
            <div style="
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 45px 25px;
                text-align: center;
                color: #888;
                margin-top: 30px;
                margin-bottom: 30px;
            ">
                <h2 style="color:#ccc;">
                    👈 Set your workout plan
                </h2>

                <p>
                    Select exercise, sets and reps
                    from the sidebar and start your workout.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.subheader(
            "📹 Live Camera"
        )

        # =====================================================
        # WEBRTC
        # =====================================================

        context = webrtc_streamer(

            key="exercise-analysis",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=VideoProcessorClass,

            rtc_configuration={

                "iceServers": [

                    {
                        "urls": [
                            "stun:stun.relay.metered.ca:80"
                        ],
                    },

                    {
                        "urls": [
                            "turn:in.relay.metered.ca:80",
                            "turn:in.relay.metered.ca:80?transport=tcp",
                            "turn:in.relay.metered.ca:443",
                            "turns:in.relay.metered.ca:443?transport=tcp",
                        ],

                        # IMPORTANT:
                        # Put your TURN credentials in
                        # Streamlit Secrets instead of
                        # hardcoding them here.

                        "username": st.secrets.get(
                            "TURN_USERNAME",
                            "",
                        ),

                        "credential": st.secrets.get(
                            "TURN_PASSWORD",
                            "",
                        ),
                    },
                ],
            },

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

        # =====================================================
        # SYNC METRICS
        # =====================================================

        if context is not None:

            try:

                sync_metrics_update(
                    context
                )

            except Exception as e:

                st.error(
                    f"Metrics error: {type(e).__name__}: {e}"
                )

        # =====================================================
        # DEBUG METRICS
        # =====================================================

        # Ye temporary debug section hai.
        # Agar metrics fir bhi 0 aaye to isse pata chalega
        # processor kya data bhej raha hai.

        if context is not None:

            processor = getattr(
                context,
                "video_processor",
                None,
            )

            if processor is not None:

                latest = processor.get_latest_metrics()

                if latest:

                    with st.expander(
                        "🔧 Debug Metrics"
                    ):

                        st.json(
                            latest
                        )

        # =====================================================
        # REFRESH
        # =====================================================

        if (
            context is not None
            and hasattr(context, "state")
            and context.state.playing
        ):

            time.sleep(0.5)

            st.rerun()

        inject_webrtc_styles()

    # =========================================================
    # WORKOUT HISTORY
    # =========================================================

    st.divider()

    st.subheader(
        "📊 Workout History"
    )

    user_id = st.session_state.get(
        "user_id",
        0,
    )

    if isinstance(user_id, int):

        try:

            history_rows = get_users_exercises(
                user_id
            )

        except Exception as e:

            history_rows = []

            st.warning(
                f"Could not load workout history: {e}"
            )

        if history_rows:

            arr = []

            for row in history_rows:

                arr.append(
                    {
                        "Exercise": row["exercise_name"],
                        "Reps": row["reps"],
                        "Sets": row["sets"],
                        "Time (sec)": row["time"],
                        "Date": row["created_at"],
                    }
                )

            df = pd.DataFrame(
                arr
            )

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

                st.dataframe(
                    agg_df,
                    width="stretch",
                    hide_index=False,
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