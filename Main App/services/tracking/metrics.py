import time
import streamlit as st

from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


def sync_metrics_update(context):

    # ---------------------------------------------------------
    # CHECK WEBRTC CONTEXT
    # ---------------------------------------------------------

    if context is None:
        return

    if not hasattr(context, "state"):
        return

    if not context.state.playing:
        return

    # ---------------------------------------------------------
    # GET VIDEO PROCESSOR
    # ---------------------------------------------------------

    processor = getattr(
        context,
        "video_processor",
        None
    )

    if processor is None:
        return

    # ---------------------------------------------------------
    # GET SELECTED EXERCISE
    # ---------------------------------------------------------

    exercise = st.session_state.get(
        "exercise_type"
    )

    if not exercise:
        return

    # IMPORTANT:
    # Tell processor which exercise user selected
    processor.set_exercise(exercise)

    # ---------------------------------------------------------
    # GET LATEST METRICS
    # ---------------------------------------------------------

    latest_metrics = processor.get_latest_metrics()

    if latest_metrics is None:
        return

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------

    # Uncomment temporarily if needed:
    #
    # st.write("DEBUG METRICS:", latest_metrics)

    # ---------------------------------------------------------
    # REPS
    # ---------------------------------------------------------

    reps = latest_metrics.get(
        "reps",
        0
    )

    if reps is None:
        reps = 0

    try:
        reps = int(reps)
    except Exception:
        reps = 0

    st.session_state.reps = reps

    # ---------------------------------------------------------
    # EXERCISE METRICS
    # ---------------------------------------------------------

    fields = METRICS_FIELDS.get(
        exercise,
        {}
    )

    for key, default in fields.items():

        value = latest_metrics.get(
            key,
            default
        )

        st.session_state[key] = value

    # ---------------------------------------------------------
    # SET / REP CALCULATION
    # ---------------------------------------------------------

    reps_per_set = int(
        st.session_state.get(
            "reps_per_set",
            0
        )
    )

    target_sets = int(
        st.session_state.get(
            "target_sets",
            0
        )
    )

    if reps_per_set > 0 and target_sets > 0:

        sets_completed = reps // reps_per_set

        current_set_reps = reps % reps_per_set

        # Don't allow sets to exceed target
        sets_completed = min(
            sets_completed,
            target_sets
        )

        workout_completed = (
            sets_completed >= target_sets
        )

    else:

        sets_completed = 0
        current_set_reps = reps
        workout_completed = False

    # ---------------------------------------------------------
    # SAVE UI STATE
    # ---------------------------------------------------------

    st.session_state.sets_completed = (
        sets_completed
    )

    st.session_state.current_set_reps = (
        current_set_reps
    )

    st.session_state.workout_completed = (
        workout_completed
    )

    # ---------------------------------------------------------
    # SAVE COMPLETED SET
    # ---------------------------------------------------------

    last_saved_sets = int(
        st.session_state.get(
            "last_saved_sets_completed",
            0
        )
    )

    if (
        target_sets > 0
        and reps_per_set > 0
        and sets_completed > last_saved_sets
    ):

        newly_completed = (
            sets_completed
            - last_saved_sets
        )

        now_ts = time.time()

        started_at = st.session_state.get(
            "set_cycle_started_at",
            now_ts
        )

        time_taken = (
            now_ts - started_at
        )

        user_id = st.session_state.get(
            "user_id",
            0
        )

        try:

            add_exercise(
                user_id,
                exercise,
                newly_completed * reps_per_set,
                newly_completed,
                time_taken
            )

        except Exception as e:

            print(
                f"Database save error: {e}"
            )

        st.session_state.set_cycle_started_at = (
            now_ts
        )

        st.session_state.last_saved_sets_completed = (
            sets_completed
        )

    # ---------------------------------------------------------
    # WORKOUT COMPLETE
    # ---------------------------------------------------------

    if workout_completed:

        if not st.session_state.get(
            "last_notified_workout_complete",
            False
        ):

            st.session_state.last_notified_workout_complete = True