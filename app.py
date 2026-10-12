"""Dashboard: pick or upload a video, run the detector, browse the violations.

    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from ultralytics import YOLO

from common import BIKE_MODEL, HEAD_MODEL, load_yoloe, pick_device
from detect_violations import LABELS, NO_HELMET, TRIPLE, analyse

CLIPS, UPLOADS, OUTPUTS = Path("data/clips"), Path("data/uploads"), Path("outputs")

st.set_page_config(page_title="Two-Wheeler Violation Detection", page_icon="🏍️", layout="wide")


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_models(use_yoloe):
    return YOLO(BIKE_MODEL), YOLO(HEAD_MODEL), load_yoloe() if use_yoloe else None


def finished_runs():
    return sorted(p.name for p in OUTPUTS.glob("*/annotated.mp4"))


# ---------------------------------------------------------------- sidebar: choose what to look at
st.sidebar.title("🏍️ Violation Detection")
st.sidebar.caption("Riders without helmets and triple riding, found in traffic video.")
source = st.sidebar.radio("Video", ["Sample clip", "Upload a video", "Past results"])
use_yoloe = st.sidebar.toggle("YOLOE helmet check (slower, fewer false alarms)", value=True)

video = None
if source == "Sample clip":
    clips = sorted(CLIPS.glob("*.mp4"))
    if clips:
        video = st.sidebar.selectbox("Clip", clips, format_func=lambda p: p.name)
    else:
        st.sidebar.warning(f"No videos in {CLIPS}/ yet.")
elif source == "Upload a video":
    up = st.sidebar.file_uploader("mp4 / mov / avi", type=["mp4", "mov", "avi"])
    if up:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        video = UPLOADS / up.name
        video.write_bytes(up.getbuffer())

name = None
if source == "Past results":
    runs = finished_runs()
    name = Path(st.sidebar.selectbox("Result", runs)).stem if runs else None
    if not runs:
        st.sidebar.info("Nothing analysed yet.")
elif video:
    if st.sidebar.button("Analyse", type="primary", use_container_width=True):
        bike_model, head_model, yoloe = load_models(use_yoloe)
        bar = st.progress(0.0, text=f"Analysing {video.name}...")
        _, n_bikes = analyse(video, bike_model, head_model, yoloe, pick_device(), show=False,
                             progress=lambda f: bar.progress(min(f, 1.0), text=f"Analysing {video.name}..."))
        bar.empty()
        st.session_state.setdefault("bikes_checked", {})[video.stem] = n_bikes
    if (OUTPUTS / video.stem / "annotated.mp4").exists():      # analysed now or earlier
        name = video.stem

# ---------------------------------------------------------------- main area: results
st.title("Two-Wheeler Violation Detection")
if not name:
    st.info("Choose a video in the sidebar and press **Analyse**, or open **Past results**.")
    st.stop()

out = OUTPUTS / name
csv = out / "violations.csv"
df = pd.read_csv(csv) if csv.exists() and csv.stat().st_size > 0 else pd.DataFrame()
n_no_helmet = int((df["violation"] == NO_HELMET).sum()) if len(df) else 0
n_triple = int((df["violation"] == TRIPLE).sum()) if len(df) else 0
bikes = st.session_state.get("bikes_checked", {}).get(name)

cols = st.columns(4)
cols[0].metric("Video", name[:22])
cols[1].metric("No helmet", n_no_helmet)
cols[2].metric("Triple riding", n_triple)
cols[3].metric("Bikes checked", bikes if bikes is not None else "n/a")

left, right = st.columns([3, 2])
with left:
    st.subheader("Annotated video")
    st.video(str(out / "annotated.mp4"))
    st.caption("Green box = helmet, red box = bare head. A red bike label means the bike was flagged. "
               "Grey = parked or not a motorcycle, so not judged.")
with right:
    st.subheader("Violations")
    if df.empty:
        st.success("No violations found in this video.")
    else:
        view = df[["time", "bike", "violation", "riders", "confidence"]].copy()
        view["violation"] = view["violation"].map(LABELS)
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.download_button("Download violations.csv", csv.read_bytes(), f"{name}_violations.csv", "text/csv")

if not df.empty:
    st.subheader("Evidence")
    for row_start in range(0, len(df), 4):
        for col, (_, r) in zip(st.columns(4), df.iloc[row_start:row_start + 4].iterrows()):
            with col:
                if Path(r["evidence"]).exists():
                    st.image(r["evidence"], use_container_width=True)
                st.markdown(f"**{LABELS[r['violation']]}**  \nbike #{r['bike']} at {r['time']}  \n"
                            f"confidence {r['confidence']}")
