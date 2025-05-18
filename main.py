# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:16:11 2025

@author: ktrpt
"""

import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tempfile

# ------------------------------------------------------
# 0. Page config
# ------------------------------------------------------
st.set_page_config(page_title="Fast Face Landmark Mesh", page_icon="⚡", layout="wide")
st.title("⚡ 高速顔ランドマークメッシュ抽出")

# ------------------------------------------------------
# 1. Sidebar settings
# ------------------------------------------------------
st.sidebar.header("🔧 設定")
max_faces = st.sidebar.number_input("同時検出する顔の数", min_value=1, max_value=4, value=1)
det_conf = st.sidebar.slider("検出信頼度", 0.1, 1.0, 0.5)
track_conf = st.sidebar.slider("追跡信頼度", 0.1, 1.0, 0.5)

# ------------------------------------------------------
# 2. Caching model creation
# ------------------------------------------------------
@st.cache_resource
def get_face_mesh(max_faces, det_conf, track_conf):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=False,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf
    )

mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ------------------------------------------------------
# 3. Landmark detection & drawing cache
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def detect_and_draw(img_bytes, max_faces, det_conf, track_conf):
    # Decode
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    # Resize for detection
    small = cv2.resize(img, (320, 320))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    # Detect
    mesh = get_face_mesh(max_faces, det_conf, track_conf)
    results = mesh.process(small_rgb)
    # Prepare output
    canvas = img.copy()
    records = []
    if not results.multi_face_landmarks:
        return None, None
    for face_landmarks in results.multi_face_landmarks:
        # Draw mesh on full-res canvas
        mp_drawing.draw_landmarks(
            canvas,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
        )
        # Collect points (scaled back to full size)
        for lm_id, lm in enumerate(face_landmarks.landmark):
            x_px = lm.x * w  # normalized to full size
            y_px = lm.y * h
            z_rel = lm.z * max(w, h)
            records.append({
                "landmark_id": lm_id,
                "x": x_px,
                "y": y_px,
                "z": z_rel
            })
    df = pd.DataFrame(records)
    # Encode canvas image
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return df, canvas_rgb

# ------------------------------------------------------
# 4. File uploader & processing
# ------------------------------------------------------
media = st.file_uploader("画像または動画1フレームをアップロード", type=["jpg","png","mp4","avi"])
if media:
    data = media.read()
    # If video, extract first frame
    if media.name.lower().endswith((".mp4",".avi")):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(data)
        cap = cv2.VideoCapture(tmp.name)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("動画からフレームを読み込めませんでした。")
            st.stop()
        _, img_encoded = cv2.imencode('.jpg', frame)
        data = img_encoded.tobytes()
    # Display input
    st.image(data, caption="入力画像", use_column_width=True)
    # Process
    if st.button("▶ 高速抽出＆描画"):
        df, canvas = detect_and_draw(data, max_faces, det_conf, track_conf)
        if df is None:
            st.warning("顔が検出されませんでした。")
        else:
            st.image(canvas, caption="メッシュ描画結果", use_column_width=True)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV DL", csv, "landmarks.csv", "text/csv")
