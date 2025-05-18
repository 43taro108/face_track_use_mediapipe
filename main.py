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
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 0. Page config
# ------------------------------------------------------
st.set_page_config(page_title="Matplotlib Face Mesh", page_icon="🎨", layout="wide")
st.title("🎨 顔ランドマークメッシュをmatplotlibで表示")

# ------------------------------------------------------
# 1. Sidebar settings
# ------------------------------------------------------
st.sidebar.header("🔧 設定")
max_faces = st.sidebar.number_input("同時検出する顔の数", min_value=1, max_value=4, value=1)
det_conf = st.sidebar.slider("検出信頼度", 0.1, 1.0, 0.5)
track_conf = st.sidebar.slider("追跡信頼度", 0.1, 1.0, 0.5)

# ------------------------------------------------------
# 2. Cached model
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

# ------------------------------------------------------
# 3. Cached landmark detection
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def detect_landmarks(img_bytes, max_faces, det_conf, track_conf):
    # decode image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    # resize for speed
    small = cv2.resize(img, (320, 320))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    # detect landmarks
    mesh = get_face_mesh(max_faces, det_conf, track_conf)
    results = mesh.process(small_rgb)
    if not results.multi_face_landmarks:
        return None
    # collect landmarks
    records = []
    for face_landmarks in results.multi_face_landmarks:
        for lm_id, lm in enumerate(face_landmarks.landmark):
            # scale back to original resolution
            x_px = lm.x * w
            y_px = lm.y * h
            records.append({
                "landmark_id": lm_id,
                "x": x_px,
                "y": y_px
            })
    return pd.DataFrame(records)

# ------------------------------------------------------
# 4. File uploader & display
# ------------------------------------------------------
media = st.file_uploader("画像または動画(mp4/avi)をアップロード", type=["jpg","png","mp4","avi"])
if media:
    data = media.read()
    # if video, extract first frame
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
    # show input
    st.image(data, caption="入力画像", use_column_width=True)
    if st.button("▶ メッシュを描画(mpl)"):
        df = detect_landmarks(data, max_faces, det_conf, track_conf)
        if df is None or df.empty:
            st.warning("顔が検出されませんでした。")
        else:
            # matplotlib 2D mesh
            face_ms = mp.solutions.face_mesh
            mesh_conns = face_ms.FACEMESH_TESSELATION
            fig, ax = plt.subplots(figsize=(6,6))
            # draw edges
            coords = {row.landmark_id: (row.x, row.y) for _,row in df.iterrows()}
            for start, end in mesh_conns:
                if start in coords and end in coords:
                    x0,y0 = coords[start]
                    x1,y1 = coords[end]
                    ax.plot([x0,x1],[y0,y1], linewidth=0.5)
            # scatter points
            ax.scatter(df.x, df.y, s=2)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')
            st.pyplot(fig)
            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV ダウンロード", csv, "landmarks.csv", "text/csv")
