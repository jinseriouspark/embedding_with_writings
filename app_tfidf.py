
# -*- coding: utf-8 -*-
import os, re, io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import umap
import hdbscan
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cafe Posts → TF-IDF Embeddings → Projector", layout="wide")

st.title("네이버 카페 글 · TF-IDF 임베딩 · 3D 시각화 · Projector 내보내기")

# ----------------------
# Helpers
# ----------------------
KOR_STOP = set("그리고 그러나 그래서 또한 또는 등의 및 아주 매우 약간 너무 정말 그냥 거의 자주 더욱 가장 제일 하다 이다 있다 없다 되다 아니다 같은 이러한 그런 저런 우리 여러분 제가 우리는".split())

def normalize_series(s, do_lower=True, do_non_alnum=True, do_ws=True, use_stop=True):
    s = s.fillna("").astype(str)
    if do_lower:
        s = s.str.lower()
    if do_non_alnum:
        s = s.str.replace(r"[^0-9a-zA-Z가-힣\s]", " ", regex=True)
    if use_stop:
        s = s.apply(lambda x: " ".join([t for t in x.split() if t not in KOR_STOP and len(t)>1]))
    if do_ws:
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

# ----------------------
# Data input
# ----------------------
st.subheader("1) CSV 업로드 또는 샘플 사용")

uploaded = st.file_uploader("CSV 업로드 (권장: cafe_posts_250users_5each.csv)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"업로드 완료: {df.shape}")
else:
    st.info("업로드가 없으면 샘플 CSV를 사용합니다.")
    default_path = "cafe_posts_250users_5each.csv"
    if not os.path.exists(default_path):
        # Create a tiny placeholder if sample missing
        df = pd.DataFrame([{"content":"샘플 본문입니다.", "persona":"신입(도전자형)", "business":"네일샵", "nickname":"user_000"}])
    else:
        df = pd.read_csv(default_path)
    st.write(df.head())

# Choose text column
text_cols = [c for c in df.columns if df[c].dtype == "object"]
if not text_cols:
    st.error("문자열(object) 컬럼이 없습니다. 텍스트가 들어있는 컬럼이 필요합니다.")
    st.stop()
default_col = "content_norm" if "content_norm" in df.columns else ("content" if "content" in df.columns else text_cols[0])
text_col = st.selectbox("임베딩에 사용할 텍스트 컬럼", text_cols, index=text_cols.index(default_col) if default_col in text_cols else 0)

with st.expander("전처리 옵션", expanded=True):
    do_lower = st.checkbox("소문자화", True)
    do_non_alnum = st.checkbox("한글/영문/숫자 외 제거", True)
    do_ws = st.checkbox("여러 공백 축약", True)
    use_stop = st.checkbox("간단 한국어 불용어 제거", True)

df["__text__"] = normalize_series(df[text_col], do_lower, do_non_alnum, do_ws, use_stop)

# ----------------------
# Embedding (TF-IDF)
# ----------------------
st.subheader("2) TF-IDF 임베딩")
max_features = st.slider("TF-IDF max_features", 512, 4096, 1024, 128)
vec = TfidfVectorizer(max_features=max_features)
X = vec.fit_transform(df["__text__"].fillna(""))
emb = X.toarray().astype("float32")
st.success(f"임베딩 완료: shape={emb.shape}")

# ----------------------
# 3D UMAP
# ----------------------
st.subheader("3) UMAP 3D 시각화")

n_neighbors = st.slider("n_neighbors", 5, 50, 12, 1)
min_dist = st.slider("min_dist", 0.0, 1.0, 0.2, 0.05)

# NOTE: fit_transform으로 학습 + 좌표 생성, 이후 NEW 포인트는 transform만 호출해 동일 좌표계 유지
um = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
xy3 = um.fit_transform(emb)  # emb: (N, D) 임베딩 행렬

# 색상 기준 컬럼 선택 (텍스트 파생 컬럼 제외)
color_by_cols = [c for c in df.columns if c != "__text__"]
default_color_idx = color_by_cols.index("persona") if "persona" in color_by_cols else 0
color_by = st.selectbox("색상 기준", color_by_cols, index=default_color_idx)

# 시각화용 데이터프레임 구성(원본 df에 좌표 붙이기)
plot_df = df.copy()
plot_df["x"] = xy3[:, 0]
plot_df["y"] = xy3[:, 1]
plot_df["z"] = xy3[:, 2]
plot_df["__color__"] = plot_df[color_by].astype(str)

# hover 정보 구성 (존재하는 컬럼만 포함)
hover_cols = [c for c in ["nickname", "persona", "business", "category", "date"] if c in plot_df.columns]
hover_data = {c: True for c in hover_cols}

# hover_name은 가독성 좋은 텍스트 컬럼으로 (없으면 None)
hover_name_col = text_cols[0] if (len(text_cols) > 0 and text_cols[0] in plot_df.columns) else None

fig3d = px.scatter_3d(
    plot_df,
    x="x", y="y", z="z",
    color="__color__",
    hover_name=hover_name_col,
    hover_data=hover_data
)
fig3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig3d, use_container_width=True)

# ----------------------
# Clustering
# ----------------------
st.subheader("4) 군집화")

alg = st.radio("알고리즘", ["HDBSCAN", "KMeans(k=5)"], horizontal=True)

if alg == "HDBSCAN":
    min_cluster_size = st.slider("min_cluster_size", 5, 50, 15, 1)
    min_samples = st.slider("min_samples", 1, 20, 5, 1)
    clt = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = clt.fit_predict(emb)
else:
    k = st.slider("k", 3, 10, 5, 1)
    clt = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = clt.fit_predict(emb)

df["cluster"] = labels
st.write("클러스터 분포:", df["cluster"].value_counts())

# ----------------------
# TensorFlow Projector export
# ----------------------
st.subheader("5) Projector 파일 내보내기 (vectors.tsv / metadata.tsv)")

# vectors.tsv
vec_buf = io.StringIO()
np.savetxt(vec_buf, emb, delimiter="\t")
st.download_button(
    "vectors.tsv 다운로드",
    data=vec_buf.getvalue(),
    file_name="vectors.tsv",
    mime="text/tab-separated-values"
)

# metadata.tsv
meta_cols = [c for c in ["id", "date", "nickname", "persona", "business", "category", "cluster"] if c in df.columns]
if not meta_cols:
    meta_cols = ["cluster"]
meta_tsv = df[meta_cols].to_csv(sep="\t", index=False)
st.download_button(
    "metadata.tsv 다운로드",
    data=meta_tsv,
    file_name="metadata.tsv",
    mime="text/tab-separated-values"
)

st.caption("projector.tensorflow.org → Load data → vectors.tsv + metadata.tsv 업로드 → Color by = persona/cluster 선택")

# ----------------------
# New post inference
# ----------------------
st.subheader("6) 신규 글 붙여넣기 → 가까운 군집 추정")

new_text = st.text_area("신규 글", height=160, placeholder="여기에 새 글을 붙여넣으세요.")
if st.button("신규 글 분석/표시"):
    if new_text.strip():
        # (1) 전처리 동일 적용
        s = pd.Series([new_text])
        s = normalize_series(s, do_lower, do_non_alnum, do_ws, use_stop)

        # (2) 기존 TF-IDF 벡터라이저(vec)로 변환
        new_vec = vec.transform(s).toarray().astype("float32")

        # (3) 군집 추정: KMeans는 predict, HDBSCAN은 최근접 이웃
        from sklearn.metrics.pairwise import cosine_distances
        if alg.startswith("KMeans"):
            pred = clt.predict(new_vec)[0]
            method = "KMeans centroid"
        else:
            d = cosine_distances(new_vec, emb)[0]
            pred = int(df.iloc[np.argmin(d)]["cluster"])
            method = "nearest neighbor"

        st.success(f"예상 클러스터: {pred} (방법: {method})")

        # (4) 기존 UMAP 모델(um)로 좌표 변환(재학습 없이 동일 공간에 투영)
        new_xy = um.transform(new_vec)  # 핵심: fit_transform로 학습한 동일 um 사용

        # (5) NEW 포인트를 기존 산점도 공간에 추가
        tmp = pd.DataFrame({
            "x": xy3[:, 0], "y": xy3[:, 1], "z": xy3[:, 2],
            "cluster": df["cluster"].astype(str)
        })
        fig_new = px.scatter_3d(tmp, x="x", y="y", z="z", color="cluster")
        fig_new.add_scatter3d(
            x=[new_xy[0, 0]], y=[new_xy[0, 1]], z=[new_xy[0, 2]],
            mode="markers+text",
            marker=dict(size=8, symbol="diamond"),
            text=["NEW"], name="NEW"
        )
        st.plotly_chart(fig_new, use_container_width=True)