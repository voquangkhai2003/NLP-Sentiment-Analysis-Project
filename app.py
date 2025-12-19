import streamlit as st
import pandas as pd
import joblib
import re
from pyvi import ViTokenizer
from datetime import datetime
import plotly.express as px

# =====================================
# 1. TIỀN XỬ LÝ
# =====================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return ViTokenizer.tokenize(text)

# =====================================
# 2. LOAD MODEL & DATA
# =====================================
@st.cache_resource
def load_assets():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df_raw = pd.read_csv("datasetnew.csv")
    return model, vectorizer, df_raw

# =====================================
# 3. MAIN APP
# =====================================
def main():
    st.set_page_config("Co.opmart Sentiment Monitor", layout="wide")
    st.title("Hệ thống Giám sát Cảm xúc Khách hàng Co.opmart")

    model, vectorizer, df_raw = load_assets()

    # =================================
    # KHỞI TẠO DỮ LIỆU
    # =================================
    if "data" not in st.session_state:
        df = df_raw[["title", "comment", "stars"]].copy()
        df.columns = ["Chi nhánh", "Nội dung", "Sao"]

        def map_star(s):
            if s <= 2: return "Tiêu cực"
            if s >= 4: return "Tích cực"
            return "Trung tính"

        df["Sentiment"] = df["Sao"].apply(map_star)
        df["Time"] = datetime.now()
        st.session_state.data = df

    # =================================
    # SIDEBAR - NHẬP REVIEW
    # =================================
    with st.sidebar:
        st.header("📥 Nhập phản hồi mới")

        branches = sorted(st.session_state.data["Chi nhánh"].unique())
        branch = st.selectbox("Chi nhánh", branches)
        stars = st.select_slider("Số sao", [1,2,3,4,5], value=5)
        text = st.text_area("Nội dung phản hồi")

        if st.button("Gửi"):
            if text.strip():
                clean = preprocess_text(text)
                vec = vectorizer.transform([clean])
                ml_pred = model.predict(vec)[0]

                if stars <= 2:
                    sentiment = "Tiêu cực"
                elif stars >= 4:
                    sentiment = "Tích cực"
                else:
                    sentiment = ml_pred

                new_row = pd.DataFrame([{
                    "Time": datetime.now(),
                    "Chi nhánh": branch,
                    "Nội dung": text,
                    "Sao": stars,
                    "Sentiment": sentiment
                }])

                st.session_state.data = pd.concat(
                    [new_row, st.session_state.data],
                    ignore_index=True
                )

                st.toast(f"✔ Đã ghi nhận: {sentiment}")
            else:
                st.warning("Không được để trống nội dung")

    # =================================
    # DASHBOARD
    # =================================
    data = st.session_state.data.copy()

    # -------- METRIC TỔNG --------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng review", len(data))
    m2.metric("Tích cực ✅", (data["Sentiment"]=="Tích cực").sum())
    m3.metric("Trung tính 😐", (data["Sentiment"]=="Trung tính").sum())
    m4.metric("Tiêu cực ❌", (data["Sentiment"]=="Tiêu cực").sum())

    st.divider()
    # =================================
    # BIỂU ĐỒ CỘT CHỒNG (STACKED BAR)
    # =================================
    st.subheader("📊 Phân bố cảm xúc theo chi nhánh")

    stacked_data = (
        data.groupby(["Chi nhánh", "Sentiment"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        stacked_data,
        x="Chi nhánh",
        y="Count",
        color="Sentiment",
        text="Count",
        color_discrete_map={
            "Tích cực": "green",
            "Trung tính": "lightgray",
            "Tiêu cực": "red"
        }
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Chi nhánh",
        yaxis_title="Số lượng phản hồi",
        legend_title="Cảm xúc"
    )

    st.plotly_chart(fig, use_container_width=True)


    # =================================
    # LEADERBOARD
    # =================================
    st.subheader("🏆 Bảng xếp hạng chi nhánh")

    leaderboard = (
        data.groupby("Chi nhánh")["Sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    leaderboard["Score"] = (
        leaderboard.get("Tích cực",0)
        - leaderboard.get("Tiêu cực",0)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌟 Top 5 phục vụ tốt nhất")
        st.dataframe(
            leaderboard.sort_values("Score", ascending=False)
            .head(5)[["Score"]],
            use_container_width=True
        )

    with col2:
        st.markdown("### ⚠️ Top 5 cần cải thiện")
        st.dataframe(
            leaderboard.sort_values("Score")
            .head(5)[["Score"]],
            use_container_width=True
        )

    # =================================
    # CHỌN CHI NHÁNH → PANEL RIÊNG
    # =================================
    st.subheader("🏬 Theo dõi chi nhánh cụ thể")

    selected_branch = st.selectbox(
        "Chọn chi nhánh",
        sorted(data["Chi nhánh"].unique())
    )

    branch_data = data[data["Chi nhánh"] == selected_branch]

    colb1, colb2 = st.columns([1,1])

    with colb1:
        st.markdown("### 📊 Phân bố sentiment")
        st.bar_chart(branch_data["Sentiment"].value_counts())

    with colb2:
        st.markdown("### ⭐ Phân bố đánh giá sao")
        st.bar_chart(branch_data["Sao"].value_counts())

    st.markdown("### 📝 Review mới nhất")
    st.dataframe(
        branch_data.sort_values("Time", ascending=False).head(15),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
