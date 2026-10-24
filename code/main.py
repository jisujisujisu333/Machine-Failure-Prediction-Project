
import streamlit as st
import pandas as pd
import plotly.express as px
import re
import unicodedata

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="기계 고장 기초통계 분석",
    page_icon="⚙️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jisujisujisu333/Machine-Failure-Prediction-Project/"
    "refs/heads/main/data/"
    "%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1"
    "%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"
)

st.title("⚙️ 기계 고장 기초통계 분석실")
st.write(
    "기계의 작동 상태를 통계와 그래프로 살펴보고 "
    "고장 여부와 관련된 특징을 분석합니다."
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    errors = []

    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            data = pd.read_csv(DATA_URL, encoding=encoding)
            data.columns = [str(c).strip() for c in data.columns]
            return data
        except Exception as error:
            errors.append(f"{encoding}: {error}")

    raise RuntimeError("\n".join(errors))


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.info(
        "GitHub 저장소가 공개되어 있는지, "
        "CSV 파일 주소가 올바른지 확인해 주세요."
    )
    st.stop()

if df.empty:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()


# ==========================================
# 3. 열 이름 정리 및 변수 확인
# ==========================================
def normalize_name(value):
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", value).lower()


target_names = {
    "machinefailure",
    "failure",
    "고장여부",
    "기계고장여부",
    "기계고장",
    "고장",
    "불량여부",
    "target"
}

target_col = None

for col in df.columns:
    if normalize_name(col) in target_names:
        target_col = col
        break


# ==========================================
# 4. 수치형 변수 자동 변환
# errors="ignore"는 사용하지 않습니다.
# ==========================================
for col in df.columns:
    original = df[col]

    # 전부 비어 있는 열은 그대로 둡니다.
    if original.notna().sum() == 0:
        continue

    converted = pd.to_numeric(original, errors="coerce")

    original_count = original.notna().sum()
    numeric_count = converted.notna().sum()
    numeric_ratio = numeric_count / original_count

    # 값의 80% 이상이 숫자로 변환될 때만 수치형으로 처리합니다.
    if numeric_ratio >= 0.8:
        df[col] = converted


# ==========================================
# 5. 데이터 한눈에 보기
# ==========================================
st.subheader("📌 데이터 한눈에 보기")

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 데이터", f"{len(df):,}행")
col2.metric("전체 변수", f"{len(df.columns):,}개")
col3.metric("수치형 변수", f"{len(numeric_cols):,}개")
col4.metric("결측값", f"{int(df.isna().sum().sum()):,}개")

with st.expander("📖 데이터 열 이름과 미리 보기"):
    st.write("실제 CSV의 열 이름")
    st.write(list(df.columns))
    st.dataframe(df.head(10), use_container_width=True)


# ==========================================
# 6. 기초통계량
# ==========================================
st.divider()
st.subheader("1. 📊 수치형 변수의 기초통계")

if numeric_cols:
    stats = df[numeric_cols].describe().T

    stats = stats.rename(
        columns={
            "count": "자료 수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "1사분위수",
            "50%": "중앙값",
            "75%": "3사분위수",
            "max": "최댓값"
        }
    )

    st.dataframe(stats.round(2), use_container_width=True)

    stat_col = st.selectbox(
        "분포를 확인할 변수를 선택하세요.",
        numeric_cols,
        key="stat_col"
    )

    hist_data = df[stat_col].dropna()

    if not hist_data.empty:
        fig_hist = px.histogram(
            df,
            x=stat_col,
            nbins=30,
            title=f"{stat_col}의 분포",
            labels={stat_col: stat_col},
            color_discrete_sequence=["#3984E8"]
        )

        fig_hist.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title=stat_col,
            yaxis_title="자료 수"
        )

        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("선택한 변수에 분석할 값이 없습니다.")
else:
    st.warning("분석할 수 있는 수치형 변수가 없습니다.")


# ==========================================
# 7. 범주형 변수 분석
# ==========================================
st.divider()
st.subheader("2. 🧩 범주형 변수의 분포")

categorical_cols = []

for col in df.columns:
    if col in numeric_cols:
        continue

    unique_count = df[col].nunique(dropna=True)

    if 1 <= unique_count <= 30:
        categorical_cols.append(col)

if categorical_cols:
    cat_col = st.selectbox(
        "분포를 확인할 범주형 변수를 선택하세요.",
        categorical_cols,
        key="cat_col"
    )

    cat_data = df[cat_col].fillna("결측값").astype(str)

    counts = (
        cat_data.value_counts()
        .rename_axis("항목")
        .reset_index(name="개수")
    )

    fig_cat = px.bar(
        counts,
        x="항목",
        y="개수",
        color="항목",
        text="개수",
        title=f"{cat_col}별 데이터 개수"
    )

    fig_cat.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("분석할 범주형 변수가 없습니다.")


# ==========================================
# 8. 고장 여부 분석
# ==========================================
st.divider()
st.subheader("3. 🔧 고장 여부 분석")

if target_col is not None:
    st.write(f"고장 여부 변수: {target_col}")

    # 원본 값의 형태를 유지하여 문자형 고장 여부도 처리합니다.
    target_data = (
        df[target_col]
        .astype("string")
        .fillna("결측값")
    )

    target_counts = (
        target_data.value_counts()
        .rename_axis("고장 여부")
        .reset_index(name="개수")
    )

    left, right = st.columns(2)

    with left:
        fig_target = px.bar(
            target_counts,
            x="고장 여부",
            y="개수",
            color="고장 여부",
            text="개수",
            title="고장 여부별 데이터 수"
        )

        fig_target.update_layout(
            template="plotly_white",
            height=380,
            showlegend=False
        )

        st.plotly_chart(fig_target, use_container_width=True)

    with right:
        fig_pie = px.pie(
            target_counts,
            names="고장 여부",
            values="개수",
            title="고장 여부별 비율",
            hole=0.4
        )

        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 고장 여부별 수치형 변수 평균 비교
    compare_cols = [
        col for col in numeric_cols
        if col != target_col
    ]

    if compare_cols:
        st.markdown("#### 고장 여부에 따른 평균 비교")

        compare_col = st.selectbox(
            "비교할 수치형 변수를 선택하세요.",
            compare_cols,
            key="compare_col"
        )

        compare_df = pd.DataFrame({
            "고장 여부": target_data,
            "측정값": df[compare_col]
        })

        compare_df = compare_df.dropna(subset=["측정값"])

        mean_df = (
            compare_df.groupby("고장 여부", as_index=False)
            ["측정값"]
            .mean()
            .rename(columns={"측정값": "평균"})
        )

        fig_mean = px.bar(
            mean_df,
            x="고장 여부",
            y="평균",
            color="고장 여부",
            text="평균",
            title=f"고장 여부에 따른 {compare_col} 평균"
        )

        fig_mean.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig_mean.update_layout(
            template="plotly_white",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_mean, use_container_width=True)
    else:
        st.info("평균을 비교할 다른 수치형 변수가 없습니다.")

else:
    st.warning(
        "고장 여부 열을 자동으로 찾지 못했습니다. "
        "실제 열 이름을 확인해 주세요."
    )
    st.write("현재 열 이름:", list(df.columns))


# ==========================================
# 9. 상관관계 분석
# ==========================================
st.divider()
st.subheader("4. 🔗 변수 사이의 상관관계")

id_words = {
    "udi",
    "productid",
    "id",
    "일련번호",
    "식별번호"
}

corr_cols = []

for col in numeric_cols:
    normalized = normalize_name(col)

    if normalized in id_words:
        continue

    if df[col].nunique(dropna=True) > 1:
        corr_cols.append(col)

if len(corr_cols) >= 2:
    corr = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="상관계수 히트맵"
    )

    fig_corr.update_layout(
        template="plotly_white",
        height=max(450, len(corr_cols) * 45)
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption(
        "상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다. "
        "상관관계만으로 직접적인 원인을 단정할 수 없습니다."
    )
else:
    st.info(
        "상관관계 분석에는 값이 서로 다른 수치형 변수가 "
        "2개 이상 필요합니다."
    )


# ==========================================
# 10. 결측값 분석
# ==========================================
st.divider()
st.subheader("5. 🧹 결측값 확인")

missing_df = (
    df.isna().sum()
    .rename_axis("변수")
    .reset_index(name="결측값 수")
)

missing_df = missing_df[
    missing_df["결측값 수"] > 0
].sort_values("결측값 수", ascending=False)

if missing_df.empty:
    st.success("결측값이 없습니다.")
else:
    missing_df["결측 비율(%)"] = (
        missing_df["결측값 수"] / len(df) * 100
    ).round(2)

    st.dataframe(missing_df, use_container_width=True)

    fig_missing = px.bar(
        missing_df,
        x="변수",
        y="결측값 수",
        text="결측값 수",
        title="변수별 결측값 수",
        color="결측값 수",
        color_continuous_scale="Blues"
    )

    fig_missing.update_layout(
        template="plotly_white",
        height=380
    )

    st.plotly_chart(fig_missing, use_container_width=True)


# ==========================================
# 11. 분석 질문
# ==========================================
st.divider()
st.subheader("📝 분석해 볼 질문")

st.markdown(
    """
    - 정상 기계와 고장 난 기계의 평균 온도는 다를까?
    - 공구 마모 시간과 고장 여부는 어떤 관계가 있을까?
    - 회전 속도와 토크는 서로 관련이 있을까?
    - 어떤 변수에서 정상 기계와 고장 기계의 차이가 나타날까?
    """
)

st.caption(
    "이 앱은 기초통계 및 탐색적 데이터 분석을 위한 도구입니다. "
    "관찰된 관계만으로 고장의 직접적인 원인을 확정할 수는 없습니다."
)
