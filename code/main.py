
import streamlit as st
import pandas as pd
import plotly.express as px
import re
import unicodedata

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="기계 고장 기초통계 분석실",
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
    "기계의 작동 상태를 기초통계와 그래프로 분석합니다. "
    "각 변수의 의미를 확인하고 고장 여부와 관련된 특징을 살펴보세요."
)


# ==========================================
# 2. 영어 변수명 → 한국어 변수명
# ==========================================
COLUMN_NAMES = {
    "UDI": "데이터 식별 번호",
    "Product ID": "제품 식별 번호",
    "Type": "제품 유형",
    "Air temperature [K]": "공기 온도(K)",
    "Process temperature [K]": "공정 온도(K)",
    "Rotational speed [rpm]": "회전 속도(rpm)",
    "Torque [Nm]": "토크(N·m)",
    "Tool wear [min]": "공구 마모 시간(분)",
    "Machine failure": "기계 고장 여부",
    "TWF": "공구 마모 고장",
    "HDF": "열 방출 고장",
    "PWF": "동력 고장",
    "OSF": "과부하 고장",
    "RNF": "무작위 고장"
}

VARIABLE_MEANINGS = {
    "데이터 식별 번호": "각 데이터 행을 구별하기 위한 번호입니다. 기계 상태를 나타내는 측정값은 아닙니다.",
    "제품 식별 번호": "각 제품을 구별하기 위한 번호입니다.",
    "제품 유형": "제품의 종류를 나타냅니다. 원본 데이터에서는 L, M, H로 구분됩니다.",
    "공기 온도(K)": "주변 공기의 절대온도입니다. K는 켈빈 단위입니다.",
    "공정 온도(K)": "기계의 작업 과정과 관련된 절대온도입니다.",
    "회전 속도(rpm)": "기계가 1분 동안 회전하는 횟수입니다.",
    "토크(N·m)": "회전 운동을 일으키는 힘의 효과를 나타내는 물리량입니다.",
    "공구 마모 시간(분)": "공구를 사용한 누적 시간입니다. 실제 마모량을 직접 측정한 값과는 다릅니다.",
    "기계 고장 여부": "기계 고장 발생 여부입니다. 일반적으로 0은 고장 없음, 1은 고장 발생을 뜻합니다.",
    "공구 마모 고장": "공구 마모와 관련된 고장 유형입니다. 1이면 해당 유형의 고장이 기록된 것입니다.",
    "열 방출 고장": "열 방출 문제와 관련된 고장 유형입니다. 1이면 해당 유형의 고장이 기록된 것입니다.",
    "동력 고장": "동력 관련 문제로 분류된 고장 유형입니다. 1이면 해당 유형의 고장이 기록된 것입니다.",
    "과부하 고장": "과도한 부하와 관련된 고장 유형입니다. 1이면 해당 유형의 고장이 기록된 것입니다.",
    "무작위 고장": "무작위 고장 유형으로 기록된 경우입니다. 1이면 해당 유형의 고장이 기록된 것입니다."
}


# ==========================================
# 3. 데이터 불러오기
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
        "GitHub 저장소가 공개 상태인지, "
        "CSV 파일 주소가 올바른지 확인해 주세요."
    )
    st.stop()

if df.empty:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 변수명 정리 및 숫자형 변환
# ==========================================
def normalize_name(value):
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", value).lower()


original_columns = list(df.columns)

# 실제 열 이름을 기준으로 고장 여부 열을 먼저 찾습니다.
target_candidates = {
    "machinefailure",
    "failure",
    "고장여부",
    "기계고장여부",
    "기계고장",
    "고장",
    "target"
}

target_original = None

for col in original_columns:
    if normalize_name(col) in target_candidates:
        target_original = col
        break

# 수치형 변수는 변환할 수 있는 경우에만 숫자로 변환합니다.
for col in original_columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    original_count = df[col].notna().sum()

    if original_count > 0:
        ratio = converted.notna().sum() / original_count

        if ratio >= 0.8:
            df[col] = converted

# 열 이름을 한국어로 바꿉니다.
rename_map = {}

for col in df.columns:
    for english_name, korean_name in COLUMN_NAMES.items():
        if normalize_name(col) == normalize_name(english_name):
            rename_map[col] = korean_name
            break

df = df.rename(columns=rename_map)

target_col = None
if target_original is not None:
    target_col = rename_map.get(target_original, target_original)

# 한국어 열 이름에 맞춰 변수 목록을 만듭니다.
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# 식별 번호는 통계 분석용 측정 변수에서 제외합니다.
id_cols = ["데이터 식별 번호", "제품 식별 번호"]
analysis_numeric_cols = [
    col for col in numeric_cols
    if col not in id_cols
]


# ==========================================
# 5. 데이터 한눈에 보기
# ==========================================
st.subheader("📌 데이터 한눈에 보기")

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 데이터", f"{len(df):,}행")
c2.metric("전체 변수", f"{len(df.columns):,}개")
c3.metric("수치형 변수", f"{len(numeric_cols):,}개")
c4.metric("결측값", f"{int(df.isna().sum().sum()):,}개")

with st.expander("📖 데이터 미리 보기"):
    st.write("아래 표에서 한국어 변수명과 실제 데이터를 확인할 수 있습니다.")
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📚 변수 뜻 알아보기"):
    for col in df.columns:
        meaning = VARIABLE_MEANINGS.get(
            col, "데이터에 기록된 변수입니다."
        )
        st.markdown(f"**{col}**")
        st.write(meaning)


# ==========================================
# 6. 기초통계량
# ==========================================
st.divider()
st.subheader("1. 📊 수치형 변수의 기초통계")

if analysis_numeric_cols:
    stats = df[analysis_numeric_cols].describe().T

    stats = stats.rename(columns={
        "count": "자료 수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값"
    })

    st.dataframe(stats.round(2), use_container_width=True)

    stat_col = st.selectbox(
        "분포를 확인할 변수",
        analysis_numeric_cols,
        key="stat_col"
    )

    st.caption(VARIABLE_MEANINGS.get(stat_col, ""))

    if df[stat_col].notna().any():
        fig = px.histogram(
            df,
            x=stat_col,
            nbins=30,
            title=f"{stat_col} 분포",
            labels={stat_col: stat_col, "count": "자료 수"},
            color_discrete_sequence=["#3984E8"]
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title=stat_col,
            yaxis_title="자료 수"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("선택한 변수에 분석할 값이 없습니다.")
else:
    st.warning("분석할 수 있는 수치형 변수가 없습니다.")


# ==========================================
# 7. 제품 유형 및 범주형 변수
# ==========================================
st.divider()
st.subheader("2. 🧩 범주형 변수의 분포")

categorical_cols = []

for col in df.columns:
    if col in numeric_cols:
        continue

    if 1 <= df[col].nunique(dropna=True) <= 30:
        categorical_cols.append(col)

if categorical_cols:
    cat_col = st.selectbox(
        "분포를 확인할 범주형 변수",
        categorical_cols,
        key="cat_col"
    )

    counts = (
        df[cat_col].fillna("결측값").astype(str)
        .value_counts()
        .rename_axis("항목")
        .reset_index(name="개수")
    )

    fig = px.bar(
        counts,
        x="항목",
        y="개수",
        color="항목",
        text="개수",
        title=f"{cat_col}별 데이터 개수",
        labels={"항목": cat_col, "개수": "데이터 개수"}
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("분석할 범주형 변수가 없습니다.")


# ==========================================
# 8. 기계 고장 여부 분석
# ==========================================
st.divider()
st.subheader("3. 🔧 기계 고장 여부 분석")

if target_col is not None and target_col in df.columns:
    st.caption(VARIABLE_MEANINGS.get(target_col, ""))

    target_data = df[target_col].astype("string").fillna("결측값")

    target_counts = (
        target_data.value_counts()
        .rename_axis("고장 여부")
        .reset_index(name="개수")
    )

    # 0과 1이 숫자 형태라면 의미를 표시합니다.
    target_counts["고장 여부"] = target_counts["고장 여부"].replace({
        "0": "고장 없음 (0)",
        "1": "고장 발생 (1)"
    })

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            target_counts,
            x="고장 여부",
            y="개수",
            color="고장 여부",
            text="개수",
            title="고장 여부별 데이터 수"
        )
        fig.update_layout(
            template="plotly_white",
            height=380,
            showlegend=False,
            xaxis_title="기계 고장 여부",
            yaxis_title="데이터 수"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(
            target_counts,
            names="고장 여부",
            values="개수",
            title="고장 여부별 비율",
            hole=0.4
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    compare_cols = [
        col for col in analysis_numeric_cols
        if col != target_col
    ]

    if compare_cols:
        st.markdown("#### 고장 여부에 따른 평균 비교")

        compare_col = st.selectbox(
            "평균을 비교할 변수",
            compare_cols,
            key="compare_col"
        )

        st.caption(VARIABLE_MEANINGS.get(compare_col, ""))

        compare_df = pd.DataFrame({
            "고장 여부": target_data,
            "측정값": df[compare_col]
        }).dropna(subset=["측정값"])

        mean_df = (
            compare_df.groupby("고장 여부", as_index=False)["측정값"]
            .mean()
            .rename(columns={"측정값": "평균"})
        )

        mean_df["고장 여부"] = mean_df["고장 여부"].replace({
            "0": "고장 없음 (0)",
            "1": "고장 발생 (1)"
        })

        fig = px.bar(
            mean_df,
            x="고장 여부",
            y="평균",
            color="고장 여부",
            text="평균",
            title=f"고장 여부에 따른 {compare_col} 평균",
            labels={"고장 여부": "기계 고장 여부", "평균": f"{compare_col} 평균"}
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("고장 여부 변수를 자동으로 찾지 못했습니다.")
    st.write("실제 데이터 열 이름:", list(df.columns))


# ==========================================
# 9. 상관관계 분석
# ==========================================
st.divider()
st.subheader("4. 🔗 변수 사이의 상관관계")

st.write(
    "상관계수는 두 수치형 변수가 함께 변하는 정도를 나타냅니다. "
    "1에 가까우면 양의 관계, -1에 가까우면 음의 관계, "
    "0에 가까우면 선형 관계가 약합니다."
)

st.caption(
    "켈빈(K)은 절대온도, rpm은 분당 회전 수, "
    "N·m는 토크의 단위, 분(min)은 공구 사용 시간입니다."
)

corr_cols = [
    col for col in analysis_numeric_cols
    if df[col].nunique(dropna=True) > 1
]

if len(corr_cols) >= 2:
    corr = df[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="변수별 상관계수 히트맵",
        labels={
            "x": "변수",
            "y": "변수",
            "color": "상관계수"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=max(450, len(corr_cols) * 55),
        xaxis_title="변수",
        yaxis_title="변수"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 고장 여부와 각 변수의 상관계수
    if target_col in corr.columns if target_col is not None else False:
        st.markdown("#### 고장 여부와 각 변수의 상관계수")

        target_corr = (
            corr[target_col]
            .drop(labels=[target_col], errors="ignore")
            .sort_values(key=abs, ascending=False)
            .rename("상관계수")
            .reset_index()
        )

        target_corr.columns = ["변수", "상관계수"]
        target_corr["상관계수"] = target_corr["상관계수"].round(3)

        st.dataframe(target_corr, use_container_width=True)

        st.caption(
            "상관계수가 높다고 해서 해당 변수가 고장의 직접적인 원인이라는 "
            "뜻은 아닙니다. 고장 유형 변수와 전체 고장 여부 사이의 관계도 "
            "데이터 구조에 따라 높게 나타날 수 있습니다."
        )
else:
    st.info("상관관계를 계산하려면 값이 서로 다른 수치형 변수가 2개 이상 필요합니다.")


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

    fig = px.bar(
        missing_df,
        x="변수",
        y="결측값 수",
        text="결측값 수",
        title="변수별 결측값 수",
        color="결측값 수",
        color_continuous_scale="Blues",
        labels={"변수": "변수명", "결측값 수": "결측값 수"}
    )
    fig.update_layout(
        template="plotly_white",
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 11. 분석 질문
# ==========================================
st.divider()
st.subheader("📝 분석해 볼 질문")

st.markdown(
    """
    1. 정상 기계와 고장 난 기계의 평균 온도는 다를까?
    2. 공구 마모 시간이 길수록 고장 비율이 높을까?
    3. 회전 속도와 토크는 어떤 관계가 있을까?
    4. 고장 여부와 상관관계가 상대적으로 높은 변수는 무엇일까?
    5. 제품 유형에 따라 고장 건수가 다를까?
    """
)

st.caption(
    "이 앱은 기초통계 및 탐색적 데이터 분석을 위한 도구입니다. "
    "통계적 관계만으로 고장의 직접적인 원인을 확정할 수는 없습니다."
)
