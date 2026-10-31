
import streamlit as st
import pandas as pd

# --------------------------------
# 1. 기본 설정
# --------------------------------
st.set_page_config(
    page_title="기계 고장 데이터 전처리",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 기계 고장 데이터 전처리")
st.write("기계 고장 데이터를 불러와 분석하기 좋은 형태로 정리합니다.")

DATA_URL = "https://raw.githubusercontent.com/jisujisujisu333/Machine-Failure-Prediction-Project/refs/heads/main/data/%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"


# --------------------------------
# 2. 데이터 불러오기
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    original_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------
# 3. 열 이름 한국어로 변경
# --------------------------------
rename_dict = {
    "UDI": "데이터 번호",
    "Product ID": "제품 번호",
    "Type": "제품 유형",
    "Air temperature [K]": "공기 온도(K)",
    "Process temperature [K]": "공정 온도(K)",
    "Rotational speed [rpm]": "회전 속도(rpm)",
    "Torque [Nm]": "토크(Nm)",
    "Tool wear [min]": "공구 마모 시간(분)",
    "Machine failure": "기계 고장 여부",
    "TWF": "공구 마모 고장",
    "HDF": "열 방출 고장",
    "PWF": "동력 고장",
    "OSF": "과부하 고장",
    "RNF": "무작위 고장"
}

df = original_df.rename(columns=rename_dict).copy()


# --------------------------------
# 4. 전처리
# --------------------------------

# 중복 데이터 제거
before_rows = len(df)
df = df.drop_duplicates().copy()
duplicate_count = before_rows - len(df)

# 분석에 불필요한 식별 번호 제거
id_columns = ["데이터 번호", "제품 번호"]
df = df.drop(columns=id_columns, errors="ignore")

# 숫자형 데이터와 문자형 데이터 구분
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

# 기계 고장 여부가 없는 행은 분석 대상에서 제외
if "기계 고장 여부" in df.columns:
    df = df.dropna(subset=["기계 고장 여부"]).copy()

# 숫자형 결측값은 각 열의 중앙값으로 처리
for col in numeric_columns:
    if col in df.columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

# 문자형 결측값은 가장 많이 등장한 값으로 처리
for col in categorical_columns:
    if col in df.columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])


# --------------------------------
# 5. 전처리 결과 요약
# --------------------------------
st.subheader("📊 전처리 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전처리 전 데이터", f"{len(original_df):,}개")
col2.metric("전처리 후 데이터", f"{len(df):,}개")
col3.metric("제거한 중복 데이터", f"{duplicate_count:,}개")
col4.metric("최종 변수 개수", f"{len(df.columns)}개")

st.divider()


# --------------------------------
# 6. 전처리된 데이터 확인
# --------------------------------
st.subheader("📋 전처리된 데이터")

st.write("식별 번호를 제거하고 열 이름을 한국어로 바꾼 데이터입니다.")

st.dataframe(df.head(20), use_container_width=True)

st.caption(f"전체 {len(df):,}개 데이터 중 처음 20개를 표시합니다.")


# --------------------------------
# 7. 결측값 확인
# --------------------------------
st.subheader("🔎 결측값 확인")

missing_df = pd.DataFrame({
    "변수 이름": df.columns,
    "결측값 개수": df.isnull().sum().values,
    "데이터 유형": df.dtypes.astype(str).values
})

st.dataframe(missing_df, use_container_width=True)

if df.isnull().sum().sum() == 0:
    st.success("결측값이 없습니다.")
else:
    st.warning("아직 결측값이 남아 있습니다.")


# --------------------------------
# 8. 기초 통계
# --------------------------------
st.subheader("📈 주요 변수의 기초 통계")

st.write("숫자로 이루어진 변수들의 평균, 표준편차, 최솟값, 최댓값 등을 확인합니다.")

st.dataframe(
    df.describe().round(2),
    use_container_width=True
)


# --------------------------------
# 9. 고장 여부별 데이터 확인
# --------------------------------
st.subheader("⚙️ 기계 고장 여부 확인")

if "기계 고장 여부" in df.columns:
    failure_count = df["기계 고장 여부"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    result_df = pd.DataFrame({
        "고장 여부": ["정상", "고장"],
        "데이터 개수": [
            int(failure_count.loc[0]),
            int(failure_count.loc[1])
        ]
    })

    st.dataframe(result_df, use_container_width=True)

    st.bar_chart(result_df.set_index("고장 여부"))


# --------------------------------
# 10. CSV 다운로드
# --------------------------------
st.divider()
st.subheader("💾 전처리 데이터 다운로드")

# 고장 유형 분석용: 고장 유형 변수 유지
analysis_csv = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 고장 유형 분석용 CSV 다운로드",
    data=analysis_csv,
    file_name="기계고장_분석용_전처리.csv",
    mime="text/csv"
)

# 예측 모델용: 고장 결과에서 파생된 유형 변수 제거
failure_type_columns = [
    "공구 마모 고장",
    "열 방출 고장",
    "동력 고장",
    "과부하 고장",
    "무작위 고장"
]

prediction_df = df.drop(
    columns=failure_type_columns,
    errors="ignore"
)

prediction_csv = prediction_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 고장 예측 모델용 CSV 다운로드",
    data=prediction_csv,
    file_name="기계고장_예측용_전처리.csv",
    mime="text/csv"
)

st.info(
    "고장 유형 분석용 파일에는 고장 유형 변수가 포함됩니다. "
    "예측 모델용 파일에서는 예측 결과를 미리 알려줄 수 있는 고장 유형 변수를 제외했습니다."
)
