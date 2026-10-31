
import streamlit as st
import pandas as pd

# =====================================
# 1. 기본 설정
# =====================================

st.set_page_config(
    page_title="기계 고장 데이터 분석",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 산업용 기계 고장 데이터 분석")
st.write(
    "산업용 제조 기계의 작동 데이터를 분석하여 "
    "기계 고장과 관련된 요인을 알아봅니다."
)

# =====================================
# 2. 데이터 불러오기
# =====================================

DATA_URL = "https://raw.githubusercontent.com/jisujisujisu333/Machine-Failure-Prediction-Project/refs/heads/main/data/%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    original_df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(e)
    st.stop()


# =====================================
# 3. 열 이름 한국어로 변경
# =====================================

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


# =====================================
# 4. 데이터 전처리
# =====================================

# 중복 데이터 제거
before_count = len(df)
df = df.drop_duplicates().copy()
duplicate_count = before_count - len(df)

# 식별 번호 제거
df = df.drop(
    columns=["데이터 번호", "제품 번호"],
    errors="ignore"
)

# 숫자형 변수 목록
numeric_columns_all = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(Nm)",
    "공구 마모 시간(분)",
    "기계 고장 여부",
    "공구 마모 고장",
    "열 방출 고장",
    "동력 고장",
    "과부하 고장",
    "무작위 고장"
]

# 숫자형 데이터 변환
for col in numeric_columns_all:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# 기계 고장 여부가 없는 행은 분석에서 제외
if "기계 고장 여부" in df.columns:
    df = df.dropna(
        subset=["기계 고장 여부"]
    ).copy()

# 나머지 숫자형 결측값은 중앙값으로 처리
for col in numeric_columns_all:
    if col in df.columns:
        if df[col].isnull().any():
            median_value = df[col].median()

            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

# 제품 유형 결측값 처리
if "제품 유형" in df.columns:
    if df["제품 유형"].isnull().any():
        mode_value = df["제품 유형"].mode()

        if not mode_value.empty:
            df["제품 유형"] = df["제품 유형"].fillna(
                mode_value.iloc[0]
            )


# =====================================
# 5. 전체 데이터 요약
# =====================================

st.divider()
st.header("📊 1. 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 데이터",
    f"{len(original_df):,}개"
)

col2.metric(
    "전처리 후 데이터",
    f"{len(df):,}개"
)

col3.metric(
    "제거한 중복 데이터",
    f"{duplicate_count:,}개"
)

col4.metric(
    "분석 변수 개수",
    f"{len(df.columns)}개"
)


# =====================================
# 6. 데이터 미리 보기
# =====================================

st.subheader("📋 전처리된 데이터 확인")

st.dataframe(
    df.head(20),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"전체 {len(df):,}개 데이터 중 처음 20개를 표시합니다."
)


# =====================================
# 7. 결측값 확인
# =====================================

st.divider()
st.header("🔎 2. 결측값 확인")

missing_df = pd.DataFrame({
    "변수 이름": df.columns,
    "결측값 개수": df.isnull().sum().values
})

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)

if df.isnull().sum().sum() == 0:
    st.success("결측값이 없습니다.")
else:
    st.warning("아직 결측값이 남아 있습니다.")


# =====================================
# 8. 기초 통계
# =====================================

st.divider()
st.header("📈 3. 기초 통계")

st.write(
    "각 숫자형 변수의 평균, 표준편차, 최솟값, "
    "최댓값 등을 확인할 수 있습니다."
)

st.dataframe(
    df.describe().round(2),
    use_container_width=True
)


# =====================================
# 9. 제품 유형별 데이터 분포
# =====================================

st.divider()
st.header("🧩 4. 범주형 변수의 분포")

if "제품 유형" in df.columns:

    type_count = (
        df["제품 유형"]
        .value_counts()
        .reindex(["L", "M", "H"], fill_value=0)
    )

    type_result = pd.DataFrame({
        "제품 유형": type_count.index,
        "데이터 개수": type_count.values
    })

    st.subheader("제품 유형별 데이터 개수")

    st.bar_chart(
        type_result.set_index("제품 유형")
    )

    st.dataframe(
        type_result,
        use_container_width=True,
        hide_index=True
    )


# =====================================
# 10. 기계 고장 여부 확인
# =====================================

st.divider()
st.header("⚙️ 5. 기계 고장 여부")

if "기계 고장 여부" in df.columns:

    failure_count = (
        df["기계 고장 여부"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    failure_result = pd.DataFrame({
        "구분": ["정상", "고장"],
        "데이터 개수": [
            int(failure_count.loc[0]),
            int(failure_count.loc[1])
        ]
    })

    total_count = int(failure_count.sum())

    if total_count > 0:

        failure_rate = (
            failure_count.loc[1] / total_count * 100
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "정상 기계",
            f"{failure_count.loc[0]:,}개"
        )

        col2.metric(
            "기계 고장률",
            f"{failure_rate:.2f}%"
        )

    st.bar_chart(
        failure_result.set_index("구분")
    )


# =====================================
# 11. 원하는 변수 2개로 기계 고장률 분석
# =====================================

st.divider()
st.header("🔍 6. 원하는 변수로 기계 고장률 분석")

st.write(
    "제품 유형과 숫자형 변수 2개를 선택하여 "
    "각 조건에 따른 기계 고장률을 퍼센트(%)로 비교합니다."
)

# 분석에 사용할 숫자형 변수
analysis_variables = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(Nm)",
    "공구 마모 시간(분)"
]

analysis_variables = [
    col for col in analysis_variables
    if col in df.columns
]

if (
    "제품 유형" not in df.columns
    or "기계 고장 여부" not in df.columns
    or len(analysis_variables) < 2
):

    st.warning(
        "분석에 필요한 변수가 부족합니다."
    )

else:

    # ---------------------------------
    # 11-1. 변수 선택
    # ---------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_type = st.selectbox(
            "① 제품 유형 선택",
            ["전체", "L", "M", "H"]
        )

    with col2:

        variable1 = st.selectbox(
            "② 첫 번째 분석 변수",
            analysis_variables,
            index=3
        )

    with col3:

        variable2_options = [
            col for col in analysis_variables
            if col != variable1
        ]

        variable2 = st.selectbox(
            "③ 두 번째 분석 변수",
            variable2_options
        )

    # ---------------------------------
    # 11-2. 제품 유형별 데이터 분리
    # ---------------------------------

    analysis_df = df.copy()

    if selected_type != "전체":

        analysis_df = analysis_df[
            analysis_df["제품 유형"] == selected_type
        ].copy()

    st.subheader("📊 분석 대상 데이터")

    st.metric(
        "선택한 제품 유형의 데이터 개수",
        f"{len(analysis_df):,}개"
    )

    # ---------------------------------
    # 11-3. 고장률 계산 함수
    # ---------------------------------

    def calculate_failure_rate(data, variable):

        temp = data[
            [variable, "기계 고장 여부"]
        ].copy()

        # 결측값 제거
        temp = temp.dropna()

        # 숫자형 변환
        temp[variable] = pd.to_numeric(
            temp[variable],
            errors="coerce"
        )

        temp["기계 고장 여부"] = pd.to_numeric(
            temp["기계 고장 여부"],
            errors="coerce"
        )

        temp = temp.dropna()

        if len(temp) < 2:
            return None

        # 중앙값 계산
        median_value = temp[variable].median()

        # 중앙값 기준으로 두 그룹 구분
        temp["구분"] = "낮은 그룹"

        temp.loc[
            temp[variable] > median_value,
            "구분"
        ] = "높은 그룹"

        # 그룹별 데이터 개수와 고장 개수
        result = temp.groupby(
            "구분",
            observed=False
        ).agg(
            전체_개수=("기계 고장 여부", "count"),
            고장_개수=("기계 고장 여부", "sum")
        )

        # 고장률 계산
        result["고장률(%)"] = (
            result["고장_개수"]
            / result["전체_개수"]
            * 100
        )

        result["중앙값"] = median_value

        return result.reset_index()

    # ---------------------------------
    # 11-4. 두 변수의 고장률 계산
    # ---------------------------------

    result1 = calculate_failure_rate(
        analysis_df,
        variable1
    )

    result2 = calculate_failure_rate(
        analysis_df,
        variable2
    )

    # ---------------------------------
    # 11-5. 결과 표시
    # ---------------------------------

    st.divider()
    st.subheader("📈 변수별 기계 고장률 비교")

    results = [
        (variable1, result1),
        (variable2, result2)
    ]

    for variable, result in results:

        st.markdown("### " + variable)

        if result is None or result.empty:

            st.warning(
                "분석할 데이터가 부족합니다."
            )

            continue

        # 중앙값 안내
        median_value = result["중앙값"].iloc[0]

        st.info(
            "분석 기준 중앙값: "
            + format(median_value, ",.2f")
        )

        # 그래프
        chart_data = result.set_index("구분")[
            ["고장률(%)"]
        ]

        st.bar_chart(
            chart_data,
            color="#2997E5"
        )

        # 결과 표
        display_result = result[
            [
                "구분",
                "전체_개수",
                "고장_개수",
                "고장률(%)"
            ]
        ].copy()

        display_result.columns = [
            "변수 구분",
            "전체 기계 수",
            "고장 기계 수",
            "고장률(%)"
        ]

        display_result["고장률(%)"] = (
            display_result["고장률(%)"].round(2)
        )

        st.dataframe(
            display_result,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------
        # 11-6. 분석 결과 설명
        # ---------------------------------

        if len(result) == 2:

            highest_index = result[
                "고장률(%)"
            ].idxmax()

            highest_group = result.loc[
                highest_index
            ]

            group_name = highest_group["구분"]
            rate = highest_group["고장률(%)"]

            st.write(
                "**분석 결과:** "
                + str(group_name)
                + "의 고장률이 "
                + str(round(rate, 2))
                + "%로 더 높게 나타났습니다."
            )

            # 두 그룹의 고장률 차이
            rate_difference = abs(
                result["고장률(%)"].iloc[0]
                - result["고장률(%)"].iloc[1]
            )

            st.caption(
                "두 그룹의 고장률 차이: "
                + str(round(rate_difference, 2))
                + "%p"
            )

        else:

            st.warning(
                "두 그룹으로 나눌 수 없어 "
                "고장률 비교가 어렵습니다."
            )

        st.divider()

    # ---------------------------------
    # 11-7. 분석 안내
    # ---------------------------------

    st.caption(
        "※ 낮은 그룹과 높은 그룹은 선택한 변수의 중앙값을 기준으로 구분합니다. "
        "중앙값과 같은 값은 낮은 그룹에 포함됩니다. "
        "고장률은 각 그룹의 고장 기계 수를 전체 기계 수로 나누어 계산합니다. "
        "분석 결과는 관련성을 보여주며 직접적인 원인을 증명하지는 않습니다."
    )


# =====================================
# 12. 데이터 다운로드
# =====================================

st.divider()
st.header("💾 7. 전처리 데이터 다운로드")

csv_data = df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="📥 전처리된 CSV 다운로드",
    data=csv_data,
    file_name="기계고장_전처리데이터.csv",
    mime="text/csv"
)

st.success(
    "기계 고장 데이터 분석이 완료되었습니다."
)
