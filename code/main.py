
# --------------------------------
# 원하는 변수 2개로 기계 고장률 분석
# --------------------------------
st.divider()
st.header("🔍 원하는 변수로 기계 고장률 분석")

st.write(
    "제품 유형과 숫자형 변수 2개를 선택하면 "
    "각 조건에 따른 기계 고장률을 퍼센트(%)로 비교합니다."
)

# 분석에 사용할 숫자형 변수
numeric_columns = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(Nm)",
    "공구 마모 시간(분)"
]

# 실제 데이터에 존재하는 변수만 사용
numeric_columns = [
    col for col in numeric_columns
    if col in df.columns
]

if (
    "제품 유형" not in df.columns
    or "기계 고장 여부" not in df.columns
    or len(numeric_columns) < 2
):
    st.warning("분석에 필요한 변수가 데이터에 없습니다.")

else:
    # --------------------------------
    # 1. 변수 선택
    # --------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_type = st.selectbox(
            "① 제품 유형 선택",
            ["전체", "L", "M", "H"]
        )

    with col2:
        variable1 = st.selectbox(
            "② 첫 번째 분석 변수",
            numeric_columns,
            index=3
        )

    with col3:
        variable2_options = [
            col for col in numeric_columns
            if col != variable1
        ]

        variable2 = st.selectbox(
            "③ 두 번째 분석 변수",
            variable2_options
        )

    # --------------------------------
    # 2. 선택한 제품 유형으로 데이터 분리
    # --------------------------------
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

    # --------------------------------
    # 3. 고장률 계산 함수
    # --------------------------------
    def calculate_failure_rate(data, variable):

        temp = data[
            [variable, "기계 고장 여부"]
        ].dropna().copy()

        if len(temp) < 2:
            return None

        # 선택한 변수의 중앙값 계산
        median_value = temp[variable].median()

        # 중앙값을 기준으로 낮은 그룹과 높은 그룹 분리
        temp["구분"] = temp[variable].apply(
            lambda x: "낮은 그룹" if x <= median_value
            else "높은 그룹"
        )

        # 각 그룹의 고장률 계산
        result = temp.groupby("구분").agg(
            전체_개수=("기계 고장 여부", "count"),
            고장_개수=("기계 고장 여부", "sum")
        )

        result["고장률(%)"] = (
            result["고장_개수"] / result["전체_개수"] * 100
        )

        result["변수"] = variable
        result["중앙값"] = median_value

        return result.reset_index()

    # --------------------------------
    # 4. 두 변수의 고장률 계산
    # --------------------------------
    result1 = calculate_failure_rate(
        analysis_df, variable1
    )

    result2 = calculate_failure_rate(
        analysis_df, variable2
    )

    # --------------------------------
    # 5. 결과 표시
    # --------------------------------
    st.divider()
    st.subheader("📈 변수별 기계 고장률 비교")

    for variable, result in [
        (variable1, result1),
        (variable2, result2)
    ]:

        st.markdown(f"### {variable}")

        if result is None or result.empty:
            st.warning("분석할 데이터가 부족합니다.")
            continue

        # 중앙값 안내
        median_value = result["중앙값"].iloc[0]

        st.info(
            f"분석 기준 중앙값: {median_value:,.2f}"
        )

        # 결과 그래프
        chart_data = result.set_index("구분")[
            ["고장률(%)"]
        ]

        st.bar_chart(
            chart_data,
            y="고장률(%)",
            color="#2997E5"
        )

        # 고장률 결과 표
        display_result = result[
            ["구분", "전체_개수", "고장_개수", "고장률(%)"]
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

        # 가장 높은 고장률 설명
        if len(result) == 2:
            high_rate = result.loc[
                result["고장률(%)"].idxmax()
            ]

            st.write(
                f"**분석 결과:** {high_rate['구분']}의 "
                f"고장률이 {high_rate['고장률(%)']:.2f}%로 "
                f더 높게 나타났습니다."
            )

        st.divider()

    st.caption(
        "※ 낮은 그룹과 높은 그룹은 선택한 제품 유형 내 변수의 중앙값을 기준으로 구분합니다. "
        "고장률은 각 그룹의 고장 기계 수를 전체 기계 수로 나누어 계산합니다. "
        "이 결과는 관련성을 보여주며 직접적인 원인을 증명하지는 않습니다."
    )
