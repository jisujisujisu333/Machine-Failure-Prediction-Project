
# --------------------------------
# 원하는 변수 3개 선택하여 관계 분석
# --------------------------------
st.divider()
st.header("🔍 원하는 변수끼리 관계 분석")

st.write(
    "제품 유형과 숫자형 변수 2개를 선택하여 "
    "각 변수 사이의 관계를 분석할 수 있습니다."
)

# 숫자형 변수 목록
numeric_columns = df.select_dtypes(include="number").columns.tolist()

# 제품 유형을 제외한 숫자형 변수
numeric_columns = [
    col for col in numeric_columns
    if col != "제품 유형"
]

if len(numeric_columns) < 2:
    st.warning("분석할 숫자형 변수가 2개 이상 필요합니다.")

elif "제품 유형" not in df.columns:
    st.warning("제품 유형 변수가 데이터에 없습니다.")

else:
    # 변수 선택 화면
    col1, col2, col3 = st.columns(3)

    with col1:
        type_variable = st.selectbox(
            "첫 번째 변수",
            ["제품 유형"]
        )

    with col2:
        variable1 = st.selectbox(
            "두 번째 변수",
            numeric_columns,
            index=0
        )

    with col3:
        variable2_options = [
            col for col in numeric_columns
            if col != variable1
        ]

        variable2 = st.selectbox(
            "세 번째 변수",
            variable2_options
        )

    # 상관비 계산 함수
    def correlation_ratio(categories, values):
        data = pd.DataFrame({
            "category": categories,
            "value": values
        }).dropna()

        if len(data) == 0:
            return None

        overall_mean = data["value"].mean()

        total_sum = (
            (data["value"] - overall_mean) ** 2
        ).sum()

        if total_sum == 0:
            return None

        between_sum = 0

        for category in data["category"].unique():
            group = data[
                data["category"] == category
            ]["value"]

            between_sum += len(group) * (
                group.mean() - overall_mean
            ) ** 2

        eta = (between_sum / total_sum) ** 0.5

        return eta

    # 선택한 세 변수의 데이터
    selected_df = df[
        [type_variable, variable1, variable2]
    ].copy()

    # 제품 유형과 숫자형 변수의 상관비
    eta1 = correlation_ratio(
        selected_df[type_variable],
        selected_df[variable1]
    )

    eta2 = correlation_ratio(
        selected_df[type_variable],
        selected_df[variable2]
    )

    # 두 숫자형 변수의 피어슨 상관계수
    pearson_df = selected_df[
        [variable1, variable2]
    ].dropna()

    if (
        pearson_df[variable1].nunique() > 1
        and pearson_df[variable2].nunique() > 1
    ):
        pearson = pearson_df[variable1].corr(
            pearson_df[variable2]
        )
    else:
        pearson = None

    # 결과 표시
    st.subheader("📊 선택한 변수의 관계 분석 결과")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            f"제품 유형 ↔ {variable1}",
            "계산 불가" if eta1 is None else f"{eta1:.3f}"
        )
        st.caption("상관비(η)")

    with result2:
        st.metric(
            f"제품 유형 ↔ {variable2}",
            "계산 불가" if eta2 is None else f"{eta2:.3f}"
        )
        st.caption("상관비(η)")

    with result3:
        st.metric(
            f"{variable1} ↔ {variable2}",
            "계산 불가" if pearson is None else f"{pearson:.3f}"
        )
        st.caption("피어슨 상관계수")

    st.info(
        "상관비(η)는 0~1 사이의 값으로, "
        "제품 유형에 따라 숫자형 변수의 값이 얼마나 달라지는지 나타냅니다. "
        "피어슨 상관계수는 -1~1 사이의 값으로 두 숫자형 변수의 "
        "선형 관계와 방향을 나타냅니다."
    )

    # 제품 유형별 선택 변수 평균 비교
    st.subheader("📈 제품 유형별 변수 평균")

    type_mean = selected_df.groupby(
        type_variable
    )[[variable1, variable2]].mean().reindex(
        ["L", "M", "H"]
    )

    st.dataframe(
        type_mean.round(3),
        use_container_width=True
    )

    st.bar_chart(type_mean)
