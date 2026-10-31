
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="기계 고장 기초통계 분석실",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 기계 고장 기초통계 분석실")
st.write(
    "기계의 작동 데이터를 분석하여 기계 고장과 관련된 "
    "요인을 알아봅니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jisujisujisu333/Machine-Failure-Prediction-Project/"
    "refs/heads/main/data/"
    "%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1"
    "%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(DATA_URL, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(DATA_URL)


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(e)
    st.stop()


# ==========================================
# 3. 변수명 한국어로 변경
# ==========================================
rename_dict = {
    "UDI": "데이터 번호",
    "Product ID": "제품 번호",
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

df = df.rename(columns=rename_dict)


# ==========================================
# 4. 변수 설명 및 숫자형 변환
# ==========================================
variable_meanings = {
    "제품 유형": "제품의 종류입니다. L, M, H 세 가지로 구분됩니다.",
    "공기 온도(K)": "주변 공기의 온도입니다.",
    "공정 온도(K)": "기계의 작업 과정에서 측정한 온도입니다.",
    "회전 속도(rpm)": "기계가 1분 동안 회전하는 횟수입니다.",
    "토크(N·m)": "회전 운동을 일으키는 힘의 효과를 나타냅니다.",
    "공구 마모 시간(분)": "공구를 사용한 누적 시간입니다.",
    "기계 고장 여부": "0은 고장 없음, 1은 고장 발생을 의미합니다.",
    "공구 마모 고장": "공구 마모와 관련된 고장 유형입니다.",
    "열 방출 고장": "열 방출 문제와 관련된 고장 유형입니다.",
    "동력 고장": "동력 문제와 관련된 고장 유형입니다.",
    "과부하 고장": "과도한 부하와 관련된 고장 유형입니다.",
    "무작위 고장": "무작위 고장 유형으로 기록된 경우입니다."
}

numeric_columns = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
    "공구 마모 시간(분)",
    "기계 고장 여부",
    "공구 마모 고장",
    "열 방출 고장",
    "동력 고장",
    "과부하 고장",
    "무작위 고장"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ==========================================
# 5. 데이터 한눈에 보기
# ==========================================
st.divider()
st.subheader("📌 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 데이터", f"{len(df):,}개")
col2.metric("전체 변수", f"{len(df.columns)}개")
col3.metric(
    "수치형 변수",
    f"{len(df.select_dtypes(include='number').columns)}개"
)
col4.metric("결측값", f"{df.isnull().sum().sum():,}개")

with st.expander("📋 데이터 미리 보기"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📖 변수의 의미 확인하기"):
    for col in df.columns:
        if col in variable_meanings:
            st.markdown(f"**{col}**")
            st.write(variable_meanings[col])


# ==========================================
# 6. 수치형 변수의 기초통계
# ==========================================
st.divider()
st.subheader("1. 📊 수치형 변수의 기초통계")

available_numeric = [
    col for col in numeric_columns
    if col in df.columns
]

if available_numeric:
    stats = df[available_numeric].describe().T

    stats = stats.rename(columns={
        "count": "데이터 개수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값"
    })

    st.dataframe(stats.round(2), use_container_width=True)

    selected_stat = st.selectbox(
        "분포를 확인할 변수",
        available_numeric,
        key="selected_stat"
    )

    fig = px.histogram(
        df,
        x=selected_stat,
        nbins=30,
        title=f"{selected_stat} 분포",
        color_discrete_sequence=["#3984E8"]
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 7. 범주형 변수의 분포
# ==========================================
st.divider()
st.subheader("2. 🧩 범주형 변수의 분포")

if "제품 유형" in df.columns:

    type_count = (
        df["제품 유형"]
        .value_counts()
        .reindex(["L", "M", "H"], fill_value=0)
        .rename_axis("제품 유형")
        .reset_index(name="데이터 개수")
    )

    fig = px.bar(
        type_count,
        x="제품 유형",
        y="데이터 개수",
        color="제품 유형",
        text="데이터 개수",
        title="제품 유형별 데이터 개수",
        color_discrete_sequence=[
            "#1675D1",
            "#81C4F7",
            "#FF4545"
        ]
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        type_count,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 8. 기계 고장 여부 분석
# ==========================================
st.divider()
st.subheader("3. 🔧 기계 고장 여부 분석")

if "기계 고장 여부" in df.columns:

    failure_count = (
        df["기계 고장 여부"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    total_count = failure_count.sum()

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
            "전체 기계 고장률",
            f"{failure_rate:.2f}%"
        )

        failure_result = pd.DataFrame({
            "구분": ["정상", "고장"],
            "데이터 개수": [
                failure_count.loc[0],
                failure_count.loc[1]
            ]
        })

        fig = px.pie(
            failure_result,
            names="구분",
            values="데이터 개수",
            title="정상 및 고장 기계 비율",
            hole=0.4,
            color="구분",
            color_discrete_map={
                "정상": "#69BCEB",
                "고장": "#F15B68"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 9. 변수별 상관관계 히트맵
# ==========================================
st.divider()
st.subheader("4. 🔗 변수별 상관관계 히트맵")

st.write(
    "상관계수는 두 변수가 함께 변화하는 정도를 나타냅니다. "
    "1에 가까우면 양의 관계, -1에 가까우면 음의 관계입니다."
)

corr_columns = [
    col for col in available_numeric
    if df[col].nunique(dropna=True) > 1
]

if len(corr_columns) >= 2:

    corr = df[corr_columns].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="변수별 상관계수 히트맵"
    )

    fig.update_layout(
        template="plotly_white",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    if "기계 고장 여부" in corr.columns:

        st.subheader("고장 여부와 각 변수의 상관계수")

        target_corr = (
            corr["기계 고장 여부"]
            .drop("기계 고장 여부")
            .sort_values(key=abs, ascending=False)
            .reset_index()
        )

        target_corr.columns = ["변수", "상관계수"]

        st.dataframe(
            target_corr.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "상관계수가 높다고 해서 해당 변수가 고장의 "
            "직접적인 원인이라는 뜻은 아닙니다."
        )


# ==========================================
# 10. 결측값 확인
# ==========================================
st.divider()
st.subheader("5. 🧹 결측값 확인")

missing_df = pd.DataFrame({
    "변수": df.columns,
    "결측값 개수": df.isnull().sum().values
})

missing_df = missing_df[
    missing_df["결측값 개수"] > 0
]

if missing_df.empty:

    st.success("결측값이 없습니다.")

else:

    missing_df = missing_df.sort_values(
        "결측값 개수",
        ascending=False
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        missing_df,
        x="변수",
        y="결측값 개수",
        title="변수별 결측값 개수",
        color="결측값 개수",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 11. 원하는 변수 2개로 기계 고장률 분석
# ==========================================
st.divider()
st.header("🔍 원하는 변수로 기계 고장률 분석")

st.write(
    "제품 유형과 숫자형 변수 2개를 선택하면 "
    "각 조건에 따른 정상·고장 비율을 원그래프로 비교합니다."
)

analysis_variables = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
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

    st.warning("분석에 필요한 변수가 데이터에 없습니다.")

else:

    # --------------------------------
    # 1. 변수 선택
    # --------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_type = st.selectbox(
            "① 제품 유형 선택",
            ["전체", "L", "M", "H"],
            key="selected_type"
        )

    with col2:
        variable1 = st.selectbox(
            "② 첫 번째 분석 변수",
            analysis_variables,
            index=3,
            key="variable1"
        )

    with col3:
        variable2_options = [
            col for col in analysis_variables
            if col != variable1
        ]

        variable2 = st.selectbox(
            "③ 두 번째 분석 변수",
            variable2_options,
            key="variable2"
        )

    # --------------------------------
    # 2. 제품 유형에 따라 데이터 분리
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
    # 3. 변수별 고장률 계산 함수
    # --------------------------------
    def calculate_failure_rate(data, variable):

        temp = data[
            [variable, "기계 고장 여부"]
        ].copy()

        temp[variable] = pd.to_numeric(
            temp[variable],
            errors="coerce"
        )

        temp["기계 고장 여부"] = pd.to_numeric(
            temp["기계 고장 여부"],
            errors="coerce"
        )

        temp = temp.dropna()

        if len(temp) < 2 or temp[variable].nunique() < 2:
            return None, None

        median_value = temp[variable].median()

        temp["구분"] = temp[variable].apply(
            lambda x: "낮은 그룹"
            if x <= median_value
            else "높은 그룹"
        )

        result = temp.groupby("구분").agg(
            전체_개수=("기계 고장 여부", "count"),
            고장_개수=("기계 고장 여부", "sum")
        )

        result["정상_개수"] = (
            result["전체_개수"] - result["고장_개수"]
        )

        result["고장률(%)"] = (
            result["고장_개수"]
            / result["전체_개수"] * 100
        )

        result = result.reindex(
            ["낮은 그룹", "높은 그룹"]
        ).dropna(subset=["전체_개수"])

        return result.reset_index(), median_value

    # --------------------------------
    # 4. 두 변수의 고장률 계산
    # --------------------------------
    result1, median1 = calculate_failure_rate(
        analysis_df,
        variable1
    )

    result2, median2 = calculate_failure_rate(
        analysis_df,
        variable2
    )

    # --------------------------------
    # 5. 원그래프 표시 함수
    # --------------------------------
    def show_failure_pies(variable, result, median_value):

        st.markdown(f"### 📌 {variable}")

        if result is None or result.empty:

            st.warning("분석할 데이터가 부족합니다.")
            return

        st.info(
            f"분석 기준 중앙값: {median_value:,.2f} "
            "이하를 낮은 그룹, 초과를 높은 그룹으로 구분했습니다."
        )

        # 낮은 그룹과 높은 그룹을 나란히 표시
        pie_col1, pie_col2 = st.columns(2)

        group_columns = [
            ("낮은 그룹", pie_col1),
            ("높은 그룹", pie_col2)
        ]

        for group_name, column in group_columns:

            group_row = result[
                result["구분"] == group_name
            ]

            with column:

                st.markdown(f"#### {group_name}")

                if group_row.empty:

                    st.warning("해당 그룹에 데이터가 없습니다.")
                    continue

                row = group_row.iloc[0]

                pie_data = pd.DataFrame({
                    "상태": ["정상", "고장"],
                    "기계 수": [
                        int(row["정상_개수"]),
                        int(row["고장_개수"])
                    ]
                })

                fig = px.pie(
                    pie_data,
                    names="상태",
                    values="기계 수",
                    hole=0.45,
                    color="상태",
                    color_discrete_map={
                        "정상": "#69BCEB",
                        "고장": "#F15B68"
                    }
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "기계 수: %{value:,}개<br>"
                        "비율: %{percent}<extra></extra>"
                    ),
                    marker=dict(
                        line=dict(
                            color="white",
                            width=2
                        )
                    )
                )

                fig.update_layout(
                    template="plotly_white",
                    height=330,
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.12,
                        xanchor="center",
                        x=0.5
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"pie_{variable}_{group_name}"
                )

                st.metric(
                    f"{group_name} 고장률",
                    f"{row['고장률(%)']:.2f}%"
                )

                st.caption(
                    f"전체 {int(row['전체_개수']):,}개 중 "
                    f"고장 {int(row['고장_개수']):,}개"
                )

        # --------------------------------
        # 6. 그룹별 상세 결과표
        # --------------------------------
        st.markdown("#### 📋 그룹별 상세 결과")

        display_result = result[
            [
                "구분",
                "전체_개수",
                "정상_개수",
                "고장_개수",
                "고장률(%)"
            ]
        ].copy()

        display_result.columns = [
            "변수 구분",
            "전체 기계 수",
            "정상 기계 수",
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

        # --------------------------------
        # 7. 그룹 간 고장률 차이 설명
        # --------------------------------
        if len(result) == 2:

            high_rate = result.loc[
                result["고장률(%)"].idxmax()
            ]

            low_rate = result.loc[
                result["고장률(%)"].idxmin()
            ]

            difference = (
                high_rate["고장률(%)"]
                - low_rate["고장률(%)"]
            )

            st.success(
                f"{high_rate['구분']}의 고장률이 "
                f"{high_rate['고장률(%)']:.2f}%로 더 높습니다. "
                f"두 그룹의 차이는 {difference:.2f}%p입니다."
            )

        st.caption(
            "원그래프는 각 그룹 안에서 정상 기계와 고장 기계가 "
            "차지하는 비율을 보여줍니다. "
            "고장률은 해당 그룹의 전체 기계 중 고장 기계의 비율입니다."
        )

    # --------------------------------
    # 8. 두 변수 분석 결과 출력
    # --------------------------------
    st.divider()
    st.subheader("📈 변수별 기계 고장률 비교")

    show_failure_pies(
        variable1,
        result1,
        median1
    )

    st.divider()

    show_failure_pies(
        variable2,
        result2,
        median2
    )


# ==========================================
# 12. 분석 질문
# ==========================================
st.divider()
st.subheader("📝 분석해 볼 질문")

st.markdown(
    """
    1. 정상 기계와 고장 난 기계의 평균 온도는 다를까?
    2. 공구 마모 시간이 길수록 고장 비율이 높을까?
    3. 회전 속도와 토크는 어떤 관계가 있을까?
    4. 고장 여부와 상관관계가 높은 변수는 무엇일까?
    5. 제품 유형에 따라 고장 건수가 다를까?
    6. 변수의 값이 높은 그룹과 낮은 그룹 중 어느 쪽의 고장률이 높을까?
    """
)

st.caption(
    "이 분석은 데이터에서 나타나는 관계를 살펴보는 탐색적 분석입니다. "
    "상관관계나 고장률 차이만으로 직접적인 원인을 확정할 수는 없습니다."
)
