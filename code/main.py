
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기계 고장 기초통계 분석실",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 기계 고장 기초통계 분석실")
st.write("기계 작동 데이터를 살펴보고 고장에 영향을 주는 요인을 분석합니다.")

DATA_URL = "https://raw.githubusercontent.com/jisujisujisu333/Machine-Failure-Prediction-Project/refs/heads/main/data/%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            data = pd.read_csv(DATA_URL, encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    return pd.read_csv(DATA_URL)


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 변수 이름 한글로 변경
# --------------------------------------------------
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

# 숫자형 데이터 변환
numeric_cols = [
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

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
st.subheader("📊 데이터 요약")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터 수", f"{len(df):,}개")
col2.metric("전체 변수 수", f"{len(df.columns)}개")

missing_total = int(df.isnull().sum().sum())
col3.metric("전체 결측치 수", f"{missing_total:,}개")

st.divider()


# --------------------------------------------------
# 1. 기초 통계 분석
# --------------------------------------------------
st.header("1️⃣ 수치형 변수의 기초 통계")

st.write(
    "기계의 온도, 회전 속도, 토크, 공구 마모 시간 등의 "
    "평균과 최솟값, 최댓값 등을 확인합니다."
)

basic_cols = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
    "공구 마모 시간(분)"
]

available_basic_cols = [c for c in basic_cols if c in df.columns]

if available_basic_cols:
    st.dataframe(
        df[available_basic_cols].describe().round(2),
        use_container_width=True
    )

    selected_basic = st.selectbox(
        "히스토그램으로 확인할 변수",
        available_basic_cols
    )

    fig = px.histogram(
        df,
        x=selected_basic,
        nbins=30,
        title=f"{selected_basic} 분포",
        labels={selected_basic: selected_basic},
        color_discrete_sequence=["#4A90E2"]
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()


# --------------------------------------------------
# 이상치 확인: 박스플롯
# --------------------------------------------------
st.header("🔎 이상치 확인하기")

st.write(
    "박스플롯을 이용해 다른 데이터와 차이가 큰 값을 확인합니다. "
    "상자 바깥에 점으로 표시되는 값은 IQR 기준 이상치 후보입니다."
)

outlier_cols = [
    c for c in basic_cols if c in df.columns
]

if outlier_cols:
    selected_outlier = st.selectbox(
        "박스플롯으로 확인할 변수",
        outlier_cols,
        key="outlier_select"
    )

    values = pd.to_numeric(
        df[selected_outlier], errors="coerce"
    ).dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_data = values[
        (values < lower_bound) | (values > upper_bound)
    ]

    fig = px.box(
        df,
        y=selected_outlier,
        points="outliers",
        title=f"{selected_outlier} 박스플롯",
        color_discrete_sequence=["#4A90E2"]
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("이상치 후보 개수", f"{len(outlier_data):,}개")
    c2.metric(
        "이상치 후보 비율",
        f"{len(outlier_data) / len(values) * 100:.2f}%"
        if len(values) > 0 else "0%"
    )
    c3.metric("분석 데이터 수", f"{len(values):,}개")

    st.write("**IQR 기준 범위**")
    st.write(f"- 하한: {lower_bound:.2f}")
    st.write(f"- 상한: {upper_bound:.2f}")

    st.caption(
        "※ 이상치 후보는 통계적으로 다른 값과 차이가 큰 데이터입니다. "
        "기계의 실제 특이 작동일 수도 있으므로 무조건 삭제하지 않습니다."
    )

    # 전체 변수별 이상치 개수
    st.subheader("변수별 이상치 후보 개수")

    outlier_summary = []

    for col in outlier_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(vals) == 0:
            continue

        q1_col = vals.quantile(0.25)
        q3_col = vals.quantile(0.75)
        iqr_col = q3_col - q1_col

        low = q1_col - 1.5 * iqr_col
        high = q3_col + 1.5 * iqr_col

        count = int(((vals < low) | (vals > high)).sum())

        outlier_summary.append({
            "변수": col,
            "이상치 후보 개수": count,
            "이상치 후보 비율(%)": round(count / len(vals) * 100, 2)
        })

    st.dataframe(
        pd.DataFrame(outlier_summary),
        use_container_width=True,
        hide_index=True
    )

st.divider()


# --------------------------------------------------
# 2. 범주형 변수의 분포
# --------------------------------------------------
st.header("2️⃣ 범주형 변수의 분포")

st.write(
    "제품 유형별 데이터가 얼마나 분포되어 있는지 원형 그래프로 확인합니다."
)

if "제품 유형" in df.columns:
    type_counts = (
        df["제품 유형"]
        .fillna("값 없음")
        .value_counts()
        .reset_index()
    )

    type_counts.columns = ["제품 유형", "개수"]

    fig = px.pie(
        type_counts,
        names="제품 유형",
        values="개수",
        title="제품 유형별 분포",
        hole=0,
        color_discrete_sequence=["#4A90E2", "#50C878", "#F4B942"]
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        type_counts,
        use_container_width=True,
        hide_index=True
    )

st.divider()


# --------------------------------------------------
# 3. 기계 고장 여부 분석
# --------------------------------------------------
st.header("3️⃣ 기계 고장 여부 분석")

st.write("전체 기계 중 정상 작동한 기계와 고장이 발생한 기계의 비율입니다.")

if "기계 고장 여부" in df.columns:
    failure_counts = (
        df["기계 고장 여부"]
        .map({0: "정상", 1: "고장"})
        .fillna("기타")
        .value_counts()
        .reset_index()
    )

    failure_counts.columns = ["상태", "개수"]

    fig = px.pie(
        failure_counts,
        names="상태",
        values="개수",
        title="정상 및 고장 비율",
        color="상태",
        color_discrete_map={
            "정상": "#4A90E2",
            "고장": "#EF5350",
            "기타": "#999999"
        }
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()


# --------------------------------------------------
# 4. 상관관계 분석
# --------------------------------------------------
st.header("4️⃣ 변수 간 상관관계 분석")

st.write(
    "변수 사이의 상관관계를 히트맵으로 확인합니다. "
    "상관계수가 1에 가까우면 양의 관계, -1에 가까우면 음의 관계를 나타냅니다."
)

corr_cols = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
    "공구 마모 시간(분)",
    "기계 고장 여부"
]

available_corr_cols = [c for c in corr_cols if c in df.columns]

if len(available_corr_cols) >= 2:
    corr = df[available_corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="기계 작동 변수 간 상관관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    if "기계 고장 여부" in corr.columns:
        st.subheader("기계 고장 여부와의 상관관계")

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

st.divider()


# --------------------------------------------------
# 5. 결측치 분석
# --------------------------------------------------
st.header("5️⃣ 결측치 확인")

st.write("각 변수에 비어 있는 데이터가 얼마나 있는지 확인합니다.")

missing_summary = pd.DataFrame({
    "변수": df.columns,
    "결측치 개수": df.isnull().sum().values,
    "결측치 비율(%)": (
        df.isnull().mean().values * 100
    ).round(2)
})

st.dataframe(
    missing_summary,
    use_container_width=True,
    hide_index=True
)

if missing_total == 0:
    st.success("결측치가 없습니다.")
else:
    st.warning(f"전체 결측치가 {missing_total:,}개 있습니다.")

st.divider()


# --------------------------------------------------
# 6. 변수별 고장률 비교
# --------------------------------------------------
st.header("6️⃣ 변수에 따른 기계 고장률 비교")

st.write(
    "제품 유형을 선택하고 두 가지 변수를 골라 낮은 그룹과 높은 그룹의 "
    "고장률을 비교합니다."
)

if "기계 고장 여부" in df.columns:

    # 제품 유형 선택
    type_options = ["전체"] + sorted(
        df["제품 유형"].dropna().unique().tolist()
    )

    selected_type = st.selectbox(
        "제품 유형 선택",
        type_options
    )

    compare_cols = [
        "공기 온도(K)",
        "공정 온도(K)",
        "회전 속도(rpm)",
        "토크(N·m)",
        "공구 마모 시간(분)"
    ]

    compare_cols = [c for c in compare_cols if c in df.columns]

    col1, col2 = st.columns(2)

    with col1:
        variable1 = st.selectbox(
            "첫 번째 비교 변수",
            compare_cols,
            index=0
        )

    with col2:
        variable2 = st.selectbox(
            "두 번째 비교 변수",
            compare_cols,
            index=1 if len(compare_cols) > 1 else 0
        )

    # 제품 유형 필터링
    compare_df = df.copy()

    if selected_type != "전체":
        compare_df = compare_df[
            compare_df["제품 유형"] == selected_type
        ]

    def show_failure_comparison(data, variable):
        temp = data[[variable, "기계 고장 여부"]].copy()

        temp[variable] = pd.to_numeric(
            temp[variable], errors="coerce"
        )

        temp = temp.dropna()

        if temp.empty:
            st.warning(f"{variable}의 분석 가능한 데이터가 없습니다.")
            return

        median_value = temp[variable].median()

        temp["그룹"] = temp[variable].apply(
            lambda x: "낮은 그룹" if x <= median_value else "높은 그룹"
        )

        st.subheader(f"📌 {variable}에 따른 고장률")

        group_cols = st.columns(2)

        for i, group_name in enumerate(["낮은 그룹", "높은 그룹"]):
            group_data = temp[temp["그룹"] == group_name]

            total = len(group_data)
            failure = int(
                (group_data["기계 고장 여부"] == 1).sum()
            )
            normal = int(
                (group_data["기계 고장 여부"] == 0).sum()
            )

            failure_rate = failure / total * 100 if total > 0 else 0

            with group_cols[i]:
                st.markdown(f"**{group_name}**")
                st.metric("고장률", f"{failure_rate:.2f}%")

                pie_data = pd.DataFrame({
                    "상태": ["정상", "고장"],
                    "개수": [normal, failure]
                })

                fig = px.pie(
                    pie_data,
                    names="상태",
                    values="개수",
                    color="상태",
                    color_discrete_map={
                        "정상": "#4A90E2",
                        "고장": "#EF5350"
                    },
                    title=f"{group_name} 정상·고장 비율"
                )

                fig.update_traces(
                    textinfo="label+percent",
                    textposition="inside"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    f"전체 {total:,}개 / 정상 {normal:,}개 / 고장 {failure:,}개"
                )

        st.caption(
            f"낮은 그룹과 높은 그룹의 구분 기준(중앙값): {median_value:.2f}"
        )

    show_failure_comparison(compare_df, variable1)

    st.divider()

    if variable2 != variable1:
        show_failure_comparison(compare_df, variable2)
    else:
        st.info("두 번째 변수는 첫 번째 변수와 다르게 선택하면 비교할 수 있습니다.")

st.divider()


# --------------------------------------------------
# 분석 질문
# --------------------------------------------------
st.header("💡 생각해 볼 분석 질문")

st.markdown("""
1. 어떤 제품 유형의 비율이 가장 높을까?
2. 기계 고장과 가장 관련이 높은 변수는 무엇일까?
3. 회전 속도나 토크가 높을수록 고장률도 높아질까?
4. 공구 마모 시간이 길어지면 고장률은 어떻게 변할까?
5. 박스플롯에서 확인한 이상치 후보는 기계 고장과 어떤 관계가 있을까?
""")

st.caption("기계 고장 기초통계 분석실")
