
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="기계 고장 예측 분석실",
    page_icon="⚙️",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/jisujisujisu333/Machine-Failure-Prediction-Project/refs/heads/main/data/%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"


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
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# ==========================================
# 3. 변수명 변경 및 자료형 변환
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

df = df.rename(columns=rename_dict).copy()


# 수치형 변수
basic_cols = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
    "공구 마모 시간(분)"
]

numeric_cols = basic_cols + [
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


# ★ 중요: 모든 메뉴에서 사용할 수 있도록 전역 위치에 설정
available_cols = [
    col for col in basic_cols
    if col in df.columns
]

missing_total = int(df.isnull().sum().sum())


# ==========================================
# 4. 사이드바 메뉴
# ==========================================
st.sidebar.title("⚙️ 메인 목차")

menu = st.sidebar.radio(
    "분석 항목",
    [
        "🏠 홈",
        "1. 문제 정의",
        "2. 데이터 분석",
        "3. 데이터 전처리",
        "4. 기준선 설정",
        "5. 모델 학습 및 평가"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.subheader("📊 데이터 정보")
st.sidebar.metric("행 개수", f"{len(df):,}개")
st.sidebar.metric("열 개수", f"{len(df.columns)}개")
st.sidebar.metric("결측치", f"{missing_total:,}개")

st.title("⚙️ 기계 고장 예측 분석실")
st.caption("기계 작동 데이터를 분석하여 고장과 관련된 요인을 알아봅니다.")


# ==========================================
# 홈
# ==========================================
if menu == "🏠 홈":

    st.header("기계 고장 예측 분석실")

    st.write(
        "이 프로젝트는 기계의 온도, 회전 속도, 토크, "
        "공구 마모 시간 등의 데이터를 분석하여 "
        "기계 고장과 관련된 요인을 알아보고 "
        "기계 고장 여부를 예측하는 것을 목표로 합니다."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("전체 데이터", f"{len(df):,}개")
    c2.metric("전체 변수", f"{len(df.columns)}개")
    c3.metric("결측치", f"{missing_total:,}개")

    st.subheader("분석 목차")

    st.markdown("""
    **1. 문제 정의**
    - 주제 선정 이유와 분석 목표

    **2. 데이터 분석**
    - 기초 통계 및 그래프
    - 제품 유형별 분포
    - 기계 고장률 분석
    - 변수 간 상관관계

    **3. 데이터 전처리**
    - 변수명 변경 및 자료형 변환
    - 결측치 확인
    - 이상치 탐색

    **4. 기준선 설정**
    - 다수 클래스 기준선 성능 확인

    **5. 모델 학습 및 평가**
    - 랜덤 포레스트 모델
    - 정확도, 정밀도, 재현율, F1 점수
    - 혼동 행렬
    """)

    st.info("왼쪽 목차를 눌러 원하는 분석을 확인하세요.")


# ==========================================
# 1. 문제 정의
# ==========================================
elif menu == "1. 문제 정의":

    st.header("1️⃣ 문제 정의")

    st.subheader("주제")

    st.write(
        "기계 고장에 영향을 주는 요인은 무엇이며, "
        "기계 고장을 예측할 수 있을까?"
    )

    st.subheader("주제 선정 이유")

    st.write(
        "기계가 갑자기 고장 나면 생산이 중단되거나 "
        "수리 비용이 발생할 수 있다. "
        "기계의 온도, 회전 속도, 토크, 공구 마모 시간 등을 "
        "분석하면 고장과 관련된 요인을 알아볼 수 있다고 생각하여 "
        "이 주제를 선택하였다."
    )

    st.subheader("분석 목표")

    st.markdown("""
    - 기계 작동 변수의 기초 통계와 분포를 확인한다.
    - 정상 작동과 고장 발생 비율을 분석한다.
    - 변수에 따라 고장률에 차이가 있는지 비교한다.
    - 기계 작동 정보를 이용해 고장 여부를 예측한다.
    - 예측 모델의 성능을 평가한다.
    """)

    st.subheader("주요 변수 설명")

    meaning = pd.DataFrame({
        "변수": [
            "제품 유형",
            "공기 온도(K)",
            "공정 온도(K)",
            "회전 속도(rpm)",
            "토크(N·m)",
            "공구 마모 시간(분)",
            "기계 고장 여부"
        ],
        "의미": [
            "제품의 유형",
            "기계 주변 공기의 온도",
            "기계 공정 중 온도",
            "기계의 회전 속도",
            "기계에 걸리는 회전력",
            "공구를 사용한 시간",
            "정상(0) 또는 고장(1)"
        ]
    })

    st.dataframe(
        meaning,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 2. 데이터 분석
# ==========================================
elif menu == "2. 데이터 분석":

    st.header("2️⃣ 데이터 분석")

    # ① 기초 통계
    st.subheader("① 기초 통계")

    if available_cols:

        st.dataframe(
            df[available_cols].describe().round(2),
            use_container_width=True
        )

        selected = st.selectbox(
            "히스토그램 변수 선택",
            available_cols
        )

        fig = px.histogram(
            df,
            x=selected,
            nbins=30,
            title=f"{selected} 분포",
            color_discrete_sequence=["#4A90E2"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ② 제품 유형별 분포
    st.subheader("② 제품 유형별 분포")

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
            color_discrete_sequence=[
                "#4A90E2",
                "#50C878",
                "#F4B942"
            ]
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

    # ③ 정상 및 고장 비율
    st.subheader("③ 정상 및 고장 비율")

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
            color="상태",
            color_discrete_map={
                "정상": "#4A90E2",
                "고장": "#EF5350",
                "기타": "#999999"
            },
            title="정상 및 고장 비율"
        )

        fig.update_traces(
            textinfo="label+percent",
            textposition="inside"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ④ 변수 간 상관관계
    st.subheader("④ 변수 간 상관관계")

    if (
        "기계 고장 여부" in df.columns
        and len(available_cols) >= 1
    ):

        corr_cols = available_cols + ["기계 고장 여부"]

        corr = df[corr_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="변수 간 상관관계"
        )

        st.plotly_chart(fig, use_container_width=True)

        target_corr = (
            corr["기계 고장 여부"]
            .drop("기계 고장 여부")
            .sort_values(key=abs, ascending=False)
            .reset_index()
        )

        target_corr.columns = ["변수", "상관계수"]

        st.write("기계 고장 여부와 각 변수의 상관계수")

        st.dataframe(
            target_corr.round(3),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ⑤ 변수에 따른 고장률 비교
    st.subheader("⑤ 변수에 따른 기계 고장률 비교")

    if (
        "제품 유형" in df.columns
        and "기계 고장 여부" in df.columns
        and len(available_cols) >= 2
    ):

        type_options = ["전체"] + sorted(
            df["제품 유형"].dropna().unique().tolist()
        )

        selected_type = st.selectbox(
            "제품 유형 선택",
            type_options
        )

        col1, col2 = st.columns(2)

        with col1:
            var1 = st.selectbox(
                "첫 번째 비교 변수",
                available_cols
            )

        with col2:
            var2 = st.selectbox(
                "두 번째 비교 변수",
                available_cols,
                index=1,
                key="var2"
            )

        compare_df = df.copy()

        if selected_type != "전체":
            compare_df = compare_df[
                compare_df["제품 유형"] == selected_type
            ]

        def show_comparison(data, variable):

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

            if temp.empty:
                st.warning("분석할 데이터가 없습니다.")
                return

            median = temp[variable].median()

            temp["그룹"] = temp[variable].apply(
                lambda x: "낮은 그룹"
                if x <= median
                else "높은 그룹"
            )

            st.markdown(f"**📌 {variable}에 따른 고장률**")

            cols = st.columns(2)

            for i, group in enumerate(
                ["낮은 그룹", "높은 그룹"]
            ):

                part = temp[temp["그룹"] == group]

                total = len(part)

                failure = int(
                    (part["기계 고장 여부"] == 1).sum()
                )

                normal = int(
                    (part["기계 고장 여부"] == 0).sum()
                )

                rate = (
                    failure / total * 100
                    if total > 0 else 0
                )

                with cols[i]:

                    st.metric(
                        f"{group} 고장률",
                        f"{rate:.2f}%"
                    )

                    pie = px.pie(
                        pd.DataFrame({
                            "상태": ["정상", "고장"],
                            "개수": [normal, failure]
                        }),
                        names="상태",
                        values="개수",
                        color="상태",
                        color_discrete_map={
                            "정상": "#4A90E2",
                            "고장": "#EF5350"
                        }
                    )

                    pie.update_traces(
                        textinfo="label+percent"
                    )

                    st.plotly_chart(
                        pie,
                        use_container_width=True
                    )

                    st.caption(
                        f"전체 {total:,}개 / "
                        f"정상 {normal:,}개 / "
                        f"고장 {failure:,}개"
                    )

            st.caption(
                f"낮은 그룹과 높은 그룹의 구분 기준 중앙값: {median:.2f}"
            )

        show_comparison(compare_df, var1)

        if var2 != var1:
            st.divider()
            show_comparison(compare_df, var2)
        else:
            st.warning(
                "두 번째 변수는 첫 번째 변수와 다르게 선택해 주세요."
            )


# ==========================================
# 3. 데이터 전처리
# ==========================================
elif menu == "3. 데이터 전처리":

    st.header("3️⃣ 데이터 전처리")

    st.subheader("① 데이터 정리")

    st.markdown("""
    - 영어 변수명을 이해하기 쉬운 한국어로 변경하였다.
    - 숫자형 변수를 숫자 자료형으로 변환하였다.
    - 제품 번호와 데이터 번호는 식별자이므로 예측 모델에서 제외한다.
    """)

    st.divider()

    # 결측치
    st.subheader("② 결측치 확인")

    st.metric(
        "전체 결측치 개수",
        f"{missing_total:,}개"
    )

    missing_table = pd.DataFrame({
        "변수": df.columns,
        "결측치 개수": df.isnull().sum().values,
        "결측치 비율(%)": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    if missing_total == 0:
        st.success("현재 데이터에는 결측치가 없습니다.")
    else:
        st.warning(
            "결측치가 있는 변수를 확인해야 합니다."
        )

    st.divider()

    # 이상치
    st.subheader("③ 이상치 탐색")

    st.write(
        "IQR 방식으로 이상치 후보를 확인합니다. "
        "이상치 후보가 반드시 잘못된 데이터라는 뜻은 아니므로 "
        "자동으로 삭제하지 않습니다."
    )

    outlier_summary = []

    for col in available_cols:

        values = df[col].dropna()

        if len(values) == 0:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        outlier_summary.append({
            "변수": col,
            "하한": round(lower, 2),
            "상한": round(upper, 2),
            "이상치 후보 개수": count,
            "이상치 후보 비율(%)": round(
                count / len(values) * 100,
                2
            )
        })

    st.dataframe(
        pd.DataFrame(outlier_summary),
        use_container_width=True,
        hide_index=True
    )

    selected_outlier = st.selectbox(
        "박스플롯 변수 선택",
        available_cols
    )

    fig = px.box(
        df,
        y=selected_outlier,
        points="outliers",
        title=f"{selected_outlier} 박스플롯",
        color_discrete_sequence=["#4A90E2"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "이상치 후보는 자동 삭제하지 않고 그래프와 통계로 확인합니다."
    )


# ==========================================
# 4. 기준선 설정
# ==========================================
elif menu == "4. 기준선 설정":

    st.header("4️⃣ 기준선 설정")

    st.write(
        "기준선은 복잡한 모델을 만들기 전에 비교 기준으로 사용하는 "
        "가장 단순한 예측 방법입니다."
    )

    st.subheader("다수 클래스 기준선")

    target = pd.to_numeric(
        df["기계 고장 여부"],
        errors="coerce"
    ).dropna()

    normal_count = int((target == 0).sum())
    failure_count = int((target == 1).sum())

    majority_count = max(
        normal_count,
        failure_count
    )

    baseline_accuracy = (
        majority_count / len(target) * 100
        if len(target) > 0 else 0
    )

    counts = pd.DataFrame({
        "상태": ["정상", "고장"],
        "개수": [normal_count, failure_count]
    })

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "다수 클래스 기준선 정확도",
        f"{baseline_accuracy:.2f}%"
    )

    if normal_count > failure_count:

        st.write(
            "정상 데이터가 더 많기 때문에 모든 기계를 정상이라고 "
            "예측하는 방식을 기준선으로 설정합니다."
        )

    else:

        st.write(
            "고장 데이터가 더 많기 때문에 모든 기계를 고장이라고 "
            "예측하는 방식을 기준선으로 설정합니다."
        )

    st.warning(
        "정확도만 높다고 고장 예측을 잘하는 것은 아닙니다. "
        "고장 데이터를 얼마나 찾아내는지 재현율도 함께 확인해야 합니다."
    )


# ==========================================
# 5. 모델 학습 및 평가
# ==========================================
elif menu == "5. 모델 학습 및 평가":

    st.header("5️⃣ 모델 학습 및 평가")

    st.write(
        "기계 작동 정보를 이용하여 기계 고장 여부를 예측하는 "
        "랜덤 포레스트 분류 모델을 학습합니다."
    )

    st.subheader("① 모델에 사용하는 변수")

    st.markdown("""
    **입력 변수**
    - 제품 유형
    - 공기 온도
    - 공정 온도
    - 회전 속도
    - 토크
    - 공구 마모 시간

    **예측 대상**
    - 기계 고장 여부 (0=정상, 1=고장)

    공구 마모 고장, 열 방출 고장, 동력 고장, 과부하 고장,
    무작위 고장은 정답과 직접 연결된 정보이므로 입력 변수에서 제외합니다.
    """)

    test_size = st.slider(
        "평가용 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.3,
        step=0.05
    )

    if st.button(
        "모델 학습 및 평가하기",
        type="primary"
    ):

        # 전역 변수 available_cols 사용
        model_cols = ["제품 유형"] + available_cols

        required_cols = model_cols + ["기계 고장 여부"]

        # 필요한 변수만 추출
        model_df = df[required_cols].copy()

        # 숫자형 변환
        model_df["기계 고장 여부"] = pd.to_numeric(
            model_df["기계 고장 여부"],
            errors="coerce"
        )

        for col in available_cols:
            model_df[col] = pd.to_numeric(
                model_df[col],
                errors="coerce"
            )

        # 학습에 필요한 결측 행 제외
        model_df = model_df.dropna()

        # 입력 변수와 정답 분리
        X = model_df[model_cols]
        y = model_df["기계 고장 여부"].astype(int)

        if len(model_df) == 0:

            st.error(
                "모델 학습에 사용할 수 있는 데이터가 없습니다."
            )

        elif y.nunique() < 2:

            st.error(
                "정상과 고장 데이터가 모두 있어야 모델을 학습할 수 있습니다."
            )

        elif y.value_counts().min() < 2:

            st.error(
                "학습과 평가를 나누기에 각 범주의 데이터가 부족합니다."
            )

        else:

            # 학습 데이터와 평가 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            # ----------------------------------
            # 기준선 모델
            # ----------------------------------
            baseline = DummyClassifier(
                strategy="most_frequent"
            )

            baseline.fit(X_train, y_train)

            baseline_pred = baseline.predict(X_test)

            # ----------------------------------
            # 랜덤 포레스트 모델
            # ----------------------------------
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "type",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        ["제품 유형"]
                    ),
                    (
                        "numeric",
                        "passthrough",
                        available_cols
                    )
                ]
            )

            model = Pipeline([
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1
                    )
                )
            ])

            with st.spinner(
                "모델을 학습하고 있습니다..."
            ):

                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

            st.success("모델 학습이 완료되었습니다.")

            st.write(
                f"학습 데이터: {len(X_train):,}개 / "
                f"평가 데이터: {len(X_test):,}개"
            )

            # ----------------------------------
            # 평가 지표 계산
            # ----------------------------------
            results = []

            for name, pred in [
                ("기준선 모델", baseline_pred),
                ("랜덤 포레스트", prediction)
            ]:

                results.append({
                    "모델": name,
                    "정확도": accuracy_score(
                        y_test, pred
                    ),
                    "정밀도": precision_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),
                    "재현율": recall_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),
                    "F1 점수": f1_score(
                        y_test,
                        pred,
                        zero_division=0
                    )
                })

            result_df = pd.DataFrame(results)

            display_df = result_df.copy()

            for col in [
                "정확도",
                "정밀도",
                "재현율",
                "F1 점수"
            ]:

                display_df[col] = (
                    display_df[col] * 100
                ).round(2).astype(str) + "%"

            st.subheader("② 모델 성능 비교")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ----------------------------------
            # 혼동 행렬
            # ----------------------------------
            st.subheader("③ 혼동 행렬")

            cm = confusion_matrix(
                y_test,
                prediction,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["실제 정상", "실제 고장"],
                columns=["예측 정상", "예측 고장"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["예측 정상", "예측 고장"],
                y=["실제 정상", "실제 고장"],
                color_continuous_scale="Blues",
                title="랜덤 포레스트 혼동 행렬"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            # ----------------------------------
            # 평가 지표 설명
            # ----------------------------------
            st.subheader("④ 평가 지표 설명")

            st.markdown("""
            - **정확도:** 전체 예측 중 맞게 예측한 비율
            - **정밀도:** 고장이라고 예측한 것 중 실제 고장인 비율
            - **재현율:** 실제 고장 중 모델이 찾아낸 비율
            - **F1 점수:** 정밀도와 재현율을 함께 고려한 점수
            """)

            st.divider()

            # ----------------------------------
            # 결과 해석
            # ----------------------------------
            st.subheader("⑤ 분석 결과 해석")

            rf = result_df[
                result_df["모델"] == "랜덤 포레스트"
            ].iloc[0]

            base = result_df[
                result_df["모델"] == "기준선 모델"
            ].iloc[0]

            if rf["F1 점수"] > base["F1 점수"]:

                st.success(
                    "랜덤 포레스트의 F1 점수가 기준선보다 높습니다. "
                    "이번 평가에서는 단순 기준선보다 고장 여부를 "
                    "구분하는 데 더 나은 결과를 보였습니다."
                )

            elif rf["F1 점수"] == base["F1 점수"]:

                st.info(
                    "랜덤 포레스트와 기준선의 F1 점수가 같습니다. "
                    "기준선보다 성능이 향상되었다고 보기 어렵습니다."
                )

            else:

                st.warning(
                    "랜덤 포레스트의 F1 점수가 기준선보다 낮습니다. "
                    "변수와 모델 설정을 추가로 검토할 필요가 있습니다."
                )

            st.caption(
                "※ 이 결과는 현재 데이터 분할과 모델 설정에 따른 "
                "평가 결과이며, 다른 데이터에서도 같은 성능을 "
                "보장하지는 않습니다."
            )

    else:

        st.info(
            "버튼을 누르면 기준선과 랜덤 포레스트 모델을 "
            "같은 평가 데이터로 비교합니다."
        )


# ==========================================
# 하단
# ==========================================
st.divider()
st.caption("기계 고장 예측 분석실")
