
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
# 3. 변수명 변경
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


# ==========================================
# 4. 자료형 변환 및 공통 변수
# ==========================================
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

available_cols = [
    col for col in basic_cols
    if col in df.columns
]

missing_total = int(df.isnull().sum().sum())


# ==========================================
# 5. 공통 모델 학습 함수
# ==========================================
def prepare_model_data():

    model_cols = ["제품 유형"] + available_cols

    required_cols = model_cols + ["기계 고장 여부"]

    model_df = df[required_cols].copy()

    model_df["기계 고장 여부"] = pd.to_numeric(
        model_df["기계 고장 여부"],
        errors="coerce"
    )

    for col in available_cols:
        model_df[col] = pd.to_numeric(
            model_df[col],
            errors="coerce"
        )

    model_df = model_df.dropna()

    X = model_df[model_cols]
    y = model_df["기계 고장 여부"].astype(int)

    return X, y, model_cols


def split_data(X, y, test_size):

    if len(y) == 0 or y.nunique() < 2:
        raise ValueError("정상과 고장 데이터가 모두 필요합니다.")

    if y.value_counts().min() < 2:
        raise ValueError("학습에 필요한 데이터가 부족합니다.")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


# ==========================================
# 6. 사이드바 메뉴
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
        "기계의 온도, 회전 속도, 토크, 공구 마모 시간 등을 "
        "분석하여 기계 고장과 관련된 요인을 알아보고 "
        "기계 고장 여부를 예측하는 프로젝트입니다."
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
    - 변수 간 상관관계
    - 변수에 따른 기계 고장률 비교

    **3. 데이터 전처리**
    - 변수명 변경 및 자료형 확인
    - 결측치 확인
    - 이상치 탐색

    **4. 기준선 설정**
    - 다수 클래스 기준선 성능 확인

    **5. 모델 학습 및 평가**
    - 로지스틱 회귀 모델
    - 랜덤 포레스트 모델
    - 평가 지표 및 혼동 행렬
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
        "수리 비용이 발생할 수 있다. 기계의 온도, 회전 속도, "
        "토크, 공구 마모 시간 등을 분석하면 고장과 관련된 "
        "요인을 알아볼 수 있다고 생각하여 이 주제를 선택하였다."
    )

    st.subheader("분석 목표")

    st.markdown("""
    - 기계 작동 변수의 특성을 파악한다.
    - 변수와 기계 고장 여부 사이의 관계를 분석한다.
    - 변수에 따른 기계 고장률을 비교한다.
    - 로지스틱 회귀와 랜덤 포레스트로 고장 여부를 예측한다.
    - 예측 모델의 성능을 평가한다.
    """)

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

    st.subheader("주요 변수 설명")
    st.dataframe(meaning, use_container_width=True, hide_index=True)


# ==========================================
# 2. 데이터 분석
# ==========================================
elif menu == "2. 데이터 분석":

    st.header("2️⃣ 데이터 분석")

    st.write(
        "변수 사이의 관계와 각 변수의 조건에 따른 "
        "기계 고장률을 비교합니다."
    )

    # --------------------------------------
    # ① 변수 간 상관관계
    # --------------------------------------
    st.subheader("① 변수 간 상관관계")

    if "기계 고장 여부" in df.columns and available_cols:

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

        st.caption(
            "상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다. "
            "상관관계가 있다고 해서 원인이라고 단정할 수는 없습니다."
        )

    st.divider()

    # --------------------------------------
    # ② 변수에 따른 기계 고장률 비교
    # --------------------------------------
    st.subheader("② 변수에 따른 기계 고장률 비교")

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
                key="compare_var2"
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
                lambda x: "낮은 그룹" if x <= median else "높은 그룹"
            )

            st.markdown(f"**📌 {variable}에 따른 고장률**")

            cols = st.columns(2)

            for i, group in enumerate(["낮은 그룹", "높은 그룹"]):

                part = temp[temp["그룹"] == group]

                total = len(part)
                failure = int(
                    (part["기계 고장 여부"] == 1).sum()
                )
                normal = int(
                    (part["기계 고장 여부"] == 0).sum()
                )

                rate = failure / total * 100 if total else 0

                with cols[i]:

                    st.metric(f"{group} 고장률", f"{rate:.2f}%")

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

                    pie.update_traces(textinfo="label+percent")

                    st.plotly_chart(
                        pie,
                        use_container_width=True
                    )

                    st.caption(
                        f"전체 {total:,}개 / 정상 {normal:,}개 / 고장 {failure:,}개"
                    )

            st.caption(
                f"낮은 그룹과 높은 그룹의 구분 기준 중앙값: {median:.2f}"
            )

        show_comparison(compare_df, var1)

        if var2 != var1:
            st.divider()
            show_comparison(compare_df, var2)
        else:
            st.warning("두 번째 변수는 다른 변수로 선택해 주세요.")

    st.divider()



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

    st.subheader("② 결측치 확인")

    st.metric("전체 결측치 개수", f"{missing_total:,}개")

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
        st.warning("결측치가 있는 변수를 확인해야 합니다.")

    st.divider()

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
                count / len(values) * 100, 2
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

    st.plotly_chart(fig, use_container_width=True)

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

    majority_count = max(normal_count, failure_count)

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

    st.write(
        "다수 클래스 기준선은 더 많은 쪽의 상태만 예측하는 "
        "단순한 모델입니다. 다른 모델의 성능을 비교할 때 사용합니다."
    )


# ==========================================
# 5. 모델 학습 및 평가
# ==========================================
elif menu == "5. 모델 학습 및 평가":

    st.header("5️⃣ 모델 학습 및 평가")

    st.write(
        "로지스틱 회귀와 랜덤 포레스트를 이용하여 기계 고장 여부를 "
        "예측하고 각 모델의 성능을 평가합니다."
    )

    # --------------------------------------
    # ① 로지스틱 회귀 분석
    # --------------------------------------
    st.subheader("① 로지스틱 회귀 분석")

    st.write(
        "로지스틱 회귀는 입력 변수들을 이용해 기계가 고장 날 "
        "확률을 계산하고, 정상과 고장 중 하나로 분류하는 방법입니다."
    )

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

    고장 유형을 나타내는 TWF, HDF, PWF, OSF, RNF는
    정답과 직접 연결된 정보이므로 입력 변수에서 제외합니다.
    """)

    test_size = st.slider(
        "평가용 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.3,
        step=0.05,
        key="logistic_test_size"
    )

    if st.button("로지스틱 회귀 실행하기", type="primary"):

        try:

            X, y, model_cols = prepare_model_data()

            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore"),
                        ["제품 유형"]
                    ),
                    (
                        "numeric",
                        StandardScaler(),
                        available_cols
                    )
                ]
            )

            logistic_model = Pipeline([
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ])

            with st.spinner("로지스틱 회귀 모델을 학습하고 있습니다..."):
                logistic_model.fit(X_train, y_train)
                prediction = logistic_model.predict(X_test)

            probability = logistic_model.predict_proba(X_test)[:, 1]

            st.success("로지스틱 회귀 분석이 완료되었습니다.")

            st.write(
                f"학습 데이터: {len(X_train):,}개 / "
                f"평가 데이터: {len(X_test):,}개"
            )

            # 평가 지표
            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(
                y_test, prediction, zero_division=0
            )
            recall = recall_score(
                y_test, prediction, zero_division=0
            )
            f1 = f1_score(
                y_test, prediction, zero_division=0
            )

            st.markdown("**모델 성능 평가**")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("정확도", f"{accuracy * 100:.2f}%")
            m2.metric("정밀도", f"{precision * 100:.2f}%")
            m3.metric("재현율", f"{recall * 100:.2f}%")
            m4.metric("F1 점수", f"{f1 * 100:.2f}%")

            st.divider()

            # 혼동 행렬
            st.markdown("**혼동 행렬**")

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

            st.dataframe(cm_df, use_container_width=True)

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["예측 정상", "예측 고장"],
                y=["실제 정상", "실제 고장"],
                color_continuous_scale="Blues",
                title="로지스틱 회귀 혼동 행렬"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # 계수 분석
            st.markdown("**변수별 영향 방향 확인**")

            fitted_preprocessor = logistic_model.named_steps[
                "preprocessor"
            ]

            fitted_classifier = logistic_model.named_steps[
                "classifier"
            ]

            feature_names = fitted_preprocessor.get_feature_names_out()

            coefficients = fitted_classifier.coef_[0]

            coef_df = pd.DataFrame({
                "변수": feature_names,
                "회귀 계수": coefficients
            })

            coef_df["영향 방향"] = coef_df["회귀 계수"].apply(
                lambda x: "고장 예측 방향" if x > 0
                else "정상 예측 방향" if x < 0
                else "영향 거의 없음"
            )

            coef_df["절댓값"] = coef_df["회귀 계수"].abs()

            coef_df = coef_df.sort_values(
                "절댓값",
                ascending=False
            )

            st.dataframe(
                coef_df[
                    ["변수", "회귀 계수", "영향 방향"]
                ].round(4),
                use_container_width=True,
                hide_index=True
            )

            coef_fig = px.bar(
                coef_df,
                x="회귀 계수",
                y="변수",
                orientation="h",
                color="회귀 계수",
                color_continuous_scale="RdBu_r",
                title="로지스틱 회귀 변수별 계수"
            )

            st.plotly_chart(
                coef_fig,
                use_container_width=True
            )

            st.caption(
                "계수가 양수이면 다른 조건이 같을 때 고장(1)으로 "
                "분류되는 방향, 음수이면 정상(0)으로 분류되는 방향을 "
                "나타냅니다. 계수의 크기는 변수의 단위와 표준화 여부를 "
                "고려해서 해석해야 합니다."
            )

            st.divider()

            # 예측 확률 예시
            st.markdown("**평가 데이터의 고장 예측 확률 예시**")

            result_table = pd.DataFrame({
                "실제 상태": y_test.map({
                    0: "정상",
                    1: "고장"
                }).values,
                "예측 상태": pd.Series(prediction).map({
                    0: "정상",
                    1: "고장"
                }).values,
                "고장 예측 확률(%)": (
                    probability * 100
                ).round(2)
            })

            st.dataframe(
                result_table.head(20),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"로지스틱 회귀 실행 중 오류가 발생했습니다: {e}")

    else:
        st.info("버튼을 누르면 로지스틱 회귀 모델을 학습하고 평가합니다.")

    st.divider()

    st.caption(
        "※ 로지스틱 회귀의 계수는 변수와 고장 예측 사이의 관계를 "
        "나타내며, 실제 고장의 원인이라고 단정할 수는 없습니다."
    )


    st.divider()
    st.subheader("② 랜덤 포레스트")

    st.write("랜덤 포레스트는 여러 결정 트리를 결합하여 예측하는 분류 모델입니다.")

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
    """)

    test_size = st.slider(
        "평가용 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.3,
        step=0.05,
        key="rf_test_size"
    )

    if st.button("랜덤 포레스트 학습 및 평가하기", type="primary"):

        try:

            X, y, model_cols = prepare_model_data()

            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore"),
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

            with st.spinner("랜덤 포레스트 모델을 학습하고 있습니다..."):
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

            st.success("랜덤 포레스트 학습이 완료되었습니다.")

            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(
                y_test, prediction, zero_division=0
            )
            recall = recall_score(
                y_test, prediction, zero_division=0
            )
            f1 = f1_score(
                y_test, prediction, zero_division=0
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("정확도", f"{accuracy * 100:.2f}%")
            m2.metric("정밀도", f"{precision * 100:.2f}%")
            m3.metric("재현율", f"{recall * 100:.2f}%")
            m4.metric("F1 점수", f"{f1 * 100:.2f}%")

            st.divider()

            st.subheader("혼동 행렬")

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

            st.dataframe(cm_df, use_container_width=True)

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["예측 정상", "예측 고장"],
                y=["실제 정상", "실제 고장"],
                color_continuous_scale="Blues",
                title="랜덤 포레스트 혼동 행렬"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("변수 중요도")

            fitted_preprocessor = model.named_steps["preprocessor"]
            fitted_classifier = model.named_steps["classifier"]

            feature_names = fitted_preprocessor.get_feature_names_out()

            importance_df = pd.DataFrame({
                "변수": feature_names,
                "중요도": fitted_classifier.feature_importances_
            }).sort_values("중요도", ascending=False)

            st.dataframe(
                importance_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                importance_df,
                x="중요도",
                y="변수",
                orientation="h",
                title="랜덤 포레스트 변수 중요도"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "변수 중요도는 모델이 예측할 때 각 변수를 얼마나 "
                "활용했는지를 나타냅니다. 인과관계를 뜻하지는 않습니다."
            )

        except Exception as e:
            st.error(f"모델 실행 중 오류가 발생했습니다: {e}")

    else:
        st.info("버튼을 누르면 랜덤 포레스트 모델을 학습하고 평가합니다.")


# ==========================================
# 하단
# ==========================================
st.divider()
st.caption("기계 고장 예측 분석실")
