
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. 한글 글꼴 설정
# =========================================================

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams["font.family"] = font_name
else:
    matplotlib.rcParams["font.family"] = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "sans-serif"
    ]

matplotlib.rcParams["axes.unicode_minus"] = False


# =========================================================
# 2. 기본 화면 설정
# =========================================================

st.set_page_config(
    page_title="기계 고장 예측 분석실",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E4EAF2;
    }

    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jisujisujisu333/Machine-Failure-Prediction-Project/"
    "refs/heads/main/data/"
    "%EA%B8%B0%EA%B3%84%EA%B3%A0%EC%9E%A5%EC%98%88%EC%B8%A1"
    "%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8.csv"
)

RENAME = {
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

NUMERIC = [
    "공기 온도(K)",
    "공정 온도(K)",
    "회전 속도(rpm)",
    "토크(N·m)",
    "공구 마모 시간(분)"
]

TARGET = "기계 고장 여부"

MODEL_FEATURES = ["제품 유형"] + NUMERIC


@st.cache_data
def load_data():
    errors = []

    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            data = pd.read_csv(DATA_URL, encoding=encoding)
            return data.rename(columns=RENAME)
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(" / ".join(errors))


try:
    df = load_data()

except Exception as exc:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(exc))
    st.stop()


for col in NUMERIC + [TARGET]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


required_columns = MODEL_FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 열이 없습니다: {missing_columns}")
    st.stop()


failure_df = df.dropna(subset=[TARGET]).copy()
failure_df[TARGET] = failure_df[TARGET].astype(int)
failure_df = failure_df[
    failure_df[TARGET].isin([0, 1])
].copy()


# =========================================================
# 4. 전처리 함수
# =========================================================

def make_preprocessor(model_kind):

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    if model_kind == "logistic":
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

    return ColumnTransformer([
        ("category", categorical, ["제품 유형"]),
        ("numeric", numeric, NUMERIC)
    ])


def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    metrics = {
        "모델": name,
        "정확도": accuracy_score(y_test, prediction) * 100,
        "정밀도": precision_score(
            y_test, prediction, zero_division=0
        ) * 100,
        "재현율": recall_score(
            y_test, prediction, zero_division=0
        ) * 100,
        "F1 점수": f1_score(
            y_test, prediction, zero_division=0
        ) * 100
    }

    return {
        "model": model,
        "prediction": prediction,
        "metrics": metrics,
        "confusion": cm
    }


def get_display_feature_names(preprocessor):

    names = preprocessor.get_feature_names_out()

    result = []

    for name in names:
        name = name.replace("numeric__", "")
        name = name.replace("category__", "")
        name = name.replace("제품 유형_", "제품 유형: ")
        result.append(name)

    return result


# =========================================================
# 5. 사이드바 메뉴
# =========================================================

st.sidebar.title("⚙️ 메인 목차")

menu = st.sidebar.radio(
    "분석 항목",
    [
        "🏠 홈",
        "1. 문제 정의",
        "2. 핵심 데이터 분석",
        "3. 데이터 전처리",
        "4. 기준선 설정",
        "5. 모델 학습 및 평가"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.subheader("📊 데이터 정보")

st.sidebar.metric("전체 데이터", f"{len(df):,}개")
st.sidebar.metric("전체 변수", f"{len(df.columns)}개")
st.sidebar.metric(
    "결측값",
    f"{int(df.isna().sum().sum()):,}개"
)

st.title("⚙️ 기계 고장 예측 분석실")
st.caption(
    "기계 작동 데이터를 그래프로 분석하고 "
    "머신러닝으로 고장 여부를 예측합니다."
)


# =========================================================
# 홈
# =========================================================

if menu == "🏠 홈":

    st.header("기계 고장 예측 프로젝트")

    st.write(
        "공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모 시간 등의 "
        "데이터를 분석해 기계 고장과 관련된 특징을 찾고, "
        "세 가지 머신러닝 모델로 고장 여부를 예측합니다."
    )

    a, b, c = st.columns(3)

    a.metric("데이터 수", f"{len(df):,}개")
    b.metric("측정 변수", f"{len(NUMERIC)}개")
    c.metric(
        "고장 데이터",
        f"{int((failure_df[TARGET] == 1).sum()):,}개"
    )

    st.subheader("분석 순서")

    st.markdown("""
    1. **문제 정의** — 분석 주제와 목표 확인
    2. **핵심 데이터 분석** — 산포도와 선그래프로 패턴 찾기
    3. **데이터 전처리** — 결측값과 분석 변수 확인
    4. **기준선 설정** — 단순 예측 방법의 성능 확인
    5. **모델 학습 및 평가** — 세 가지 머신러닝 모델 비교
    """)

    st.info("왼쪽 메뉴에서 원하는 분석 항목을 선택하세요.")


# =========================================================
# 1. 문제 정의
# =========================================================

elif menu == "1. 문제 정의":

    st.header("1️⃣ 문제 정의")

    st.subheader("연구 주제")

    st.write(
        "기계 고장과 관련된 요인은 무엇이며, "
        "머신러닝으로 기계 고장을 예측할 수 있을까?"
    )

    st.subheader("주제 선정 이유")

    st.write(
        "산업 현장에서 기계가 갑자기 고장 나면 생산이 중단되고 "
        "수리 비용이 발생할 수 있다. 기계의 작동 데이터를 분석하면 "
        "고장과 관련된 특징을 찾고, 고장을 미리 예측하는 데 "
        "도움을 줄 수 있다고 생각하여 이 주제를 선정하였다."
    )

    st.subheader("데이터셋 소개")

    st.markdown("""
    - 데이터셋: AI4I 2020 기계 고장 예측 데이터
    - 데이터 수: 약 10,000개
    - 주요 변수: 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모 시간
    - 예측 대상: 기계 고장 여부 (0=정상, 1=고장)
    """)

    meanings = pd.DataFrame({
        "변수": [
            "공기 온도(K)",
            "공정 온도(K)",
            "회전 속도(rpm)",
            "토크(N·m)",
            "공구 마모 시간(분)",
            "기계 고장 여부"
        ],
        "뜻": [
            "기계 주변 공기의 온도",
            "기계가 작업하는 과정의 온도",
            "1분 동안 회전하는 횟수",
            "회전 운동을 일으키는 힘의 효과",
            "공구를 사용한 누적 시간",
            "정상(0) 또는 고장(1)"
        ]
    })

    st.subheader("변수의 뜻")

    st.dataframe(
        meanings,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 2. 핵심 데이터 분석
# =========================================================

elif menu == "2. 핵심 데이터 분석":

    st.header("2️⃣ 핵심 데이터 분석")

    st.write(
        "그래프를 통해 기계 고장과 관련된 데이터의 특징을 살펴봅니다."
    )

    normal_count = int((failure_df[TARGET] == 0).sum())
    failure_count = int((failure_df[TARGET] == 1).sum())
    failure_rate = failure_count / len(failure_df) * 100

    x1, x2, x3 = st.columns(3)

    x1.metric("정상 데이터", f"{normal_count:,}개")
    x2.metric("고장 데이터", f"{failure_count:,}개")
    x3.metric("전체 고장률", f"{failure_rate:.2f}%")

    st.divider()

    # 산포도

    st.subheader("그래프 1. 토크와 회전 속도의 관계")

    st.write(
        "점 하나가 기계 데이터 한 건을 뜻합니다. "
        "파란색은 정상, 빨간색은 고장입니다."
    )

    scatter_data = failure_df[
        ["토크(N·m)", "회전 속도(rpm)", TARGET]
    ].dropna().copy()

    if len(scatter_data) > 3500:
        scatter_data = scatter_data.sample(
            3500,
            random_state=42
        )

    scatter_data["상태"] = scatter_data[TARGET].map({
        0: "정상",
        1: "고장"
    })

    fig_scatter = px.scatter(
        scatter_data,
        x="토크(N·m)",
        y="회전 속도(rpm)",
        color="상태",
        color_discrete_map={
            "정상": "#3984E8",
            "고장": "#E45756"
        },
        opacity=0.65,
        title="토크와 회전 속도에 따른 기계 상태"
    )

    fig_scatter.update_layout(
        height=470,
        template="plotly_white",
        xaxis_title="토크(N·m)",
        yaxis_title="회전 속도(rpm)"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    st.caption(
        "고장 데이터가 어느 영역에 분포하는지 확인할 수 있습니다. "
        "점의 분포만으로 직접적인 고장 원인을 확정할 수는 없습니다."
    )

    st.divider()

    # 선그래프

    st.subheader("그래프 2. 공구 마모 시간에 따른 고장률")

    st.write(
        "공구 마모 시간을 구간별로 나누고, "
        "각 구간에서 고장이 발생한 비율을 계산했습니다."
    )

    wear_data = failure_df[
        ["공구 마모 시간(분)", TARGET]
    ].dropna().copy()

    bins = [-0.001, 50, 100, 150, 200, 250, np.inf]

    labels = [
        "0~50분",
        "51~100분",
        "101~150분",
        "151~200분",
        "201~250분",
        "251분 이상"
    ]

    wear_data["사용 시간 구간"] = pd.cut(
        wear_data["공구 마모 시간(분)"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    wear_summary = (
        wear_data
        .groupby("사용 시간 구간", observed=False)
        .agg(
            전체_수=(TARGET, "count"),
            고장_수=(TARGET, "sum")
        )
        .reset_index()
    )

    wear_summary["고장률(%)"] = (
        wear_summary["고장_수"]
        / wear_summary["전체_수"].replace(0, np.nan)
        * 100
    ).fillna(0)

    fig_line = px.line(
        wear_summary,
        x="사용 시간 구간",
        y="고장률(%)",
        markers=True,
        title="공구 사용 시간 구간별 기계 고장률",
        hover_data={
            "전체_수": True,
            "고장_수": True,
            "고장률(%)": ":.2f"
        }
    )

    fig_line.update_traces(
        line_width=3,
        marker_size=10
    )

    fig_line.update_layout(
        height=430,
        template="plotly_white",
        xaxis_title="공구 마모 시간 구간",
        yaxis_title="고장률(%)"
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True
    )

    st.caption(
        "선의 높이는 각 사용 시간 구간의 고장률입니다. "
        "구간별 데이터 수가 다를 수 있으므로 함께 확인해야 합니다."
    )

    with st.expander("구간별 실제 수치 확인"):

        show_wear = wear_summary.rename(columns={
            "사용 시간 구간": "공구 사용 시간",
            "전체_수": "전체 기계 수",
            "고장_수": "고장 기계 수"
        }).copy()

        show_wear["고장률(%)"] = (
            show_wear["고장률(%)"].round(2)
        )

        st.dataframe(
            show_wear,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # 상관관계 그래프

    st.subheader("그래프 3. 고장 여부와 측정 변수의 상관관계")

    st.write(
        "상관계수는 두 변수 사이의 선형적인 관련성을 나타냅니다. "
        "0에 가까울수록 선형 상관관계가 약합니다."
    )

    corr = (
        failure_df[NUMERIC + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
    )

    corr_df = (
        corr.rename_axis("변수")
        .reset_index(name="상관계수")
        .sort_values("상관계수", ascending=True)
    )

    fig_corr = px.bar(
        corr_df,
        x="상관계수",
        y="변수",
        orientation="h",
        color="상관계수",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="기계 고장 여부와 측정 변수의 상관관계"
    )

    fig_corr.update_layout(
        height=420,
        template="plotly_white",
        xaxis_title="상관계수",
        yaxis_title="측정 변수"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    strongest = corr.abs().idxmax()

    st.info(
        f"고장 여부와 절댓값 기준 상관관계가 가장 큰 변수는 "
        f"**{strongest}**입니다. "
        f"(상관계수 {corr[strongest]:.3f})"
    )

    st.caption(
        "상관관계는 관련성을 나타내며, 직접적인 원인이라는 뜻은 아닙니다."
    )


# =========================================================
# 3. 데이터 전처리
# =========================================================

elif menu == "3. 데이터 전처리":

    st.header("3️⃣ 데이터 전처리")

    st.subheader("분석에 사용한 처리")

    st.markdown("""
    - 영어 변수명을 이해하기 쉬운 한국어로 변경했습니다.
    - 측정값과 고장 여부를 숫자형으로 변환했습니다.
    - 제품 유형과 5개 측정 변수를 모델 입력값으로 사용합니다.
    - 제품 번호와 데이터 번호는 식별자이므로 모델에서 제외합니다.
    - TWF, HDF, PWF, OSF, RNF는 고장 유형을 나타내며 정답과 직접 연결된 정보이므로 예측 입력에서 제외합니다.
    - 모델 학습 과정에서 결측값은 중앙값 또는 최빈값으로 처리합니다.
    - 로지스틱 회귀의 수치형 변수는 표준화합니다.
    """)

    st.divider()

    st.subheader("결측값 확인")

    missing = (
        df.isna()
        .sum()
        .rename_axis("변수")
        .reset_index(name="결측값 수")
    )

    missing = missing[
        missing["결측값 수"] > 0
    ]

    if missing.empty:
        st.success("현재 데이터에는 결측값이 없습니다.")
    else:
        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("데이터 구성")

    st.write(f"- 전체 행: {len(df):,}개")
    st.write(f"- 전체 열: {len(df.columns)}개")
    st.write(
        f"- 모델 입력 변수: {', '.join(MODEL_FEATURES)}"
    )

    st.dataframe(
        df[MODEL_FEATURES + [TARGET]].head(8),
        use_container_width=True
    )


# =========================================================
# 4. 기준선 설정
# =========================================================

elif menu == "4. 기준선 설정":

    st.header("4️⃣ 기준선 설정")

    st.write(
        "기준선은 가장 많은 상태만 계속 예측하는 단순한 방법입니다. "
        "머신러닝 모델이 이 기준선보다 나은지 비교하는 출발점으로 사용합니다."
    )

    y = failure_df[TARGET].astype(int)
    X = failure_df[MODEL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    base_acc = accuracy_score(
        y_test,
        baseline_pred
    ) * 100

    normal_n = int((y == 0).sum())
    failure_n = int((y == 1).sum())

    a, b, c = st.columns(3)

    a.metric("정상 데이터", f"{normal_n:,}개")
    b.metric("고장 데이터", f"{failure_n:,}개")
    c.metric("기준선 정확도", f"{base_acc:.2f}%")

    st.bar_chart(
        pd.DataFrame(
            {"데이터 수": [normal_n, failure_n]},
            index=["정상", "고장"]
        )
    )

    st.info(
        f"기준선은 모든 기계를 다수 클래스인 "
        f"{'정상' if normal_n >= failure_n else '고장'}으로 예측합니다. "
        f"기준선 정확도는 {base_acc:.2f}%입니다."
    )


# =========================================================
# 5. 모델 학습 및 평가
# =========================================================

elif menu == "5. 모델 학습 및 평가":

    st.header("5️⃣ 모델 학습 및 평가")

    st.write(
        "같은 학습용·평가용 데이터로 세 모델을 학습하여 비교합니다."
    )

    st.subheader("세 모델의 예측 방식")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**로지스틱 회귀**")
        st.write(
            "입력 변수로 고장 확률을 계산해 "
            "정상 또는 고장으로 분류합니다."
        )

    with col2:
        st.markdown("**랜덤 포레스트**")
        st.write(
            "여러 의사결정트리의 예측 결과를 모아 "
            "최종 결과를 결정합니다."
        )

    with col3:
        st.markdown("**의사결정트리**")
        st.write(
            "질문과 조건을 반복하며 가지를 나누어 "
            "정상 또는 고장으로 분류합니다."
        )

    test_size = st.slider(
        "평가용 데이터 비율",
        min_value=0.2,
        max_value=0.4,
        value=0.3,
        step=0.05,
        key="test_size"
    )

    st.caption(
        f"학습용 {100-test_size*100:.0f}% / "
        f"평가용 {test_size*100:.0f}%"
    )

    if st.button(
        "세 모델 학습 및 비교하기",
        type="primary"
    ):

        X = failure_df[MODEL_FEATURES].copy()
        y = failure_df[TARGET].astype(int)

        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            # 로지스틱 회귀

            logistic = Pipeline([
                ("preprocessor", make_preprocessor("logistic")),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ))
            ])

            # 랜덤 포레스트

            forest = Pipeline([
                ("preprocessor", make_preprocessor("forest")),
                ("classifier", RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            # 의사결정트리

            tree = Pipeline([
                ("preprocessor", make_preprocessor("forest")),
                ("classifier", DecisionTreeClassifier(
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42
                ))
            ])

            with st.spinner(
                "세 모델을 학습하고 평가하고 있습니다..."
            ):

                logistic_result = evaluate_model(
                    logistic,
                    X_train, X_test, y_train, y_test,
                    "로지스틱 회귀"
                )

                forest_result = evaluate_model(
                    forest,
                    X_train, X_test, y_train, y_test,
                    "랜덤 포레스트"
                )

                tree_result = evaluate_model(
                    tree,
                    X_train, X_test, y_train, y_test,
                    "의사결정트리"
                )

            st.session_state["model_results"] = {
                "logistic": logistic_result,
                "forest": forest_result,
                "tree": tree_result,
                "train_count": len(X_train),
                "test_count": len(X_test),
                "test_size": test_size
            }

            st.success(
                "세 모델의 학습과 평가가 완료되었습니다."
            )

        except Exception as exc:
            st.error(
                f"모델 학습 중 문제가 발생했습니다: {exc}"
            )

    results = st.session_state.get("model_results")

    if results:

        st.write(
            f"학습 데이터: {results['train_count']:,}개 · "
            f"평가 데이터: {results['test_count']:,}개"
        )

        # -------------------------------------------------
        # 모델 성능 비교표
        # -------------------------------------------------

        st.subheader("모델 성능 비교")

        metric_df = pd.DataFrame([
            results["logistic"]["metrics"],
            results["forest"]["metrics"],
            results["tree"]["metrics"]
        ])

        st.dataframe(
            metric_df.set_index("모델").round(2),
            use_container_width=True
        )

        # -------------------------------------------------
        # 세 모델 성능 비교 그래프
        # -------------------------------------------------

        st.subheader("세 모델의 성능 비교 그래프")

        plot_metrics = metric_df.melt(
            id_vars="모델",
            var_name="평가 지표",
            value_name="점수(%)"
        )

        fig_metrics = px.bar(
            plot_metrics,
            x="평가 지표",
            y="점수(%)",
            color="모델",
            barmode="group",
            text_auto=".1f",
            title="세 가지 머신러닝 모델의 성능 비교"
        )

        fig_metrics.update_layout(
            height=450,
            template="plotly_white",
            yaxis_title="점수(%)",
            xaxis_title="평가 지표",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig_metrics,
            use_container_width=True
        )

        st.caption(
            "정확도: 전체 예측 중 맞힌 비율 / "
            "정밀도: 고장이라고 예측한 것 중 실제 고장 비율 / "
            "재현율: 실제 고장 중 찾아낸 비율 / "
            "F1: 정밀도와 재현율의 균형"
        )

        # -------------------------------------------------
        # 혼동 행렬
        # -------------------------------------------------

        st.divider()
        st.subheader("혼동 행렬 비교")

        cm_columns = st.columns(3)

        model_list = [
            ("logistic", "로지스틱 회귀"),
            ("forest", "랜덤 포레스트"),
            ("tree", "의사결정트리")
        ]

        for col, (key, title) in zip(
            cm_columns,
            model_list
        ):

            cm = results[key]["confusion"]

            with col:

                st.markdown(f"**{title}**")

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=["예측 정상", "예측 고장"],
                    y=["실제 정상", "실제 고장"],
                    color_continuous_scale="Blues",
                    title=title
                )

                fig_cm.update_layout(height=350)

                st.plotly_chart(
                    fig_cm,
                    use_container_width=True
                )

        # -------------------------------------------------
        # 모델별 예측 과정
        # -------------------------------------------------

        st.divider()
        st.header("모델별 예측 과정 살펴보기")

        st.write(
            "세 모델은 기계 고장을 예측하는 방식이 서로 다릅니다. "
            "각 모델이 데이터를 활용하는 방법을 그래프로 확인합니다."
        )

        logistic_model = results["logistic"]["model"]
        forest_model = results["forest"]["model"]
        tree_model = results["tree"]["model"]

        logistic_preprocessor = logistic_model.named_steps["preprocessor"]
        forest_preprocessor = forest_model.named_steps["preprocessor"]
        tree_preprocessor = tree_model.named_steps["preprocessor"]

        logistic_classifier = logistic_model.named_steps["classifier"]
        forest_classifier = forest_model.named_steps["classifier"]
        tree_classifier = tree_model.named_steps["classifier"]

        # -------------------------------------------------
        # ① 로지스틱 회귀 계수 그래프
        # -------------------------------------------------

        st.subheader("① 로지스틱 회귀 - 변수별 회귀 계수")

        st.write(
            "로지스틱 회귀는 각 변수의 계수를 이용해 "
            "기계가 고장일 확률을 계산합니다."
        )

        logistic_names = get_display_feature_names(
            logistic_preprocessor
        )

        logistic_coefficients = logistic_classifier.coef_[0]

        logistic_df = pd.DataFrame({
            "변수": logistic_names,
            "회귀 계수": logistic_coefficients
        })

        logistic_df["절댓값"] = (
            logistic_df["회귀 계수"].abs()
        )

        logistic_df = logistic_df.sort_values(
            "절댓값",
            ascending=True
        )

        fig_logistic = px.bar(
            logistic_df,
            x="회귀 계수",
            y="변수",
            orientation="h",
            color="회귀 계수",
            color_continuous_scale="RdBu_r",
            title="로지스틱 회귀 변수별 계수"
        )

        fig_logistic.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )

        fig_logistic.update_layout(
            height=480,
            template="plotly_white",
            xaxis_title="회귀 계수",
            yaxis_title="변수"
        )

        st.plotly_chart(
            fig_logistic,
            use_container_width=True
        )

        st.info(
            "양수 계수는 다른 조건이 같을 때 고장으로 예측하는 방향, "
            "음수 계수는 고장 확률을 낮추는 방향으로 작용합니다. "
            "제품 유형은 기준 범주와 비교해 해석해야 합니다."
        )

        with st.expander("로지스틱 회귀 계수 자세히 보기"):

            st.dataframe(
                logistic_df[
                    ["변수", "회귀 계수"]
                ].sort_values(
                    "회귀 계수",
                    ascending=False
                ).round(4),
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # ② 랜덤 포레스트 변수 중요도
        # -------------------------------------------------

        st.divider()
        st.subheader("② 랜덤 포레스트 - 변수 중요도")

        st.write(
            "랜덤 포레스트는 여러 의사결정트리의 결과를 모아 예측합니다. "
            "아래 그래프는 예측 과정에서 각 변수가 얼마나 중요하게 활용되었는지 보여줍니다."
        )

        forest_names = get_display_feature_names(
            forest_preprocessor
        )

        forest_importance_df = pd.DataFrame({
            "변수": forest_names,
            "변수 중요도": forest_classifier.feature_importances_
        }).sort_values(
            "변수 중요도",
            ascending=True
        )

        fig_forest = px.bar(
            forest_importance_df,
            x="변수 중요도",
            y="변수",
            orientation="h",
            color="변수 중요도",
            color_continuous_scale="Blues",
            title="랜덤 포레스트의 변수 중요도"
        )

        fig_forest.update_layout(
            height=480,
            template="plotly_white",
            xaxis_title="변수 중요도",
            yaxis_title="변수"
        )

        st.plotly_chart(
            fig_forest,
            use_container_width=True
        )

        most_important_forest = (
            forest_importance_df.iloc[-1]
        )

        st.info(
            f"랜덤 포레스트에서 가장 높은 중요도를 보인 변수는 "
            f"**{most_important_forest['변수']}**입니다. "
            f"(중요도: {most_important_forest['변수 중요도']:.3f})"
        )

        st.caption(
            "변수 중요도는 예측 과정에서 활용된 정도이며, "
            "실제 고장의 원인이나 영향의 방향을 직접 증명하지는 않습니다."
        )

        # -------------------------------------------------
        # ③ 의사결정트리 판단 과정
        # -------------------------------------------------

        st.divider()
        st.subheader("③ 의사결정트리 - 판단 과정")

        st.write(
            "의사결정트리는 변수의 조건을 차례로 확인하며 "
            "정상 또는 고장을 판단합니다."
        )

        tree_names = get_display_feature_names(
            tree_preprocessor
        )

        fig_tree, ax = plt.subplots(
            figsize=(24, 13),
            dpi=130
        )

        plot_tree(
            tree_classifier,
            feature_names=tree_names,
            class_names=["정상", "고장"],
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=10,
            impurity=True,
            ax=ax
        )

        ax.set_title(
            "의사결정트리의 기계 고장 판단 과정",
            fontsize=20,
            pad=20
        )

        plt.tight_layout()

        st.pyplot(
            fig_tree,
            use_container_width=True
        )

        plt.close(fig_tree)

        st.caption(
            "화면에는 이해하기 쉽도록 최대 3단계까지만 표시합니다. "
            "실제 예측에는 최대 깊이 5로 학습한 모델이 사용됩니다."
        )

        st.info(
            "의사결정트리는 각 조건을 통과하면서 데이터를 나누고, "
            "마지막 단계에서 정상 또는 고장으로 분류합니다."
        )

        # -------------------------------------------------
        # 세 모델 차이점 정리
        # -------------------------------------------------

        st.divider()
        st.subheader("세 모델의 차이점")

        comparison = pd.DataFrame({
            "모델": [
                "로지스틱 회귀",
                "랜덤 포레스트",
                "의사결정트리"
            ],
            "예측 방법": [
                "변수의 계수로 고장 확률 계산",
                "여러 나무의 예측 결과 결합",
                "조건을 따라 단계별 분류"
            ],
            "그래프에서 확인할 내용": [
                "변수별 계수와 방향",
                "변수별 중요도",
                "정상·고장 판단 과정"
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # 결과 해석
        # -------------------------------------------------

        st.divider()
        st.subheader("모델 결과 해석")

        best_f1 = metric_df.loc[
            metric_df["F1 점수"].idxmax()
        ]

        st.info(
            f"현재 평가 결과에서 F1 점수가 가장 높은 모델은 "
            f"**{best_f1['모델']}**이며, 점수는 "
            f"**{best_f1['F1 점수']:.2f}%**입니다."
        )

        st.write(
            "F1 점수는 정밀도와 재현율을 함께 고려한 값입니다. "
            "다만 어떤 모델이 가장 적합한지는 고장을 놓치지 않는 것이 중요한지, "
            "잘못된 고장 경고를 줄이는 것이 중요한지에 따라서도 달라질 수 있습니다."
        )

    else:

        st.info(
            "위의 '세 모델 학습 및 비교하기' 버튼을 누르면 "
            "실제 평가 결과가 표시됩니다."
        )


# =========================================================
# 마지막 안내
# =========================================================

st.divider()

st.caption(
    "주의: 그래프의 상관관계와 변수 중요도는 데이터에서 나타난 관련성을 보여줍니다. "
    "그 자체만으로 기계 고장의 직접적인 원인이라고 단정할 수는 없습니다."
)
