st.write("아래에서 실제 CSV의 열 이름과 일부 데이터를 확인할 수 있습니다.")
    st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.subheader("1. 📊 수치형 변수의 기초통계")
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    stats = df[numeric_cols].describe().T
    stats = stats.rename(columns={
        "count": "자료 수", "mean": "평균", "std": "표준편차",
        "min": "최솟값", "25%": "1사분위수", "50%": "중앙값",
        "75%": "3사분위수", "max": "최댓값"
    }).round(2)
    st.dataframe(stats, use_container_width=True)
    selected_stat_col = st.selectbox("분포를 확인할 수치형 변수 선택", numeric_cols)
    fig_hist = px.histogram(
        df, x=selected_stat_col, nbins=30,
        title=f"{selected_stat_col}의 분포",
        labels={selected_stat_col: selected_stat_col, "count": "자료 수"},
        color_discrete_sequence=["#3984E8"]
    )
    fig_hist.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning("수치형 변수가 없어 기초통계표와 히스토그램을 만들 수 없습니다.")

st.divider()
st.subheader("2. 🧩 범주형 변수의 분포")
categorical_cols = [
    c for c in df.columns
    if c not in numeric_cols and df[c].nunique(dropna=True) <= 30
]
if categorical_cols:
    cat_col = st.selectbox("분포를 확인할 범주형 변수 선택", categorical_cols)
    cat_counts = (
        df[cat_col].fillna("결측값").astype(str)
        .value_counts().rename_axis(cat_col).reset_index(name="개수")
    )
    fig_cat = px.bar(
        cat_counts, x=cat_col, y="개수", color=cat_col,
        title=f"{cat_col}별 자료 수",
        text="개수"
    )
    fig_cat.update_layout(template="plotly_white", height=380, showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("범주형 변수가 없거나 범주 수가 많아 표시할 변수가 없습니다.")

st.divider()
st.subheader("3. 🔧 고장 여부 살펴보기")
if target_col:
    st.write(f"**고장 여부 변수:** `{target_col}`")
    target_values = df[target_col].fillna("결측값").astype(str).value_counts().rename_axis("고장 여부").reset_index(name="개수")
    c1, c2 = st.columns(2)
    with c1:
        fig_target = px.bar(
            target_values, x="고장 여부", y="개수", color="고장 여부",
            title="고장 여부별 데이터 수", text="개수"
        )
        fig_target.update_layout(template="plotly_white", showlegend=False, height=360)
        st.plotly_chart(fig_target, use_container_width=True)
    with c2:
        fig_pie = px.pie(
            target_values, names="고장 여부", values="개수",
            title="고장 여부 비율", hole=0.4
        )
        fig_pie.update_layout(height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 고장 여부에 따른 수치형 변수 평균 비교")
    compare_cols = [c for c in numeric_cols if c != target_col]
    if compare_cols:
        compare_col = st.selectbox("평균을 비교할 변수 선택", compare_cols)
        compare_df = df.groupby(target_col, dropna=False)[compare_col].mean().reset_index()
        compare_df[target_col] = compare_df[target_col].astype(str)
        fig_compare = px.bar(
            compare_df, x=target_col, y=compare_col, color=target_col,
            title=f"고장 여부에 따른 {compare_col} 평균",
            labels={target_col: "고장 여부", compare_col: f"{compare_col} 평균"},
            text_auto=".2f"
        )
        fig_compare.update_layout(template="plotly_white", showlegend=False, height=380)
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("고장 여부 외에 비교할 수치형 변수가 없습니다.")
else:
    st.warning("고장 여부 열을 자동으로 찾지 못했습니다. 아래 데이터 열 이름을 확인하고, 코드의 target_candidates 목록에 실제 열 이름을 추가해 주세요.")
    st.write("현재 열 이름:", ", ".join(map(str, df.columns)))

st.divider()
st.subheader("4. 🔗 수치형 변수 사이의 상관관계")
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="상관계수 히트맵"
    )
    fig_corr.update_layout(height=max(450, 35 * len(numeric_cols)), template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("상관계수는 두 변수의 선형적인 관련성을 나타냅니다. 상관관계만으로 원인과 결과를 단정할 수는 없습니다.")
else:
    st.info("상관관계를 계산하려면 수치형 변수가 2개 이상 필요합니다.")

st.divider()
st.subheader("5. 🧹 결측값 확인")
missing = df.isna().sum().rename_axis("변수").reset_index(name="결측값 수")
missing = missing[missing["결측값 수"] > 0].sort_values("결측값 수", ascending=False)
if missing.empty:
    st.success("결측값이 없습니다.")
else:
    missing["결측 비율(%)"] = (missing["결측값 수"] / len(df) * 100).round(2)
    st.dataframe(missing, use_container_width=True)
    fig_missing = px.bar(
        missing, x="변수", y="결측값 수",
        title="변수별 결측값 수", text="결측값 수",
        color="결측값 수", color_continuous_scale="Blues"
    )
    fig_missing.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig_missing, use_container_width=True)

st.divider()
st.caption("분석 안내: 이 앱은 기초통계와 탐색적 분석을 위한 도구입니다. 관찰된 차이나 상관관계가 곧바로 고장의 직접적인 원인을 의미하지는 않습니다.")
