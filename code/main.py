result2, median2 = make_failure_rate(
            selected_data, variable2
        )

        # 두 변수의 결과를 각각 표시
        for variable, result, median_value in [
            (variable1, result1, median1),
            (variable2, result2, median2)
        ]:
            st.markdown(f"#### 📈 {variable}에 따른 고장률")

            if result is None or result.empty:
                st.info(
                    "선택한 제품 유형에서는 두 그룹을 나눌 만큼 "
                    "서로 다른 값이 충분하지 않습니다."
                )
                continue

            st.caption(
                f"낮은 그룹과 높은 그룹은 중앙값 "
                f"({median_value:,.2f})을 기준으로 구분했습니다."
            )

            # 고장률 그래프
            chart_data = result.set_index("변수 구분")[["고장률(%)"]]

            fig = px.bar(
                result,
                x="변수 구분",
                y="고장률(%)",
                color="변수 구분",
                text="고장률(%)",
                title=f"{variable} 그룹별 기계 고장률",
                labels={
                    "변수 구분": "변수 그룹",
                    "고장률(%)": "기계 고장률(%)"
                },
                color_discrete_sequence=["#3984E8", "#F28E2B"]
            )
            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )
            fig.update_layout(
                template="plotly_white",
                height=380,
                yaxis_title="기계 고장률(%)",
                xaxis_title="변수 그룹",
                yaxis_range=[
                    0,
                    max(5, float(result["고장률(%)"].max()) * 1.25)
                ],
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # 결과 표
            display_result = result.rename(columns={
                "변수 구분": "변수 그룹",
                "전체_기계_수": "전체 기계 수",
                "고장_기계_수": "고장 기계 수"
            }).copy()

            display_result["고장률(%)"] = (
                display_result["고장률(%)"].round(2)
            )

            st.dataframe(
                display_result,
                use_container_width=True,
                hide_index=True
            )

            # 쉬운 결과 설명
            if len(result) == 2:
                highest = result.loc[
                    result["고장률(%)"].idxmax()
                ]
                lowest = result.loc[
                    result["고장률(%)"].idxmin()
                ]

                difference = (
                    highest["고장률(%)"] - lowest["고장률(%)"]
                )

                st.info(
                    f"{highest['변수 구분']}의 고장률은 "
                    f"{highest['고장률(%)']:.2f}%로, "
                    f"{lowest['변수 구분']}의 "
                    f"{lowest['고장률(%)']:.2f}%보다 "
                    f"{difference:.2f}%p 높게 나타났습니다."
                )

            st.divider()

        st.caption(
            "※ 각 변수는 선택한 제품 유형 안에서 중앙값을 기준으로 "
            "낮은 그룹과 높은 그룹으로 나눕니다. 중앙값과 같은 값은 "
            "낮은 그룹에 포함됩니다. 고장률은 각 그룹의 고장 기계 수를 "
            "전체 기계 수로 나누어 계산합니다. 이 결과는 데이터에서 "
            "나타난 관련성을 보여주며, 고장의 직접적인 원인을 증명하지는 않습니다."
        )

    else:
        st.warning(
            "고장 여부가 0 또는 1로 기록된 데이터와 "
            "분석할 숫자형 변수가 2개 이상 필요합니다."
        )
else:
    st.warning(
        "제품 유형, 기계 고장 여부, 숫자형 변수를 확인할 수 없어 "
        "고장률 분석을 표시하지 못했습니다."
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
    4. 고장 여부와 상관관계가 상대적으로 높은 변수는 무엇일까?
