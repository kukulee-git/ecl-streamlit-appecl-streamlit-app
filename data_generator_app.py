import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64

# 页面配置
st.set_page_config(
    page_title="金融信贷数据生成器",
    page_icon="📊",
    layout="wide"
)

# 标题
st.title("📊 金融信贷数据生成器")
st.markdown("专门为 ECL 模型设计的数据生成工具")
st.markdown("---")

# 侧边栏设置
st.sidebar.header("⚙️ 生成参数设置")
st.sidebar.markdown("调整参数来控制生成数据的规模和特征")

# 全局参数
n_customers = st.sidebar.slider("客户数量", 100, 5000, 500, step=100)
random_seed = st.sidebar.number_input("随机种子", 1, 999, 42, step=1)
missing_rate = st.sidebar.slider("缺失值比例 (%)", 0, 30, 10, step=5)

np.random.seed(random_seed)

# 功能选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏦 信贷客户数据", 
    "🔍 缺失值演示数据", 
    "📈 宏观经济数据", 
    "🌐 外部舆情数据",
    "⏰ 时间序列数据"
])

# ============================================================
# Tab 1: 信贷客户数据
# ============================================================
with tab1:
    st.header("🏦 信贷客户数据")
    st.markdown("用于**异常值检测**和**信用风险评估**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 数据特征")
        st.markdown("""
        - **客户ID**: 唯一标识
        - **年龄**: 22-65岁
        - **年收入**: 5-200万元
        - **信用评分**: 300-850分
        - **违约概率**: 0.1%-30%
        - **贷款金额**: 5-500万元
        - **抵押品价值**: 贷款金额的0.5-2倍
        - **已还款比例**: 0-100%
        - **逾期次数**: 0-5次
        - **负债收入比**: 0-80%
        """)
    
    with col2:
        st.subheader("⚠️ 异常值类型")
        st.markdown("""
        - 🔴 违约概率异常高 (>30%)
        - 🔴 抵押品价值为负数
        - 🔴 贷款金额远超收入
        - 🔴 年龄异常 (>100岁)
        - 🔴 逾期次数过高 (>10次)
        - 🔴 负债收入比 >1
        - 🔴 信用评分过低 (<300)
        """)
    
    if st.button("📊 生成信贷客户数据", key="btn_customer"):
        with st.spinner("正在生成数据..."):
            # 生成客户数据
            customer_data = pd.DataFrame({
                '客户ID': [f'CUST{str(i).zfill(6)}' for i in range(1, n_customers + 1)],
                '年龄': np.random.choice(range(22, 66), n_customers),
                '年收入_万': np.random.gamma(2, 15, n_customers).round(1),
                '信用评分': np.random.normal(650, 80, n_customers).clip(300, 850).astype(int),
                '违约概率': (1 - (np.random.normal(650, 80, n_customers).clip(300, 850) - 300) / 550 * 0.25 
                            + np.random.normal(0, 0.03, n_customers)).clip(0.001, 0.3).round(4),
                '贷款金额_万': np.random.gamma(2, 40, n_customers).clip(5, 500).round(1),
                '抵押品价值_万': (np.random.uniform(0.5, 2, n_customers) * 
                                np.random.gamma(2, 40, n_customers).clip(5, 500)).round(1),
                '已还款比例': np.random.beta(2, 5, n_customers).round(3),
                '逾期次数': np.random.poisson(0.5, n_customers).clip(0, 5),
                '负债收入比': np.random.beta(2, 8, n_customers).round(3)
            })
            
            # 添加异常值
            anomaly_indices = [12, 45, 78, 89, 123, 156, 234, 345]
            for idx in anomaly_indices:
                if idx < len(customer_data):
                    if idx == 12:
                        customer_data.loc[idx, '违约概率'] = 0.45
                        customer_data.loc[idx, '信用评分'] = 380
                    elif idx == 45:
                        customer_data.loc[idx, '抵押品价值_万'] = -10
                    elif idx == 78:
                        customer_data.loc[idx, '贷款金额_万'] = 800
                        customer_data.loc[idx, '年收入_万'] = 15
                    elif idx == 89:
                        customer_data.loc[idx, '年龄'] = 180
                    elif idx == 123:
                        customer_data.loc[idx, '逾期次数'] = 15
                    elif idx == 156:
                        customer_data.loc[idx, '负债收入比'] = 1.2
                    elif idx == 234:
                        customer_data.loc[idx, '信用评分'] = 250
                    elif idx == 345:
                        customer_data.loc[idx, '贷款金额_万'] = -50
            
            # 显示预览
            st.success(f"✅ 成功生成 {len(customer_data)} 条客户数据")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均年龄", f"{customer_data['年龄'].mean():.0f}岁")
            with col2:
                st.metric("平均年收入", f"{customer_data['年收入_万'].mean():.1f}万")
            with col3:
                st.metric("平均信用评分", f"{customer_data['信用评分'].mean():.0f}")
            with col4:
                st.metric("平均违约概率", f"{customer_data['违约概率'].mean()*100:.1f}%")
            
            st.subheader("📊 数据预览（前10行）")
            st.dataframe(customer_data.head(10))
            
            # 下载按钮
            csv = customer_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载信贷客户数据 (CSV)",
                data=csv,
                file_name=f"信贷客户数据_{n_customers}条.csv",
                mime="text/csv"
            )

# ============================================================
# Tab 2: 缺失值演示数据
# ============================================================
with tab2:
    st.header("🔍 缺失值演示数据")
    st.markdown("用于测试**缺失值填补算法**的效果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 数据特征")
        st.markdown("""
        - 基于信贷客户数据
        - 在关键字段引入缺失值
        - 缺失率可调节 (0-30%)
        - 可用于对比填补算法效果
        """)
    
    with col2:
        st.subheader("🔧 缺失值字段")
        st.markdown("""
        - 年收入_万
        - 信用评分
        - 负债收入比
        - 抵押品价值_万
        """)
    
    if st.button("📊 生成缺失值演示数据", key="btn_missing"):
        with st.spinner(f"正在生成数据（缺失率{missing_rate}%）..."):
            # 先生成基础数据
            base_data = pd.DataFrame({
                '客户ID': [f'CUST{str(i).zfill(6)}' for i in range(1, n_customers + 1)],
                '年龄': np.random.choice(range(22, 66), n_customers),
                '年收入_万': np.random.gamma(2, 15, n_customers).round(1),
                '信用评分': np.random.normal(650, 80, n_customers).clip(300, 850).astype(int),
                '历史违约记录': np.random.choice([0, 1], n_customers, p=[0.9, 0.1]),
                '行业景气指数': np.random.uniform(80, 120, n_customers).round(1),
                '负债收入比': np.random.beta(2, 8, n_customers).round(3),
                '抵押品价值_万': np.random.gamma(2, 40, n_customers).clip(5, 500).round(1)
            })
            
            # 引入缺失值
            missing_cols = ['年收入_万', '信用评分', '负债收入比', '抵押品价值_万']
            for col in missing_cols:
                mask = np.random.random(n_customers) < (missing_rate / 100)
                base_data.loc[mask, col] = np.nan
            
            st.success(f"✅ 成功生成 {len(base_data)} 条数据，缺失率约 {missing_rate}%")
            
            # 统计缺失情况
            st.subheader("📊 缺失值统计")
            missing_stats = pd.DataFrame({
                '列名': missing_cols,
                '缺失数量': [base_data[col].isnull().sum() for col in missing_cols],
                '缺失比例': [f"{base_data[col].isnull().sum()/len(base_data)*100:.1f}%" for col in missing_cols]
            })
            st.dataframe(missing_stats)
            
            st.subheader("📊 数据预览（前10行）")
            st.dataframe(base_data.head(10))
            
            # 下载按钮
            csv = base_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载缺失值演示数据 (CSV)",
                data=csv,
                file_name=f"缺失值演示数据_缺失率{missing_rate}%.csv",
                mime="text/csv"
            )

# ============================================================
# Tab 3: 宏观经济数据
# ============================================================
with tab3:
    st.header("📈 宏观经济季度数据")
    st.markdown("用于**LSTM时序预测**和**经济周期分析**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quarters = st.slider("季度数量", 20, 200, 100, step=10)
        start_year = st.number_input("起始年份", 2000, 2020, 2000, step=1)
    
    with col2:
        st.subheader("📋 数据指标")
        st.markdown("""
        - GDP增长率 (%)
        - CPI (%)
        - 失业率 (%)
        - 基准利率 (%)
        - 房地产价格指数
        """)
    
    if st.button("📊 生成宏观经济数据", key="btn_macro"):
        with st.spinner("正在生成宏观经济数据..."):
            start_date = datetime(start_year, 1, 1)
            dates = [start_date + timedelta(days=90*i) for i in range(quarters)]
            t = np.arange(quarters)
            
            # 生成经济指标
            gdp_growth = (4.5 + 2 * np.sin(t * 2 * np.pi / 20) + 
                         0.5 * np.sin(t * 2 * np.pi / 8) + 
                         np.random.normal(0, 0.3, quarters)).clip(1, 10).round(2)
            
            cpi = (2.5 + 0.8 * np.sin(t * 2 * np.pi / 12) +
                   np.random.normal(0, 0.2, quarters)).clip(0.5, 5).round(2)
            
            unemployment = (5 + 1.5 * np.sin(t * 2 * np.pi / 24) +
                           np.random.normal(0, 0.3, quarters)).clip(2.5, 9).round(1)
            
            interest_rate = (3 + 1.2 * np.sin(t * 2 * np.pi / 20) +
                            np.random.normal(0, 0.2, quarters)).clip(0, 6).round(2)
            
            house_price = (100 + t * 0.8 + 10 * np.sin(t * 2 * np.pi / 30) +
                          np.random.normal(0, 3, quarters)).clip(80, 180).round(1)
            
            macro_data = pd.DataFrame({
                '日期': dates,
                'GDP增长率_%': gdp_growth,
                'CPI_%': cpi,
                '失业率_%': unemployment,
                '基准利率_%': interest_rate,
                '房地产价格指数': house_price
            })
            
            # 添加金融危机影响
            crisis_years = [2008, 2009]
            for year in crisis_years:
                crisis_idx = [i for i, d in enumerate(dates) if d.year == year]
                for idx in crisis_idx:
                    macro_data.loc[idx, 'GDP增长率_%'] = max(0, macro_data.loc[idx, 'GDP增长率_%'] - 3)
                    macro_data.loc[idx, '失业率_%'] = macro_data.loc[idx, '失业率_%'] + 1.5
            
            # 添加疫情冲击
            covid_idx = [i for i, d in enumerate(dates) if d.year == 2020]
            for idx in covid_idx:
                macro_data.loc[idx, 'GDP增长率_%'] = max(0, macro_data.loc[idx, 'GDP增长率_%'] - 2)
            
            st.success(f"✅ 成功生成 {quarters} 个季度数据（{start_year}年 - {dates[-1].year}年）")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均GDP增长", f"{macro_data['GDP增长率_%'].mean():.1f}%")
            with col2:
                st.metric("平均CPI", f"{macro_data['CPI_%'].mean():.1f}%")
            with col3:
                st.metric("平均失业率", f"{macro_data['失业率_%'].mean():.1f}%")
            
            st.subheader("📈 数据趋势图")
            st.line_chart(macro_data.set_index('日期')[['GDP增长率_%', 'CPI_%']])
            
            st.subheader("📊 数据预览（前10行）")
            st.dataframe(macro_data.head(10))
            
            # 下载按钮
            csv = macro_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载宏观经济数据 (CSV)",
                data=csv,
                file_name=f"宏观经济数据_{quarters}季度.csv",
                mime="text/csv"
            )

# ============================================================
# Tab 4: 外部舆情数据
# ============================================================
with tab4:
    st.header("🌐 外部舆情数据")
    st.markdown("用于**多源数据融合**和**舆情分析**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        industries = ['制造业', '服务业', '科技业', '零售业', '建筑业', '金融业', '医疗业', '教育业']
        selected_industries = st.multiselect("选择行业", industries, default=industries[:4])
    
    with col2:
        st.subheader("📋 数据指标")
        st.markdown("""
        - 所属行业
        - 行业景气指数
        - 新闻情感评分 (-1 到 1)
        - 舆情热度 (0-100)
        - ESG评分 (0-100)
        - 监管关注度 (0-10)
        """)
    
    if st.button("📊 生成外部舆情数据", key="btn_external"):
        with st.spinner("正在生成舆情数据..."):
            if not selected_industries:
                selected_industries = industries
            
            industry_weights = [1/len(selected_industries)] * len(selected_industries)
            
            external_data = pd.DataFrame({
                '客户ID': [f'CUST{str(i).zfill(6)}' for i in range(1, n_customers + 1)],
                '所属行业': np.random.choice(selected_industries, n_customers, p=industry_weights),
                '行业景气指数': np.random.normal(100, 15, n_customers).clip(50, 150).astype(int),
                '新闻情感评分': np.random.normal(0.2, 0.4, n_customers).clip(-0.8, 0.9).round(3),
                '舆情热度': np.random.gamma(2, 15, n_customers).clip(0, 100).astype(int),
                'ESG评分': np.random.normal(65, 12, n_customers).clip(30, 95).astype(int),
                '监管关注度': np.random.poisson(2, n_customers).clip(0, 10)
            })
            
            st.success(f"✅ 成功生成 {len(external_data)} 条舆情数据")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均行业景气", f"{external_data['行业景气指数'].mean():.0f}")
            with col2:
                st.metric("平均情感评分", f"{external_data['新闻情感评分'].mean():.2f}")
            with col3:
                st.metric("平均舆情热度", f"{external_data['舆情热度'].mean():.0f}")
            with col4:
                st.metric("平均ESG评分", f"{external_data['ESG评分'].mean():.0f}")
            
            st.subheader("📊 行业分布")
            industry_dist = external_data['所属行业'].value_counts()
            st.bar_chart(industry_dist)
            
            st.subheader("📊 数据预览（前10行）")
            st.dataframe(external_data.head(10))
            
            # 下载按钮
            csv = external_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载外部舆情数据 (CSV)",
                data=csv,
                file_name=f"外部舆情数据_{n_customers}条.csv",
                mime="text/csv"
            )

# ============================================================
# Tab 5: 时间序列数据
# ============================================================
with tab5:
    st.header("⏰ 时间序列客户数据")
    st.markdown("用于**客户行为预测**和**时序分析**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_customers_ts = st.slider("客户数量（时间序列）", 20, 500, 100, step=10)
        months = st.slider("时间跨度（月）", 12, 60, 24, step=6)
    
    with col2:
        st.subheader("📋 数据特征")
        st.markdown("""
        - 每个客户24-60个月的历史数据
        - 月收入/支出变化
        - 违约概率动态变化
        - 适合LSTM时序预测
        """)
    
    if st.button("📊 生成时间序列数据", key="btn_timeseries"):
        with st.spinner(f"正在生成 {n_customers_ts} 个客户 × {months} 个月的数据..."):
            start_date = datetime(2022, 1, 1)
            dates_ts = [start_date + timedelta(days=30*i) for i in range(months)]
            
            ts_data_list = []
            for cust_id in range(1, n_customers_ts + 1):
                base_risk = np.random.uniform(0.01, 0.15)
                base_income = np.random.uniform(3, 30)
                
                for month, date in enumerate(dates_ts):
                    # 收入有季节性波动
                    income = base_income * (1 + 0.1 * np.sin(month * 2 * np.pi / 12))
                    income = max(1, income + np.random.normal(0, 1))
                    
                    # 支出与收入相关
                    expense = income * np.random.uniform(0.3, 0.8)
                    
                    # 违约概率随时间变化
                    pd = base_risk + 0.02 * np.sin(month * 2 * np.pi / 12) + np.random.normal(0, 0.005)
                    pd = max(0.001, min(0.3, pd))
                    
                    ts_data_list.append({
                        '客户ID': f'TS_CUST{str(cust_id).zfill(4)}',
                        '日期': date,
                        '月收入_万': round(income, 2),
                        '月支出_万': round(expense, 2),
                        '违约概率': round(pd, 4),
                        '是否违约': 1 if np.random.random() < pd else 0
                    })
            
            ts_data = pd.DataFrame(ts_data_list)
            
            total_records = len(ts_data)
            st.success(f"✅ 成功生成 {total_records} 条时序数据（{n_customers_ts}客户 × {months}月）")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总记录数", f"{total_records:,}")
            with col2:
                st.metric("平均月收入", f"{ts_data['月收入_万'].mean():.2f}万")
            with col3:
                st.metric("平均违约率", f"{ts_data['违约概率'].mean()*100:.1f}%")
            
            # 展示一个客户的时序数据
            sample_cust = ts_data['客户ID'].unique()[0]
            sample_data = ts_data[ts_data['客户ID'] == sample_cust]
            
            st.subheader(f"📈 示例客户 {sample_cust} 的时序数据")
            st.line_chart(sample_data.set_index('日期')['月收入_万'])
            
            st.subheader("📊 数据预览（前10行）")
            st.dataframe(ts_data.head(10))
            
            # 下载按钮
            csv = ts_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载时间序列数据 (CSV)",
                data=csv,
                file_name=f"时间序列数据_{n_customers_ts}客户_{months}月.csv",
                mime="text/csv"
            )

# 页脚
st.markdown("---")
st.markdown("""
### 📖 使用说明

1. **左侧边栏**：调整数据生成参数（数量、随机种子、缺失率等）
2. **选择选项卡**：选择要生成的数据类型
3. **点击生成按钮**：系统会自动生成符合金融业务逻辑的合理数据
4. **下载数据**：生成后可以一键下载CSV文件

### 💡 数据用途

| 数据类型 | 主要用途 | 配合模块 |
|---------|---------|---------|
| 信贷客户数据 | 异常值检测 | Tab 1 |
| 缺失值演示数据 | 缺失值填补 | Tab 2 |
| 宏观经济数据 | LSTM时序预测 | Tab 3 |
| 外部舆情数据 | 多源数据融合 | Tab 4 |
| 时间序列数据 | 客户行为预测 | Tab 3 |

### ⚠️ 注意事项

- 所有数据均为模拟生成，仅用于学习和测试
- 数据特征基于真实金融业务逻辑设计
- 缺失值比例、客户数量等参数可根据需要调整
""")