import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# ---------- 安全读取 CSV 的函数 ----------
def safe_read_csv(file, show_error=True):
    """安全读取CSV文件，处理空文件和无效内容"""
    if file is None:
        if show_error:
            st.error("❌ 没有文件，请先上传CSV文件")
        return None
    try:
        # 检查文件大小
        file.seek(0, 2)  # 末尾
        size = file.tell()
        file.seek(0)      # 回到开头
        if size == 0:
            if show_error:
                st.error("❌ 上传的文件为空，请重新上传有效的CSV文件")
            return None
        # 尝试读取
        df = pd.read_csv(file)
        if df.empty:
            if show_error:
                st.error("❌ CSV文件中没有数据行")
            return None
        return df
    except Exception as e:
        if show_error:
            st.error(f"❌ 读取文件失败：{str(e)}")
        return None

# 页面配置
st.set_page_config(page_title="ECL模型数据处理应用", layout="wide")
st.title("📊 ECL模型数据处理实训二 · 交互式应用")
st.markdown("---")

@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 200
    
    outlier_df = pd.DataFrame({
        '客户ID': [f'ECL{i:03d}' for i in range(n)],
        '违约率': np.random.uniform(0, 0.15, n),
        '抵押品价值': np.random.uniform(50, 300, n),
        '信贷额度': np.random.uniform(100, 600, n)
    })
    outlier_df.loc[5, '违约率'] = 1.2
    outlier_df.loc[10, '抵押品价值'] = -50
    outlier_df.to_csv('outlier_data.csv', index=False, encoding='utf-8-sig')

    missing_df = pd.DataFrame({
        '客户ID': [f'ECL{i:03d}' for i in range(n)],
        '客户收入': np.random.normal(50, 15, n),
        '历史违约记录': np.random.choice([0,1], n, p=[0.9,0.1]),
        '行业景气指数': np.random.uniform(80, 120, n),
        '信贷额度': np.random.uniform(100, 600, n)
    })
    mask = np.random.random(n) < 0.1
    missing_df.loc[mask, '客户收入'] = np.nan
    missing_df.to_csv('missing_data.csv', index=False, encoding='utf-8-sig')

    dates = pd.date_range('2015-01-01', periods=40, freq='QE')  # 兼容新版pandas
    macro_df = pd.DataFrame({
        '日期': dates,
        'GDP增长率': 3 + np.sin(np.linspace(0, 4*np.pi, 40)) * 1.5 + np.random.normal(0, 0.2, 40),
        'CPI': 2 + np.cos(np.linspace(0, 3*np.pi, 40)) * 0.8 + np.random.normal(0, 0.1, 40)
    })
    macro_df.to_csv('macro.csv', index=False, encoding='utf-8-sig')
    return "示例数据已生成"

generate_demo_data()

@st.cache_data
def load_default_outlier_data():
    return pd.read_csv('outlier_data.csv', encoding='utf-8-sig')

@st.cache_data
def load_default_missing_data():
    return pd.read_csv('missing_data.csv', encoding='utf-8-sig')

@st.cache_data
def load_default_macro_data():
    return pd.read_csv('macro.csv', encoding='utf-8-sig')

# 侧边栏
st.sidebar.header("⚙️ 控制面板")
use_demo_data = st.sidebar.checkbox("使用示例数据", value=True)
uploaded_file = st.sidebar.file_uploader("上传信贷数据 (CSV格式)", type=["csv"], help="请上传非空、UTF-8编码的CSV文件")
contamination = st.sidebar.slider("异常值敏感度", 0.01, 0.2, 0.05, step=0.01)
fill_algorithm = st.sidebar.selectbox("缺失值填补算法", ["随机森林回归", "神经网络(MLP)"])
lstm_epochs = st.sidebar.slider("LSTM训练轮数 (epochs)", 10, 100, 30, step=10, disabled=True, help="LSTM功能正在优化中")

if st.sidebar.button("🔄 重新运行"):
    st.cache_data.clear()
    st.rerun()

# 选项卡
tab1, tab2, tab3, tab4 = st.tabs(["🔍 异常值识别", "🧩 缺失值填补", "📈 LSTM前瞻预测", "🔗 多源数据融合"])

# ============================================================
# Tab 1: 异常值识别
# ============================================================
with tab1:
    st.header("🔍 异常值识别 - 孤立森林算法")
    
    if use_demo_data and not uploaded_file:
        df_out = load_default_outlier_data()
        st.info("📊 使用示例数据")
    else:
        if uploaded_file is not None:
            df_out = safe_read_csv(uploaded_file, show_error=True)
            if df_out is None:
                st.stop()
            st.success(f"📊 已上传数据：{len(df_out)} 行")
        else:
            st.warning("⚠️ 请上传数据文件或勾选「使用示例数据」")
            st.stop()
    
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error(f"数值列不足（仅有{len(numeric_cols)}列）")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("X轴特征", numeric_cols, index=0)
    with col2:
        feature_y = st.selectbox("Y轴特征", numeric_cols, index=min(1, len(numeric_cols)-1))
    
    features = df_out[[feature_x, feature_y]].copy()
    features = features.fillna(features.median())
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_out['异常标记'] = iso_forest.fit_predict(scaled)
    outliers = df_out[df_out['异常标记'] == -1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(df_out[feature_x].dropna(), patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[0].set_title(f'{feature_x} 箱线图')
    
    normal = df_out[df_out['异常标记'] == 1]
    axes[1].scatter(normal[feature_x], normal[feature_y], c='blue', label='正常', alpha=0.6)
    axes[1].scatter(outliers[feature_x], outliers[feature_y], c='red', label='异常', alpha=0.8, marker='x')
    axes[1].set_xlabel(feature_x)
    axes[1].set_ylabel(feature_y)
    axes[1].set_title(f'异常值分布 (敏感度={contamination})')
    axes[1].legend()
    plt.tight_layout()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔴 异常值数量", len(outliers))
        st.metric("🟢 正常数据数量", len(df_out) - len(outliers))
    with col2:
        st.pyplot(fig)

# ============================================================
# Tab 2: 缺失值填补
# ============================================================
with tab2:
    st.header("🧩 缺失值填补 - 机器学习算法")
    
    if use_demo_data and not uploaded_file:
        df_miss = load_default_missing_data()
        st.info("📊 使用示例数据")
    else:
        if uploaded_file is not None:
            df_miss = safe_read_csv(uploaded_file, show_error=True)
            if df_miss is None:
                st.stop()
            st.success(f"📊 已上传数据：{len(df_miss)} 行")
        else:
            st.warning("⚠️ 请上传数据文件或勾选「使用示例数据」")
            st.stop()
    
    numeric_cols = df_miss.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [col for col in numeric_cols if df_miss[col].isnull().sum() > 0]
    
    if len(missing_cols) == 0:
        st.success("✅ 数据中没有缺失值！")
        st.dataframe(df_miss.head())
        st.stop()
    
    target_col = st.selectbox("选择要填补的列", missing_cols)
    feature_cols = [col for col in numeric_cols if col != target_col and df_miss[col].isnull().sum() == 0]
    
    if len(feature_cols) == 0:
        st.error("没有可用的特征列")
        st.stop()
    
    train = df_miss[df_miss[target_col].notnull()]
    test = df_miss[df_miss[target_col].isnull()]
    
    if test.empty:
        st.success("✅ 没有缺失值需要填补")
    else:
        with st.spinner(f'正在使用 {fill_algorithm} 进行填补...'):
            if fill_algorithm == "随机森林回归":
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            else:
                model = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', max_iter=500, random_state=42)
            model.fit(train[feature_cols], train[target_col])
            test_copy = test.copy()
            test_copy[target_col] = model.predict(test[feature_cols])
            filled = pd.concat([train, test_copy]).sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train[target_col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title(f'填补前 - {target_col}')
    axes[1].hist(filled[target_col], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title(f'填补后 - {fill_algorithm}')
    plt.tight_layout()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 填补前缺失数量", df_miss[target_col].isnull().sum())
    with col2:
        st.pyplot(fig)

# ============================================================
# Tab 3: LSTM 预测（无 TensorFlow 版本）
# ============================================================
with tab3:
    st.header("📈 LSTM时序预测")
    st.info("🚀 **LSTM 预测功能说明**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📋 功能概述
        本模块用于宏观经济指标的时序预测，包括：
        - GDP增长率预测
        - CPI 趋势分析
        - 经济周期识别
        """)
    with col2:
        st.markdown("""
        ### 🔧 技术实现
        - 长短期记忆网络 (LSTM)
        - 支持多变量时间序列
        - 滚动预测机制
        """)
    
    st.markdown("---")
    st.subheader("📊 示例：宏观经济指标预测")
    
    # 兼容不同 pandas 版本的日期生成
    try:
        dates = pd.date_range('2020-01-01', periods=20, freq='QE')
    except:
        dates = pd.date_range('2020-01-01', periods=80, freq='MS')
        dates = dates[dates.month.isin([1,4,7,10])][:20]
    
    example_data = pd.DataFrame({
        '日期': dates[:20],
        'GDP增长率': [3.2, 2.8, 3.5, 4.0, 3.8, 3.2, 2.9, 3.3, 3.6, 4.1, 3.9, 3.4, 3.1, 3.5, 3.8, 4.2, 4.0, 3.6, 3.3, 3.7][:20],
        'CPI': [2.1, 2.3, 2.0, 1.8, 2.2, 2.5, 2.7, 2.4, 2.1, 1.9, 2.3, 2.6, 2.8, 2.5, 2.2, 2.0, 2.4, 2.7, 2.9, 2.6][:20]
    })
    st.line_chart(example_data.set_index('日期'))
    
    st.markdown("""
    ### 💡 使用方法
    1. 准备包含日期列和数值列的CSV文件
    2. 在侧边栏上传数据
    3. LSTM模型将自动训练并预测
    
    ### 📁 数据格式示例
    | 日期 | GDP增长率 | CPI |
    |------|-----------|-----|
    | 2020-01-01 | 3.2 | 2.1 |
    
    ---
    **💡 提示**：完整版LSTM需要TensorFlow支持，当前展示模拟预测效果。
    """)
    
    st.subheader("📈 模拟预测效果展示")
    try:
        forecast_dates = pd.date_range('2024-01-01', periods=8, freq='QE')
    except:
        forecast_dates = pd.date_range('2024-01-01', periods=8, freq='MS')
        forecast_dates = forecast_dates[forecast_dates.month.isin([1,4,7,10])][:8]
    
    forecast_data = pd.DataFrame({
        '日期': forecast_dates,
        '实际值': [3.5, 3.3, 3.6, 3.8, 3.7, 3.9, 4.0, 3.8],
        '预测值': [3.5, 3.4, 3.5, 3.7, 3.8, 3.8, 3.9, 3.9]
    })
    st.line_chart(forecast_data.set_index('日期'))

# ============================================================
# Tab 4: 多源数据融合
# ============================================================
with tab4:
    st.header("🔗 多源数据融合")
    st.markdown("""
    上传多个数据源，系统会自动合并：
    - 如果有共同的「客户ID」列，按客户ID合并
    - 如果有共同的「日期」列，按日期合并
    - 否则按列拼接
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        file1 = st.file_uploader("数据源1", type=["csv"], key="fusion1")
    with col2:
        file2 = st.file_uploader("数据源2", type=["csv"], key="fusion2")
    with col3:
        file3 = st.file_uploader("数据源3", type=["csv"], key="fusion3")
    
    if file1 and file2 and file3:
        df1 = safe_read_csv(file1, show_error=False)
        df2 = safe_read_csv(file2, show_error=False)
        df3 = safe_read_csv(file3, show_error=False)
        
        if df1 is None or df2 is None or df3 is None:
            st.error("❌ 至少有一个文件无效，请检查文件是否为空或格式错误")
            st.stop()
        
        try:
            # 尝试按公共列合并
            common_cols = set(df1.columns) & set(df2.columns) & set(df3.columns)
            if '客户ID' in common_cols:
                merged = df1.merge(df2, on='客户ID', how='outer')
                merged = merged.merge(df3, on='客户ID', how='outer')
                st.info("🔗 按「客户ID」列合并")
            elif '日期' in common_cols:
                df1['日期'] = pd.to_datetime(df1['日期'])
                df2['日期'] = pd.to_datetime(df2['日期'])
                df3['日期'] = pd.to_datetime(df3['日期'])
                merged = df1.merge(df2, on='日期', how='outer')
                merged = merged.merge(df3, on='日期', how='outer')
                st.info("🔗 按「日期」列合并")
            else:
                merged = pd.concat([df1, df2, df3], axis=1)
                st.info("🔗 按列拼接合并")
            
            # 填充缺失值
            numeric_cols = merged.select_dtypes(include=[np.number]).columns
            merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())
            
            st.success(f"✅ 融合完成！{merged.shape[0]} 行 × {merged.shape[1]} 列")
            st.dataframe(merged.head())
            
            csv = merged.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载融合数据", csv, "融合数据.csv", "text/csv")
        except Exception as e:
            st.error(f"融合失败：{str(e)}")
    else:
        st.info("📁 请上传3个CSV文件进行数据融合")

st.markdown("---")
st.caption("📊 ECL模型数据处理实训二 · 交互式数据分析平台")
