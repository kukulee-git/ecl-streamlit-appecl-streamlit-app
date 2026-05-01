import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import io
import os
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(page_title="ECL模型数据处理应用", layout="wide")
st.title("📊 ECL模型数据处理实训二 · 交互式应用")
st.markdown("---")

# 检查文件是否为空或无效的函数
def is_valid_file(file):
    if file is None:
        return False
    try:
        # 检查文件大小
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size == 0:
            return False
        return True
    except:
        return False

# 安全读取CSV的函数
def safe_read_csv(file):
    try:
        if file is None:
            return None
        # 重置文件指针
        file.seek(0)
        # 读取文件内容
        content = file.read()
        if len(content) == 0:
            st.error("文件为空，请上传有效的CSV文件")
            return None
        # 尝试读取CSV
        df = pd.read_csv(io.BytesIO(content))
        if df.empty or len(df.columns) == 0:
            st.error("CSV文件没有有效的数据列")
            return None
        return df
    except Exception as e:
        st.error(f"读取文件失败：{str(e)}")
        return None

@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 200
    
    # 异常值数据
    outlier_df = pd.DataFrame({
        '客户ID': [f'ECL{i:03d}' for i in range(n)],
        '违约率': np.random.uniform(0, 0.15, n),
        '抵押品价值': np.random.uniform(50, 300, n),
        '信贷额度': np.random.uniform(100, 600, n)
    })
    outlier_df.loc[5, '违约率'] = 1.2
    outlier_df.loc[10, '抵押品价值'] = -50
    outlier_df.to_csv('outlier_data.csv', index=False, encoding='utf-8-sig')

    # 缺失值数据
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

    # 宏观数据
    dates = pd.date_range('2015-01-01', periods=40, freq='Q')
    macro_df = pd.DataFrame({
        '日期': dates,
        'GDP增长率': 3 + np.sin(np.linspace(0, 4*np.pi, 40)) * 1.5 + np.random.normal(0, 0.2, 40),
        'CPI': 2 + np.cos(np.linspace(0, 3*np.pi, 40)) * 0.8 + np.random.normal(0, 0.1, 40)
    })
    macro_df.to_csv('macro.csv', index=False, encoding='utf-8-sig')
    return "示例数据已生成"

# 生成示例数据（如果不存在）
if not os.path.exists('outlier_data.csv'):
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

# 侧边栏：控制面板
st.sidebar.header("⚙️ 控制面板")
use_demo_data = st.sidebar.checkbox("使用示例数据", value=True)

# 文件上传区域
st.sidebar.markdown("---")
st.sidebar.subheader("📁 上传你的数据")
uploaded_file = st.sidebar.file_uploader(
    "上传信贷数据 (CSV格式)", 
    type=["csv"], 
    help="支持UTF-8编码的CSV文件，文件不能为空"
)

# 参数调节
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 参数设置")
contamination = st.sidebar.slider("异常值敏感度", 0.01, 0.2, 0.05, step=0.01, 
                                   help="数值越大，标记为异常的点越多")
fill_algorithm = st.sidebar.selectbox("缺失值填补算法", ["随机森林回归", "神经网络(MLP)"])
lstm_epochs = st.sidebar.slider("LSTM训练轮数", 10, 100, 30, step=10)

if st.sidebar.button("🔄 重新运行"):
    st.cache_data.clear()
    st.rerun()

# 显示数据状态
st.sidebar.markdown("---")
if use_demo_data:
    st.sidebar.success("✅ 当前使用：示例数据")
else:
    if uploaded_file is not None and is_valid_file(uploaded_file):
        st.sidebar.success("✅ 当前使用：上传的数据")
    else:
        st.sidebar.warning("⚠️ 请上传数据文件")

# 四个选项卡
tab1, tab2, tab3, tab4 = st.tabs(["🔍 异常值识别", "🧩 缺失值填补", "📈 LSTM前瞻预测", "🔗 多源数据融合"])

# ============================================================
# Tab 1: 异常值识别
# ============================================================
with tab1:
    st.header("🔍 异常值识别 - 孤立森林算法")
    
    # 加载数据
    if use_demo_data:
        df_out = load_default_outlier_data()
        st.info("📊 使用示例数据（包含人工构造的异常值）")
    else:
        if uploaded_file is not None and is_valid_file(uploaded_file):
            df_out = safe_read_csv(uploaded_file)
            if df_out is None:
                st.stop()
            st.success(f"📊 已上传数据：{len(df_out)} 行 × {len(df_out.columns)} 列")
        else:
            st.warning("⚠️ 请上传数据文件或勾选「使用示例数据」")
            st.stop()
    
    # 找出数值列
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error(f"❌ 数据中数值列不足（仅有{len(numeric_cols)}列），无法进行异常值检测")
        st.info("💡 请确保数据包含至少2列数值数据（如：违约率、抵押品价值、贷款金额等）")
        st.stop()
    
    # 选择特征列
    st.subheader("📈 异常值检测结果")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("X轴特征", numeric_cols, index=0)
    with col2:
        feature_y = st.selectbox("Y轴特征", numeric_cols, index=min(1, len(numeric_cols)-1))
    
    # 执行异常值检测
    features = df_out[[feature_x, feature_y]].copy()
    
    # 处理缺失值
    if features.isnull().sum().sum() > 0:
        st.warning(f"⚠️ 数据中存在{features.isnull().sum().sum()}个缺失值，将用中位数填充")
        features = features.fillna(features.median())
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_out['异常标记'] = iso_forest.fit_predict(scaled)
    outliers = df_out[df_out['异常标记'] == -1]
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 箱线图
    axes[0].boxplot(df_out[feature_x].dropna(), patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[0].set_title(f'{feature_x} 箱线图')
    axes[0].set_ylabel(feature_x)
    
    # 散点图
    normal = df_out[df_out['异常标记'] == 1]
    axes[1].scatter(normal[feature_x], normal[feature_y], c='blue', label='正常', alpha=0.6, s=30)
    axes[1].scatter(outliers[feature_x], outliers[feature_y], c='red', label='异常', alpha=0.8, s=50, marker='x')
    axes[1].set_xlabel(feature_x)
    axes[1].set_ylabel(feature_y)
    axes[1].set_title(f'异常值分布图 (敏感度={contamination})')
    axes[1].legend()
    plt.tight_layout()
    
    # 显示结果
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔴 异常值数量", len(outliers))
        st.metric("🟢 正常数据数量", len(df_out) - len(outliers))
        st.metric("📊 异常值占比", f"{len(outliers)/len(df_out)*100:.1f}%")
    with col2:
        st.pyplot(fig)
    
    # 显示异常值列表
    if len(outliers) > 0:
        with st.expander(f"📋 查看{len(outliers)}个异常值详情"):
            st.dataframe(outliers.head(20))
    
    st.markdown("💡 **提示**：增大「异常值敏感度」会标记更多异常点，减小则更宽松。")

# ============================================================
# Tab 2: 缺失值填补
# ============================================================
with tab2:
    st.header("🧩 缺失值填补 - 机器学习算法")
    
    # 加载数据
    if use_demo_data:
        df_miss = load_default_missing_data()
        st.info("📊 使用示例数据（包含约10%的缺失值）")
    else:
        if uploaded_file is not None and is_valid_file(uploaded_file):
            df_miss = safe_read_csv(uploaded_file)
            if df_miss is None:
                st.stop()
            st.success(f"📊 已上传数据：{len(df_miss)} 行 × {len(df_miss.columns)} 列")
        else:
            st.warning("⚠️ 请上传数据文件或勾选「使用示例数据」")
            st.stop()
    
    # 找出有缺失值的数值列
    numeric_cols = df_miss.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [col for col in numeric_cols if df_miss[col].isnull().sum() > 0]
    
    if len(missing_cols) == 0:
        st.success("✅ 数据中没有缺失值，无需填补！")
        st.dataframe(df_miss.head())
        st.stop()
    
    # 选择要填补的列
    target_col = st.selectbox("选择要填补的列（包含缺失值）", missing_cols)
    
    # 找出可用于预测的特征列
    feature_cols = [col for col in numeric_cols if col != target_col and df_miss[col].isnull().sum() == 0]
    
    if len(feature_cols) == 0:
        st.error("❌ 没有可用的特征列进行填补（需要至少1列完整的数值列作为特征）")
        st.stop()
    
    st.info(f"📊 使用 {len(feature_cols)} 个特征列进行预测：{', '.join(feature_cols[:5])}")
    
    # 执行填补
    train = df_miss[df_miss[target_col].notnull()]
    test = df_miss[df_miss[target_col].isnull()]
    
    if test.empty:
        st.success("✅ 目标列没有缺失值！")
    else:
        with st.spinner(f'正在使用 {fill_algorithm} 进行填补...'):
            if fill_algorithm == "随机森林回归":
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            else:
                model = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', max_iter=500, random_state=42)
            
            model.fit(train[feature_cols], train[target_col])
            test_copy = test.copy()
            test_copy[target_col] = model.predict(test[feature_cols])
            filled = pd.concat([train, test_copy]).sort_index()
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(train[target_col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title(f'填补前 - {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('频次')
    
    axes[1].hist(filled[target_col], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title(f'填补后 - {fill_algorithm}')
    axes[1].set_xlabel(target_col)
    axes[1].set_ylabel('频次')
    plt.tight_layout()
    
    # 显示结果
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 填补前缺失数量", df_miss[target_col].isnull().sum())
        st.metric("📊 填补后数据总量", len(filled))
        if not test.empty:
            st.metric("🔢 填补值数量", len(test))
    with col2:
        st.pyplot(fig)
    
    # 显示填补值
    if not test.empty:
        with st.expander(f"📋 查看{len(test)}个填补值详情"):
            comparison = pd.DataFrame({
                '填补前': test[target_col],
                '填补后': test_copy[target_col]
            })
            st.dataframe(comparison)
    
    st.markdown("💡 **提示**：切换算法对比不同填补方法的效果。")

# ============================================================
# Tab 3: LSTM预测
# ============================================================
with tab3:
    st.header("📈 LSTM时序预测")
    
    # 加载数据
    if use_demo_data:
        macro_df = load_default_macro_data()
        st.info("📊 使用宏观经济示例数据（40个季度）")
    else:
        if uploaded_file is not None and is_valid_file(uploaded_file):
            macro_df = safe_read_csv(uploaded_file)
            if macro_df is None:
                st.stop()
            st.success(f"📊 已上传数据：{len(macro_df)} 行")
        else:
            st.warning("⚠️ 请上传数据文件或勾选「使用示例数据」")
            st.stop()
    
    # 处理日期列
    if '日期' in macro_df.columns:
        macro_df['日期'] = pd.to_datetime(macro_df['日期'])
        macro_df = macro_df.set_index('日期').sort_index()
        st.line_chart(macro_df.select_dtypes(include=[np.number]))
    
    # 选择数值列
    numeric_cols = macro_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error(f"❌ 数值列不足（仅有{len(numeric_cols)}列），需要至少2列进行预测")
        st.stop()
    
    # 选择预测目标
    target_cols = st.multiselect("选择要预测的指标", numeric_cols, default=numeric_cols[:2])
    
    if len(target_cols) < 1:
        st.warning("请至少选择一个预测指标")
        st.stop()
    
    if len(macro_df) < 10:
        st.error(f"❌ 数据量不足（仅有{len(macro_df)}条），需要至少10条数据进行LSTM训练")
        st.stop()
    
    # LSTM预测（简化版）
    data = macro_df[target_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    time_step = min(4, len(data_scaled) // 3)
    if time_step < 2:
        time_step = 2
    
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    
    if len(X) < 5:
        st.error("数据量不足，无法训练LSTM模型")
        st.stop()
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    with st.spinner(f'正在训练LSTM模型（{lstm_epochs}轮）...'):
        model = Sequential()
        model.add(LSTM(32, input_shape=(time_step, len(target_cols)), return_sequences=False))
        model.add(Dense(len(target_cols)))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=lstm_epochs, batch_size=min(4, len(X_train)), verbose=0)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)
    
    # 下一期预测
    last_seq = data_scaled[-time_step:].reshape(1, time_step, len(target_cols))
    next_pred = scaler.inverse_transform(model.predict(last_seq))
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test_actual[:, 0], label=f'真实{target_cols[0]}', color='blue', linewidth=2)
    ax.plot(y_pred[:, 0], label=f'预测{target_cols[0]}', color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('时间步')
    ax.set_ylabel(target_cols[0])
    ax.set_title(f'LSTM预测结果（训练{lstm_epochs}轮）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    col1, col2 = st.columns(2)
    with col1:
        for i, col in enumerate(target_cols):
            st.metric(f"📈 下一期{col}预测", f"{next_pred[0, i]:.2f}")
    with col2:
        st.pyplot(fig)
    
    st.markdown("💡 **提示**：增加训练轮数可以提高预测精度，但训练时间也会增加。")

# ============================================================
# Tab 4: 多源数据融合
# ============================================================
with tab4:
    st.header("🔗 多源数据融合")
    
    st.markdown("""
    ### 使用说明
    上传多个数据源，系统会自动合并：
    - 如果有共同的「客户ID」列，会按客户ID合并
    - 如果有共同的「日期」列，会按日期合并
    - 否则会按行拼接
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file1 = st.file_uploader("数据源1", type=["csv"], key="fusion1")
    with col2:
        file2 = st.file_uploader("数据源2", type=["csv"], key="fusion2")
    with col3:
        file3 = st.file_uploader("数据源3", type=["csv"], key="fusion3")
    
    if file1 and file2 and file3:
        df1 = safe_read_csv(file1)
        df2 = safe_read_csv(file2)
        df3 = safe_read_csv(file3)
        
        if df1 is not None and df2 is not None and df3 is not None:
            st.success("✅ 三个文件读取成功")
            
            # 尝试合并
            try:
                # 查找共同列
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
                
                # 处理缺失值
                numeric_cols = merged.select_dtypes(include=[np.number]).columns
                merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())
                
                st.success(f"✅ 融合完成！{merged.shape[0]} 行 × {merged.shape[1]} 列")
                st.dataframe(merged.head(10))
                
                # 下载按钮
                csv = merged.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 下载融合数据",
                    csv,
                    "融合数据.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"融合失败：{str(e)}")
    else:
        st.info("📁 请上传3个CSV文件进行数据融合")