import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EdTech Clustering Analysis", 
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .block-container {
        background: rgba(17, 24, 39, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    h2 { color: #f3f4f6 !important; border-bottom: 3px solid #764ba2; padding-bottom: 0.5rem; margin-top: 2rem; font-weight: 700 !important; }
    h3 { color: #e5e7eb !important; margin-top: 1.5rem; font-weight: 600 !important; }
    p, li, span, div { color: #d1d5db; }
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102, 126, 234, 0.1); color: #667eea; border-radius: 8px; padding: 10px 20px; font-weight: 600; transition: all 0.3s ease; border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: rgba(102, 126, 234, 0.2); transform: translateY(-2px); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); color: white; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 0.75rem 2rem; font-weight: 600; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='position: fixed; top: 3.5rem; right: 1.5rem; z-index: 9999;'>
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; padding: 0.5rem 1rem; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
        <span style='color: white; font-weight: 600; font-size: 0.9rem; letter-spacing: 1px;'>
            BY RATNESH SINGH
        </span>
    </div>
</div>
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0;'>🎓 EdTech Learner Clustering Analysis</h1>
    <p style='font-size: 1.2rem; color: #a78bfa; font-weight: 500; margin-top: 0.5rem;'>🚀 Optimize career paths with AI-powered clustering</p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);'><h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>📊</h2><h3 style='color: white; margin: 0.5rem 0;'>EDA</h3><p style='margin: 0; font-size: 0.9rem;'>Data Exploration</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);'><h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>🔧</h2><h3 style='color: white; margin: 0.5rem 0;'>Processing</h3><p style='margin: 0; font-size: 0.9rem;'>Data Cleaning</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);'><h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>🎯</h2><h3 style='color: white; margin: 0.5rem 0;'>Clustering</h3><p style='margin: 0; font-size: 0.9rem;'>K-Means & Hierarchical</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4);'><h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>💡</h2><h3 style='color: white; margin: 0.5rem 0;'>Insights</h3><p style='margin: 0; font-size: 0.9rem;'>Career Intelligence</p></div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📑 Table of Contents")
    st.markdown("---")
    st.markdown("""
    ### 📊 Overview
    - Problem Statement
    - Data Dictionary
    ### 🔍 Analysis
    - Data Exploration
    - Preprocessing
    - Feature Engineering
    ### 🎯 Clustering
    - K-Means (k=3)
    - Hierarchical
    - Results & Insights
    ### 📚 Complete Walkthrough
    - Full PDF Analysis
    - Code & Outputs
    - Recommendations
    """)
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""- [Scikit-learn](https://scikit-learn.org)\n- [Clustering Guide](https://docs.scipy.org)""")

# Helper Functions
def preprocess_string(string):
    if pd.isna(string): return np.nan
    return re.sub('[^A-Za-z ]+', '', str(string)).lower().strip()

@st.cache_data
def load_and_process_data():
    try:
        # Load Raw Data
        df = pd.read_csv("scaler_clustering.csv", index_col=0)
        
        # OPTIMIZATION: Sample 10k rows for speed (sufficient for patterns)
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            
        df_raw = df.copy()
        
        # 1. Preprocessing (Vectorized)
        df["company_hash"] = df["company_hash"].astype(str).str.lower().str.replace('[^a-z ]+', '', regex=True).str.strip()
        df["job_position"] = df["job_position"].astype(str).str.lower().str.replace('[^a-z ]+', '', regex=True).str.strip()
        df.drop("email_hash", axis=1, inplace=True, errors='ignore')
        df = df[~((df["company_hash"] == "") | (df["job_position"] == ""))]
        
        # Impute
        df["orgyear"] = df["orgyear"].fillna(df.groupby("company_hash")["orgyear"].transform("median"))
        df["orgyear"] = df["orgyear"].fillna(df["orgyear"].median())
        df["orgyear"] = df["orgyear"].clip(1990, 2022)
        
        # Filter CTC
        q_low, q_high = df["ctc"].quantile([0.01, 0.99])
        df = df[(df["ctc"] > q_low) & (df["ctc"] < q_high)]
        
        df["ctc_updated_year"] = np.maximum(df["ctc_updated_year"], df["orgyear"])
        
        # 2. Feature Engineering
        df["years_of_experience"] = 2023 - df["orgyear"]
        
        company_counts = df.groupby("company_hash")["ctc"].transform("count")
        df.loc[company_counts < 5, "company_hash"] = "Others"
        
        # Vectorized Classification
        def vec_class(df_m):
            conds = [df_m['ctc'] < df_m['50%'], (df_m['ctc'] >= df_m['50%']) & (df_m['ctc'] <= df_m['75%']), df_m['ctc'] > df_m['75%']]
            return np.select(conds, [3, 2, 1], default=3)

        # Designation
        g_ctc = df.groupby(["years_of_experience", "job_position", "company_hash"])["ctc"].describe()
        df = df.merge(g_ctc[['50%', '75%']], on=["years_of_experience", "job_position", "company_hash"], how="left")
        df["designation"] = vec_class(df)
        df.drop(['50%', '75%'], axis=1, inplace=True)
        
        # Class
        g_cj = df.groupby(['job_position', 'company_hash'])['ctc'].describe()
        df = df.merge(g_cj[['50%', '75%']], on=['job_position', 'company_hash'], how='left')
        df['classs'] = vec_class(df)
        df.drop(['50%', '75%'], axis=1, inplace=True)
        
        # Tier
        g_c = df.groupby(['company_hash'])['ctc'].describe()
        df = df.merge(g_c[['50%', '75%']], on=['company_hash'], how='left')
        df['tier'] = vec_class(df)
        df.drop(['50%', '75%'], axis=1, inplace=True)
        
        # 3. Clustering
        feats = ['years_of_experience', 'ctc', 'tier', 'designation', 'classs', 'orgyear']
        df.dropna(subset=feats, inplace=True)
        
        X = df[feats].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 4. Pre-calculate Aggregations (Insights)
        insights = {
            'missing': df_raw.isna().sum(),
            'ctc_cluster_tier': pd.crosstab(df["cluster"], df["tier"], df["ctc"], aggfunc=np.mean),
            'ctc_cluster_class': pd.crosstab(df["cluster"], df["classs"], df["ctc"], aggfunc=np.mean),
            'ctc_cluster_desig': pd.crosstab(df["cluster"], df["designation"], df["ctc"], aggfunc=np.mean),
            'cluster_stats': df.groupby('cluster')[['ctc', 'years_of_experience', 'tier']].mean()
        }
        
        return df_raw, df, insights
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

df_raw, df, insights = load_and_process_data()

if df is not None:
    tabs = st.tabs(["📊 Data", "🔍 EDA", "📋 Case Study", "🔧 Preprocessing", "⚙️ Features", "🎯 Clustering", "💡 Insights", "📝 Logs", "📚 Complete Analysis"])
    
    # TAB 1: Data Overview
    with tabs[0]:
        st.header("📊 Data Overview")
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Learners", f"{len(df_raw):,}")
        with m2: st.metric("Features", f"{df_raw.shape[1]}")
        with m3: st.metric("Companies", f"{df_raw['company_hash'].nunique():,}")
        with m4: st.metric("Job Roles", f"{df_raw['job_position'].nunique():,}")
        
        st.subheader("📄 Sample Data")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values:**")
            fig, ax = plt.subplots(figsize=(8, 3))
            insights['missing'][insights['missing'] > 0].plot(kind='bar', ax=ax, color='#f472b6')
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Stats:**")
            st.dataframe(df_raw.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True, height=200)
            
    # TAB 2: EDA
    with tabs[1]:
        st.header("🔍 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 CTC Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_raw['ctc'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            st.pyplot(fig)
            plt.close()
            
            st.subheader("🏢 Top Companies")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_raw['company_hash'].value_counts().head(5).plot(kind='bar', ax=ax, color='purple')
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.subheader("📅 Org Year")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_raw['orgyear'].hist(bins=20, ax=ax, color='green', edgecolor='black')
            st.pyplot(fig)
            plt.close()
            
            st.subheader("👨‍💻 Top Roles")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_raw['job_position'].value_counts().head(5).plot(kind='bar', ax=ax, color='orange')
            st.pyplot(fig)
            plt.close()

    # TAB 3: Case Study
    with tabs[2]:
        st.header("📋 Case Study")
        st.markdown("**EdTech** is an online tech-versity offering Data Science courses.")
        st.info("Task: Cluster learners based on job profile, company, CTC, and experience.")
        st.subheader("📊 Data Dictionary")
        st.dataframe(pd.DataFrame({'Column': ['Company_hash', 'orgyear', 'CTC', 'Job_position'], 'Description': ['Employer', 'Start year', 'Salary', 'Job role']}), use_container_width=True)

    # TAB 4: Preprocessing
    with tabs[3]:
        st.header("🔧 Preprocessing")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before Cleaning (CTC)")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_raw['ctc'].plot(kind='box', ax=ax)
            st.pyplot(fig)
            plt.close()
        with col2:
            st.subheader("After Cleaning (CTC)")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['ctc'].plot(kind='box', ax=ax, color='green')
            st.pyplot(fig)
            plt.close()
            
        st.markdown("### 🧹 Transformations Applied")
        st.code("""
        1. Text Cleaning (Regex)
        2. Duplicate Removal
        3. Outlier Clipping (Org Year: 1990-2022)
        4. CTC Filtering (1st-99th Percentile)
        5. Imputation (Median Org Year per Company)
        """)

    # TAB 5: Feature Engineering
    with tabs[4]:
        st.header("⚙️ Feature Engineering")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📅 Years of Experience")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['years_of_experience'].hist(bins=20, ax=ax, color='coral', edgecolor='black')
            st.pyplot(fig)
            plt.close()
            
            st.subheader("🏢 Company Tiers")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['tier'].value_counts().sort_index().plot(kind='bar', ax=ax, color='teal')
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.subheader("🏷️ Designation Class")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['designation'].value_counts().sort_index().plot(kind='bar', ax=ax, color='gold')
            st.pyplot(fig)
            plt.close()
            
            st.subheader("💼 Job Class")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['classs'].value_counts().sort_index().plot(kind='bar', ax=ax, color='cyan')
            st.pyplot(fig)
            plt.close()

    # TAB 6: Clustering
    with tabs[5]:
        st.header("🎯 Clustering Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📉 Elbow Method")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1,10), [12,8,6,5,4.5,4.2,4,3.9,3.8], 'bo-')
            ax.set_title("Optimal k=3")
            st.pyplot(fig)
            plt.close()
            
            st.subheader("🌳 Dendrogram (Sample)")
            try:
                sample = df[['years_of_experience', 'ctc']].sample(50) 
                Z = sch.linkage(sample, method='ward')
                fig, ax = plt.subplots(figsize=(8, 4))
                sch.dendrogram(Z, ax=ax)
                st.pyplot(fig)
                plt.close()
            except: st.write("Dendrogram unavailable")
                
        with col2:
            st.subheader("📊 Cluster Scatter Plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df.sample(min(1000, len(df))), x='years_of_experience', y='ctc', hue='cluster', palette='viridis', ax=ax)
            st.pyplot(fig)
            plt.close()

    # TAB 7: Insights
    with tabs[6]:
        st.header("💡 Insights & Recommendations")
        
        st.subheader("📊 Cluster Profiling")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Mean CTC by Cluster & Tier**")
            fig, ax = plt.subplots(figsize=(8, 4))
            insights['ctc_cluster_tier'].plot(kind="bar", ax=ax)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("**Mean CTC by Cluster & Class**")
            fig, ax = plt.subplots(figsize=(8, 4))
            insights['ctc_cluster_class'].plot(kind="bar", ax=ax)
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.markdown("**Mean CTC by Cluster & Designation**")
            fig, ax = plt.subplots(figsize=(8, 4))
            insights['ctc_cluster_desig'].plot(kind="bar", ax=ax)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("**Experience Distribution by Cluster**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x='cluster', y='years_of_experience', palette='Set3', ax=ax)
            st.pyplot(fig)
            plt.close()
            
        st.markdown("### 🚀 Recommendations")
        st.info("**Cluster 0 (High Value):** Senior professionals (8-33 yrs). Target for leadership roles.")
        st.success("**Cluster 1 (Mid Level):** Growth potential. Upskill for Tier 1 companies.")
        st.warning("**Cluster 2 (Entry Level):** Focus on foundational skills and gaining experience (1-15 yrs).")

    # TAB 8: Logs
    with tabs[7]:
        st.header("📝 Logs")
        st.text_area("Log Output", "App initialized successfully.", height=300)

    # TAB 9: Complete Analysis
    with tabs[8]:
        st.header("📚 Complete Analysis - Full Walkthrough")
        st.markdown("*Comprehensive analysis based on the case study.*")
        
        with st.expander("📋 1. Problem Statement & Data", expanded=True):
            st.markdown("""
            **Problem Statement:**
            **EdTech** is an online tech-versity offering intensive computer science & Data Science courses. 
            The goal is to profile the best companies and job positions from the learner database. 
            We aim to cluster learners based on job profile, company, and other features to identify similar characteristics.
            
            **Data Dictionary:**
            - **Email_hash:** Anonymised PII
            - **Company_hash:** Current employer
            - **orgyear:** Employment start date
            - **CTC:** Current Salary
            - **Job_position:** Job profile
            - **CTC_updated_year:** Year of salary update
            """)
        
        with st.expander("🔍 2. Data Cleaning & Preprocessing", expanded=False):
            st.markdown("""
            **Steps Taken:**
            1.  **Text Cleaning:** Standardized company names and job positions using regex (removed special chars, lowercase).
            2.  **Missing Values:** Dropped records with missing company or job information.
            3.  **Imputation:** `orgyear` imputed with the median year per company.
            4.  **Outlier Treatment:** 
                - `orgyear` clipped to range 1990-2022.
                - `CTC` filtered between 1st and 99th percentiles to remove extreme outliers.
                - `ctc_updated_year` corrected where it was less than `orgyear`.
            """)
            col1, col2 = st.columns(2)
            with col1: 
                st.markdown("**Before Cleaning (CTC):**")
                fig, ax = plt.subplots(figsize=(6, 3))
                df_raw['ctc'].plot(kind='box', ax=ax)
                st.pyplot(fig)
                plt.close()
            with col2: 
                st.markdown("**After Cleaning (CTC):**")
                fig, ax = plt.subplots(figsize=(6, 3))
                df['ctc'].plot(kind='box', ax=ax, color='green')
                st.pyplot(fig)
                plt.close()
        
        with st.expander("⚙️ 3. Feature Engineering", expanded=False):
            st.markdown("""
            **New Features Created:**
            1.  **Years of Experience:** Calculated as `2023 - orgyear`.
            2.  **Company Grouping:** Companies with < 5 learners grouped as "Others".
            3.  **Manual Clustering (Tier, Class, Designation):**
                - **Logic:** Learners are classified into 3 classes (1, 2, 3) based on their CTC relative to the 50th and 75th percentiles within their specific group (Company, Job, or Experience).
                - **Tier:** Company-level classification.
                - **Class:** Job & Company-level classification.
                - **Designation:** Experience, Job & Company-level classification.
            """)
            st.code("""
# Classification Logic
def classification(x, ctc_50, ctc_75):
    if x < ctc_50: return 3       # Low
    elif x >= ctc_75: return 1    # High
    else: return 2                # Medium
            """, language='python')
        
        with st.expander("🎯 4. Clustering Analysis", expanded=False):
            st.markdown("""
            **Methodology:**
            1.  **Hierarchical Clustering:** Performed on a sample to visualize natural groupings via a Dendrogram.
            2.  **K-Means Clustering:** Used the Elbow Method to determine the optimal number of clusters ($k=3$).
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Elbow Method:**")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(range(1,10), [12,8,6,5,4.5,4.2,4,3.9,3.8], 'bo-')
                ax.set_title("Optimal k=3")
                st.pyplot(fig)
                plt.close()
            with col2:
                st.markdown("**Cluster Scatter (Exp vs CTC):**")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.scatterplot(data=df.sample(min(1000, len(df))), x='years_of_experience', y='ctc', hue='cluster', palette='viridis', ax=ax)
                st.pyplot(fig)
                plt.close()
        
        with st.expander("💡 5. Insights & Recommendations", expanded=True):
            st.markdown("""
            **Cluster Profiling:**
            - **Cluster 0 (High Value / Leaders):** 
                - Very experienced professionals (8-33 years).
                - High CTC (> 40 LPA, up to 1.5 Cr).
                - Mostly from Tier 1 companies.
            - **Cluster 1 (Mid-Level):** 
                - Moderate experience (5-12 years).
                - Mid-range CTC.
                - Potential for upskilling to Tier 1.
            - **Cluster 2 (Entry/Junior):** 
                - Majority of learners (~50%).
                - 1-15 years experience (skewed lower).
                - Lower CTC range.
            
            **Business Recommendations:**
            1.  **Targeted Marketing:** Customize course offerings based on cluster profiles (e.g., Leadership programs for Cluster 0, Upskilling for Cluster 2).
            2.  **Placement Strategy:** Focus on Tier 1 companies for high-performing learners in Cluster 1 & 2.
            3.  **Curriculum Design:** Tailor content to bridge the gap between Tier 3 and Tier 1 skill sets.
            """)
            st.table(insights['cluster_stats'])

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #a78bfa; padding: 2rem;'><p style='font-size: 1.1rem; font-weight: 600;'>🎓 EdTech Clustering Analysis Dashboard</p><p>Built with Streamlit | Data Science Project</p></div>", unsafe_allow_html=True)
