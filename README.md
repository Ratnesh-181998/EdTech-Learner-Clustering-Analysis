# 🎓 EdTech Learner Clustering Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview
**EdTech Learner Clustering Analysis** is a sophisticated Data Science project designed to segment and profile learners from an online tech-versity. By leveraging unsupervised machine learning techniques (K-Means and Hierarchical Clustering), this application identifies distinct learner groups based on their professional background, compensation, and experience.

The insights derived from this analysis empower the EdTech platform to:
*   **Personalize Learning Paths:** Tailor course recommendations for different career stages.
*   **Optimize Placements:** Match high-potential learners with top-tier companies.
*   **Strategic Decision Making:** Understand the demographic composition of the learner base.

This project features a **high-performance Streamlit Dashboard** that provides an interactive and visually engaging walkthrough of the entire analysis, from raw data exploration to actionable business recommendations.

## � Application Modules
The dashboard is organized into 9 comprehensive sections:

1.  **📊 Data Overview:**
    *   High-level metrics (Total Learners, Features, Unique Companies).
    *   Sample raw data view.
    *   Missing value analysis and descriptive statistics.

2.  **🔍 Exploratory Data Analysis (EDA):**
    *   Visualizations of key distributions: CTC (Salary), Organization Year, Top Companies, and Job Roles.
    *   Understanding the "shape" of the learner database.

3.  **📋 Case Study:**
    *   Detailed problem statement.
    *   Data dictionary explaining every column (e.g., `orgyear`, `CTC`, `Company_hash`).

4.  **🔧 Preprocessing:**
    *   **Before vs. After:** Visual comparison of data cleaning (e.g., Outlier removal in CTC).
    *   Documentation of steps: Regex text cleaning, Imputation, and Outlier clipping.

5.  **⚙️ Feature Engineering:**
    *   **New Features:** 'Years of Experience', 'Company Tiers', 'Designation Class'.
    *   Visual distribution of engineered features.

6.  **🎯 Clustering Analysis:**
    *   **Elbow Method:** Visualization to justify $k=3$ clusters.
    *   **Dendrogram:** Hierarchical clustering sample to show natural groupings.
    *   **Cluster Scatter Plot:** Interactive 2D visualization of clusters (Experience vs. CTC).

7.  **💡 Insights & Recommendations:**
    *   **Cluster Profiling:** Detailed breakdown of each cluster by Tier, Class, and Designation.
    *   **Business Actions:** Specific strategies for High Value, Mid-Level, and Entry-Level learners.

8.  **📝 Logs:**
    *   System logs for debugging and tracking application status.

9.  **📚 Complete Analysis:**
    *   A full, narrative-style walkthrough of the entire case study, mirroring the depth of a technical report.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn (StandardScaler, KMeans), SciPy (Hierarchical Clustering)

## 📂 Project Structure
```
├── app.py                      # Main Streamlit application
├── scaler_clustering.csv       # Dataset (Learner data)
├── clustering_analysis_final.py # Analysis script (Reference)
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ratnesh-181998/EdTech-Learner-Clustering.git
    cd EdTech-Learner-Clustering
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 💡 Insights & Clusters

The analysis identified 3 distinct learner clusters:

| Cluster | Profile | Characteristics | Recommendation |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | **High Value / Leaders** | Senior professionals (8-33 yrs exp), High CTC (>40 LPA), Tier 1 Companies. | Target for leadership roles and mentorship programs. |
| **Cluster 1** | **Mid-Level** | Moderate experience (5-12 yrs), Mid-range CTC. | Focus on upskilling for Tier 1 company transitions. |
| **Cluster 2** | **Entry / Junior** | Early career (1-15 yrs), Lower CTC range. | Focus on foundational skills and gaining experience. |

## 📞 Contact

**RATNESH SINGH**

- 📧 Email: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/ratneshkumar1998/](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 GitHub: [https://github.com/Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 Phone: +91-947XXXXX46

### Project Links
- 🌐 Live Demo: [Streamlit App](https://share.streamlit.io/) *(Link to be updated after deployment)*
- 📖 Documentation: [GitHub Wiki](https://github.com/Ratnesh-181998/EdTech-Learner-Clustering/wiki)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/Ratnesh-181998/EdTech-Learner-Clustering/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
