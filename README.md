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

---
## 🎬 Demo
- **Streamlit Profile** - https://share.streamlit.io/user/ratnesh-181998
- **Project Demo** - https://edtech-learner-clustering-analysis-q7cozhkz4zafdrwwul9mpv.streamlit.app/
  
---
## � Application Modules
The dashboard is organized into 9 comprehensive sections:

1.  **📊 Data Overview:**
    *   High-level metrics (Total Learners, Features, Unique Companies).
    *   Sample raw data view.
    *   Missing value analysis and descriptive statistics.
      
    <img width="2858" height="1702" alt="image" src="https://github.com/user-attachments/assets/0f36ec59-bd49-4ce2-875d-fb2bd78be044" />
    <img width="2879" height="1701" alt="image" src="https://github.com/user-attachments/assets/b83d52c6-a67d-4ba2-857c-8c67bc5d1e9b" />

2.  **🔍 Exploratory Data Analysis (EDA):**
    *   Visualizations of key distributions: CTC (Salary), Organization Year, Top Companies, and Job Roles.
    *   Understanding the "shape" of the learner database.
      
    <img width="2861" height="1714" alt="image" src="https://github.com/user-attachments/assets/fd923733-c7b2-4b0e-89e9-5e730caef29c" />
    <img width="2875" height="1689" alt="image" src="https://github.com/user-attachments/assets/21c0e6b8-5c2e-43ea-ab87-4cf081209b51" />

3.  **📋 Case Study:**
    *   Detailed problem statement.
    *   Data dictionary explaining every column (e.g., `orgyear`, `CTC`, `Company_hash`).
      
      <img width="2825" height="1672" alt="image" src="https://github.com/user-attachments/assets/95b3242e-608a-4cf5-a47b-ebe3c83b7a84" />


4.  **🔧 Preprocessing:**
    *   **Before vs. After:** Visual comparison of data cleaning (e.g., Outlier removal in CTC).
    *   Documentation of steps: Regex text cleaning, Imputation, and Outlier clipping.
      
     <img width="2860" height="1689" alt="image" src="https://github.com/user-attachments/assets/905e9b73-60b1-4673-b41d-ca9deea2f6a2" />

5.  **⚙️ Feature Engineering:**
    *   **New Features:** 'Years of Experience', 'Company Tiers', 'Designation Class'.
    *   Visual distribution of engineered features.
      
   <img width="2855" height="1712" alt="image" src="https://github.com/user-attachments/assets/5295fbbf-c12b-4a40-ab69-1001342fd0c3" />
   <img width="2868" height="1705" alt="image" src="https://github.com/user-attachments/assets/97638b4e-e461-42f2-96a6-01a8a6c7d0ae" />


6.  **🎯 Clustering Analysis:**
    *   **Elbow Method:** Visualization to justify $k=3$ clusters.
    *   **Dendrogram:** Hierarchical clustering sample to show natural groupings.
    *   **Cluster Scatter Plot:** Interactive 2D visualization of clusters (Experience vs. CTC).
      
    <img width="2843" height="1669" alt="image" src="https://github.com/user-attachments/assets/42f36e68-6306-42b3-aa45-e6005fb7a529" />
    <img width="2865" height="1674" alt="image" src="https://github.com/user-attachments/assets/0e9fb074-0432-4cd9-8365-a767c827922b" />


7.  **💡 Insights & Recommendations:**
    *   **Cluster Profiling:** Detailed breakdown of each cluster by Tier, Class, and Designation.
    *   **Business Actions:** Specific strategies for High Value, Mid-Level, and Entry-Level learners.
     <img width="2872" height="1704" alt="image" src="https://github.com/user-attachments/assets/f70407d5-1b09-4bf1-9365-aa754772baf7" />
     <img width="2876" height="1664" alt="image" src="https://github.com/user-attachments/assets/45a9b2e4-0a85-45ec-83dc-23cc34569311" />


8.  **📝 Logs:**
    *   System logs for debugging and tracking application status.
    <img width="2879" height="1546" alt="image" src="https://github.com/user-attachments/assets/e52d659c-9b22-4826-85fb-5867904c09e4" />

9.  **📚 Complete Analysis:**
    *   A full, narrative-style walkthrough of the entire case study, mirroring the depth of a technical report.
    <img width="2879" height="1660" alt="image" src="https://github.com/user-attachments/assets/1048ef26-747f-4e3c-a468-32e2ad5a0756" />
    <img width="2515" height="1367" alt="image" src="https://github.com/user-attachments/assets/a3a2bce3-e5a1-4044-8dea-8035a92a3bac" />
    <img width="2426" height="740" alt="image" src="https://github.com/user-attachments/assets/ed42f473-9112-44c7-8d25-f1b6cbad84c2" />
    <img width="2492" height="1047" alt="image" src="https://github.com/user-attachments/assets/291d7c9f-b493-4397-90eb-9eeb40e0f9f7" />
    <img width="2512" height="1270" alt="image" src="https://github.com/user-attachments/assets/7ca361fd-9376-4bf8-b41e-9dd979d73d19" />
  
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
- 🌐 Live Demo: [Streamlit](https://edtech-learner-clustering-analysis-q7cozhkz4zafdrwwul9mpv.streamlit.app/)
- 📖 Documentation: [GitHub Wiki](https://github.com/Ratnesh-181998/EdTech-Learner-Clustering-Analysis/wiki)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/Ratnesh-181998/EdTech-Learner-Clustering-Analysis/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=24,20,12,6&height=3" width="100%">


## 📜 **License**

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

**Licensed under the MIT License** - Feel free to fork and build upon this innovation! 🚀

---

# 📞 **CONTACT & NETWORKING** 📞


### 💼 Professional Networks

[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ratneshkumar1998/)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ratnesh-181998)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/RatneshS16497)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-FF6B6B?style=for-the-badge&logo=google-chrome&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![Email](https://img.shields.io/badge/✉️_Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rattudacsit2021gate@gmail.com)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@rattudacsit2021gate)
[![Stack Overflow](https://img.shields.io/badge/Stack_Overflow-F58025?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/32068937/ratnesh-kumar)

### 🚀 AI/ML & Data Science
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/RattuDa98)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/rattuda)

### 💻 Competitive Programming (Including all coding plateform's 5000+ Problems/Questions solved )
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/Ratnesh_1998/)
[![HackerRank](https://img.shields.io/badge/HackerRank-00EA64?style=for-the-badge&logo=hackerrank&logoColor=black)](https://www.hackerrank.com/profile/rattudacsit20211)
[![CodeChef](https://img.shields.io/badge/CodeChef-5B4638?style=for-the-badge&logo=codechef&logoColor=white)](https://www.codechef.com/users/ratnesh_181998)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white)](https://codeforces.com/profile/Ratnesh_181998)
[![GeeksforGeeks](https://img.shields.io/badge/GeeksforGeeks-2F8D46?style=for-the-badge&logo=geeksforgeeks&logoColor=white)](https://www.geeksforgeeks.org/profile/ratnesh1998)
[![HackerEarth](https://img.shields.io/badge/HackerEarth-323754?style=for-the-badge&logo=hackerearth&logoColor=white)](https://www.hackerearth.com/@ratnesh138/)
[![InterviewBit](https://img.shields.io/badge/InterviewBit-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://www.interviewbit.com/profile/rattudacsit2021gate_d9a25bc44230/)


---

## 📊 **GitHub Stats & Metrics** 📊



![Profile Views](https://komarev.com/ghpvc/?username=Ratnesh-181998&color=blueviolet&style=for-the-badge&label=PROFILE+VIEWS)





<img src="https://github-readme-streak-stats.herokuapp.com/?user=Ratnesh-181998&theme=radical&hide_border=true&background=0D1117&stroke=4ECDC4&ring=F38181&fire=FF6B6B&currStreakLabel=4ECDC4" width="48%" />




<img src="https://github-readme-activity-graph.vercel.app/graph?username=Ratnesh-181998&theme=react-dark&hide_border=true&bg_color=0D1117&color=4ECDC4&line=F38181&point=FF6B6B" width="48%" />

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&lines=Ratnesh+Kumar+Singh;Data+Scientist+%7C+AI%2FML+Engineer;4%2B+Years+Building+Production+AI+Systems" alt="Typing SVG" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=Built+with+passion+for+the+AI+Community+🚀;Innovating+the+Future+of+AI+%26+ML;MLOps+%7C+LLMOps+%7C+AIOps+%7C+GenAI+%7C+AgenticAI+Excellence" alt="Footer Typing SVG" />


<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%">


