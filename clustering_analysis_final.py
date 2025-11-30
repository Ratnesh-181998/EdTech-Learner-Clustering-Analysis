"""
Problem Statement
Scaler is an online tech-versity offering intensive computer science & Data Science courses through live classes delivered by tech leaders and subject matter experts.
The meticulously structured program enhances the skills of software professionals by offering a modern curriculum with exposure to the latest technologies. It is a product by InterviewBit.
You are working as a data scientist with the analytics vertical of Scaler, focused on profiling the best companies and job positions to work for from the Scaler database.
You are provided with the information for a segment of learners and tasked to cluster them on the basis of their job profile, company, and other features. Ideally, these clusters should have similar characteristics.

Data Dictionary:
‘Unnamed 0’- Index of the dataset
Email_hash- Anonymised Personal Identifiable Information (PII)
Company_hash- Current employer of the learner
orgyear- Employment start date
CTC- Current CTC
Job_position- Job profile in the company
CTC_updated_year: Year in which CTC got updated (Yearly increments, Promotions)

Concept Used:
Manual Clustering
Unsupervised Clustering - K- means, Hierarchical Clustering
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (12,8)
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("scaler_clustering.csv", index_col=0)

# Display basic info (commented out output)
print(df.sample(10))
print(df.shape) # 205843 learners data
df.info()
print(df.isna().sum())

print(df.describe())
print(df.describe(include="object"))

def preprocess_string(string):
    new_string = re.sub('[^A-Za-z ]+', '', string).lower().strip()
    return new_string

mystring='\tAirtel\\\\&&**() X Labs'
print(preprocess_string(mystring))

# Data Cleaning
print(df["company_hash"].nunique())
df["company_hash"] = df["company_hash"].apply(lambda x: preprocess_string(str(x)))
print(df["company_hash"].nunique())

print(df["job_position"].nunique()) # 1017 unique job positions
df["job_position"] = df["job_position"].apply(lambda x: preprocess_string(str(x)))
print(df["job_position"].nunique()) # 857 unique job positions

# removing the email_hash
df.drop("email_hash", axis=1, inplace=True)
print(df.sample(5))

print(df.duplicated().sum()) # 17597 duplicated records
# df.drop_duplicates(inplace=True) # This was commented out in some parts but used later? 
# The PDF shows df.duplicated().sum() then later df.drop_duplicates(inplace=True) in In [188] context?
# Wait, looking at line 60 of extracted text: "df.duplicated().sum() df.drop_duplicates(inplace=True)"
# So it seems it was executed.
# But wait, line 26 says "df.duplicated().sum() # 17597 duplicated records".
# Then line 60 says "df.drop_duplicates(inplace=True)".
# I will assume it should be dropped.

# Handling Missing Values
print((df["company_hash"] == "").sum())
print((df["company_hash"] == "nan").sum())
print((df["job_position"] == "").sum())
print((df["job_position"] == "nan").sum())

# removing the records where company or job_position records are not available
print(df[(df["company_hash"] == "") | (df["job_position"] == "")].sample(10))
print(len(df[(df["company_hash"] == "") | (df["job_position"] == "")]))

df = df[~((df["company_hash"] == "") | (df["job_position"] == ""))]

# Data Preprocessing
# imputing Employee Start Year as per the median year as per each company.
print(df["orgyear"].isna().sum())
df.groupby("company_hash")["orgyear"].transform("median")
df["orgyear"].fillna(df['orgyear'].isnull().sum(), inplace=True)
print(df["orgyear"].isna().sum())

# Outliers Treatment : employement start year
print(df.sample(5))
print(df["orgyear"].value_counts())

import seaborn as sns
sns.countplot(x=df["orgyear"])
plt.xticks(rotation = 90)
plt.show()

sns.histplot(np.log(df["orgyear"]))
plt.show()
print(df["orgyear"].quantile(0.001))
print(df["orgyear"].quantile(0.999))
df["orgyear"] = df["orgyear"].clip(1990, 2022)

# ctc updated_year
sns.countplot(x=df["orgyear"]) 
plt.xticks(rotation = 90)
plt.show()

print(df["ctc_updated_year"].quantile(0.001))
print(df["ctc_updated_year"].quantile(0.99))

# outlier treatment for CTC
sns.countplot(x=df["ctc_updated_year"])
plt.xticks(rotation = 90)
plt.show()

print(df["ctc"].quantile(0.01))
print(df["ctc"].quantile(0.999))
df = df.loc[((df.ctc) > df.ctc.quantile(0.01)) & ((df.ctc) < df.ctc.quantile(0.99))]

# replacing string "nan" to np.nan
sns.distplot(df["ctc"])
plt.show()

# Feature Engineering
# Masked company name to "Others" having count less than 5
# years of experience = current year - employement start year

df.loc[df['job_position']=='nan', 'job_position']=np.nan
df.loc[df["company_hash"]=="nan","company_hash"] = np.nan

df.loc[df.groupby("company_hash")["ctc"].transform("count") < 5, "company_hash"] = "Others"
print((df["company_hash"] == "Others").sum())

# years of experience
df["years_of_experience_in_organization"] = 2023 - df["orgyear"]

sns.countplot(x=df["years_of_experience_in_organization"])
plt.show()

# df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(df.shape)

# treating records having ctc_updated_year higher than their organization joining year
print((df["ctc_updated_year"] < df["orgyear"]).sum())
df.ctc_updated_year = df[["ctc_updated_year", "orgyear"]].max(axis=1)
print((df["ctc_updated_year"] < df["orgyear"]).sum())

df['job_position'] = df['job_position'].fillna('Others')
df['company_hash'] = df['company_hash'].fillna('Others')
print(df.isna().sum())

print(df.describe())
df.info()

# Manual Clustering based on Company , Job position and Years of experience
# Learner's "designation_in_organization"

sns.scatterplot(x=df.ctc, y=df.years_of_experience_in_organization)
plt.show()

GROUPED_CTC = df.groupby(["years_of_experience_in_organization", "job_position", "company_hash"])["ctc"].describe()
df_GROUPED_CTC_BY_E_P_C = df.merge(GROUPED_CTC, on=["years_of_experience_in_organization", "job_position", "company_hash"], how="left")

# whichever learner has ctc compared to their years of experience , respective company , position
# giving designation as 3 when ctc is < 50th percentile in his position ,experience and company
# giving designation as 2 when ctc is between 50th and 75th percentile in his position ,experience and company
# giving designation as 1 when ctc is > 75th percentile in his position ,experience and company

def classification(x, ctc_50, ctc_75):
    if x < ctc_50:
        return 3
    elif x >= ctc_50 and x <= ctc_75:
        return 2
    elif x >= ctc_75:
        return 1

df_GROUPED_CTC_BY_E_P_C["designation_in_organization"] = df_GROUPED_CTC_BY_E_P_C.apply(lambda x: classification(x["ctc"], x["50%"], x["75%"]), axis=1)

print(df_GROUPED_CTC_BY_E_P_C.designation_in_organization.value_counts(normalize=True))

df_GROUPED_CTC_BY_E_P_C.drop(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)

# Manual Clustering on company and job position
# grouping by each job_position and company, finding which class of job an individual have,
# based on his ctc compared to his job_position and respective company.

GROUPED_C_J = df.groupby(['job_position', 'company_hash'])['ctc'].describe()
df_GROUPED_C_J = df.merge(GROUPED_C_J, on=['job_position', 'company_hash'], how='left')

# creating classes basis on the salary in their respective company
df_GROUPED_C_J['classs'] = df_GROUPED_C_J.apply(lambda x: classification(x['ctc'], x['50%'], x['75%']), axis=1)

print(df_GROUPED_C_J.classs.value_counts(normalize=True))
df_GROUPED_C_J.drop(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)

# Manual Clustering based on comapny
# based on ctc per company , assigning company as tier 1 2 and 3 per each learners

df_Grouped = df_GROUPED_CTC_BY_E_P_C.merge(df_GROUPED_C_J, on=['company_hash', 'orgyear', 'ctc', 'job_position', 'years_of_experience_in_organization', 'ctc_updated_year'], how='left')

GROUPED_C = df.groupby(['company_hash'])['ctc'].describe()
df_company = df.merge(GROUPED_C, on=['company_hash'], how='left')

df_company['tier'] = df_company.apply(lambda x: classification(x['ctc'], x['50%'], x['75%']), axis=1)

print(df_company.tier.value_counts(normalize=True))
df_company.drop(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1, inplace=True)

df_Grouped = df_Grouped.merge(df_company, on=['company_hash', 'orgyear', 'ctc', 'job_position', 'years_of_experience_in_organization', 'ctc_updated_year'])

# Final data for Model :
print(df_Grouped.head())

X = df_Grouped.copy()
X_data = X.drop(["company_hash", "job_position"], axis=1)

# Standardization:
scaler = StandardScaler()
scaler.fit(X_data)
X_sc = pd.DataFrame(scaler.transform(X_data), columns=X_data.columns, index=X_data.index)

# hierarchical Custering :
# trying to get a high level idea about how many clusters we can from, by taking sample of 500 learners multiple times and forming hierarchy and visualising in dendrogram.

sample = X_sc.sample(500)
Z = sch.linkage(sample, method='ward')
fig, ax1 = plt.subplots(figsize=(20, 12))
sch.dendrogram(Z, labels=sample.index, ax=ax1, color_threshold=2)
plt.xticks(rotation=90)
ax1.set_ylabel('distance')
plt.show()

# (Repeated plots in PDF)

# Based on dendrogram , we can observe there are 3 clusters in the data based on similarity
# Further checking appropriate number of clusters using Elbow Method using k-Means clustering :

# KMeans
for i in range(1,10):
    k = 4
    kM = KMeans(n_clusters=k, random_state=654)
    y_pred = kM.fit_predict(X_sc)

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_sc) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(12, 8))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow', xy=(3, inertias[2]), xytext=(0.55, 0.55), textcoords='figure fraction', fontsize=16, arrowprops=dict(facecolor='black', shrink=0.1))
plt.show()

# KMeans with n_clusters = 3
k = 3
kM = KMeans(n_clusters=k, random_state=654)
y_pred = kM.fit_predict(X_sc)
clusters = pd.DataFrame(X, columns=X.columns)
clusters['label'] = kM.labels_

# Insights | EDA after Clustering :
sns.scatterplot(x=clusters["orgyear"], y=clusters["ctc"], hue=clusters["label"])
plt.show()

# based on above scatter plot , we can observe , a cluster of learners received CTC upto 30 LPA who joined after 2006-07.
# there's a group of learners who are very much experienced.
# and also learners joined after 2012-13 receiving CTC between 20 LPA to upto 1.5cr.

pd.crosstab(index=clusters["label"], columns=clusters["tier"], values=clusters["ctc"], aggfunc=np.mean).plot(kind="bar")
plt.show()

# Based on k-Means Clustering algorithm output , as well as manual clustering , learners from tier1 company receiving very high CTC.

pd.crosstab(index=clusters["label"], columns=clusters["classs"], values=clusters["ctc"], aggfunc=np.mean).plot(kind="bar")
plt.show()

pd.crosstab(index=clusters["label"], columns=clusters["designation_in_organization"], values=clusters["ctc"], aggfunc=np.mean).plot(kind="bar")
plt.show()

pd.crosstab(columns=clusters["label"], index=clusters["years_of_experience_in_organization"], values=clusters["ctc"], aggfunc=np.mean).plot(kind="bar")
plt.show()

# Cluster label 0 , are those learners who are very very experienced,experienced learners between 6 to 10 years of experience, earning above 40 LPA up tp 1.5Cr.

# Majority of Learners are experienced between 1 to 15 years . (49.73%)- (Cluster 2)
# there is a group of learners having 8 to upto 33 years of experience. (33%) - (Cluster 0)
# 16.95% of learners who have experiences - (cluster 1)

pd.crosstab(columns=clusters["label"], index=clusters["years_of_experience_in_organization"]).plot(kind="bar")
plt.show()

print(clusters.label.value_counts(normalize=True)*100)

# years_of_experience_in_organization per each cluster group of learners
pd.crosstab(index=clusters["label"], columns=clusters["tier"], values=clusters["years_of_experience_in_organization"], aggfunc=np.mean).plot(kind="bar")
plt.show()

# Statistical Summury based on Each Cluster :
print(clusters.groupby("label").describe()[["ctc","classs","tier","years_of_experience_in_organization"]].T)

print("Analysis complete. Script executed successfully.")
