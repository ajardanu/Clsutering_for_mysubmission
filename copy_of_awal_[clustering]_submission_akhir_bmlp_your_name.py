# -*- coding: utf-8 -*-
# **1. Import Library**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

# 2. Memuat Dataset

url='https://drive.google.com/uc?id=1gnLO9qvEPqv1uBt1928AcsCmdvzqjC5m'
df = pd.read_csv(url)

df.head()

df.info()

df.describe()

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

num_vars = df.shape[1]

n_cols = 4
n_rows = -(-num_vars // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

axes = axes.flatten()

for i, column in enumerate(df.columns):
    df[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#3. Pembersihan dan Pra Pemrosesan Data


df.isnull().sum()

df.duplicated().sum()

numeric_features = df.select_dtypes(include=['number']).columns

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
df.head()

FiturTidakRelevan = ['TransactionID', 'AccountID','DeviceID', 'IP Address', 'MerchantID']

df = df.drop(columns=FiturTidakRelevan, errors='ignore')
df.head()

categorical_features = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in categorical_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

df_lencoder.head()

df_lencoder.columns.tolist()

"""(Opsional) Pembersihan dan Pra Pemrosesan Data [Skilled]

**Biarkan kosong jika tidak menerapkan kriteria skilled**
"""

for col in numeric_features:
    df_lencoder[col] = df_lencoder[col].fillna(df_lencoder[col].mean())

for col in categorical_features:
    df_lencoder[col] = df_lencoder[col].fillna(df_lencoder[col].mode()[0])
df_lencoder.isnull().sum()

duplicates = df_lencoder.duplicated()
df_lencoder = df_lencoder.drop_duplicates()
df_lencoder.head()

Q1 = df_lencoder[numeric_features].quantile(0.25)
Q3 = df_lencoder[numeric_features].quantile(0.75)
IQR = Q3 - Q1

condition = ~((df_lencoder[numeric_features] < (Q1 - 1.5 * IQR)) |
              (df_lencoder[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

df_lencoder = df_lencoder.loc[condition].reset_index(drop=True)

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_lencoder[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

# Melakukan binning data berdasarkan kondisi rentang nilai pada fitur numerik,
# lakukan pada satu sampai dua fitur numerik.
# Silahkan lakukan encode hasil binning tersebut menggunakan LabelEncoder.
# Pastikan kamu mengerjakan tahapan ini pada satu cell.

# 4. Membangun Model Clustering

df_lencoder.describe()

kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(1, 10))

visualizer.fit(df_lencoder)

visualizer.show()

k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(df_lencoder)

joblib.dump(kmeans, "model_clustering.h5")

"""(Opsional) Membangun Model Clustering [Skilled]

**Biarkan kosong jika tidak menerapkan kriteria skilled**
"""

sil_score = silhouette_score(df_lencoder, labels)

print(f"Silhouette Score untuk k={k}: {sil_score:.4f}")

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_lencoder)

plt.figure(figsize=(8,6))
plt.scatter(df_pca[:,0], df_pca[:,1], c=labels, cmap="viridis", alpha=0.6)
plt.title("Visualisasi Cluster dengan PCA (k=3)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

"""(Opsional) Membangun Model Clustering [Advanced]

**Biarkan kosong jika tidak menerapkan kriteria advanced**
"""

pca = PCA(n_components=2)

df_pca = pca.fit_transform(df_lencoder)

data_final = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data_final)

labels = kmeans.labels_

data_final['target'] = labels
score = silhouette_score(data_final.drop("target", axis=1), labels)

print(f"Silhouette Score untuk k={k}: {score:.4f}")

joblib.dump(kmeans, "PCA_model_clustering.h5")

#5. Interpretasi Cluster



df_clustered = df_lencoder.copy()
df_clustered["target"] = data_final["target"]

fitur_pilihan = ["TransactionAmount", "CustomerAge",
                 "TransactionDuration", "LoginAttempts",
                 "AccountBalance", "target"]

df_pilihan = df_clustered[fitur_pilihan]

deskriptif = df_pilihan.groupby("target").agg(["mean", "min", "max"])
deskriptif


df_clustered.head()

df_clustered.to_csv('data_clustering.csv', index=False)

"""(Opsional) Interpretasi Hasil Clustering [Skilled]

**Biarkan kosong jika tidak menerapkan kriteria skilled**
"""

df_clustered[numeric_features] = scaler.inverse_transform(df_clustered[numeric_features])
df_clustered.head()

