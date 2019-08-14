from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

X_PCA = pca.transform(X_scaled)
print("원본 데이터 형태:", X_scaled.shape)
print("축소된 데이터 형태:", X_PCA.shape)

# pc의 특성 추출을 n_components을 통해, 결과를 판단해 PCA로 가져옴

# PCA의 원리: 하나의 축을 선택하여, 점을 몰아넣는 방법
