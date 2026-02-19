#EDA-Method
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\Heart_discase_model\heart_disease_realistic_large.csv")
print(df.head(5))

# shape df
print(df.shape)

# info of data
print(df.info())

# described for stast
print(df.describe())

# Target Desciptions
print(df['target'].value_counts())

# isull values check karu
print(df.isnull().sum())
print(df.isna().sum())

#Check duplicated
print("duplicate value is ",df.duplicated().sum())

# Outliers check
plt.figure(figsize=(10,6))
num_colum=df.select_dtypes(include=["int64","float64"])
num_colum.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of All Features")
plt.show()

#Count-Plot # 0=No disease,1=Disease
plt.figure(figsize=(10,6))
sns.countplot(x="target",data=df)
plt.title("Heart-disease Distribution")
plt.show()

# Heat-map
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=False,cmap='coolwarm')
plt.title("correlation-Heatmap")
plt.show()

# scatter-plot
#plt.style.use("dark_background")
sns.set_style("whitegrid")
sns.scatterplot(x="age", y="chol", hue="target", data=df,palette=["#1f77b4","#ff4c4c"],alpha=0.6)
plt.title("Age vs Cholesterol",fontsize=14,fontweight="bold")
plt.show()

# Histogram for age distribution 
plt.figure(figsize=(6,4))
sns.histplot(df['age'],bins=20,kde=True)
plt.title("Histogram  Age Distribution")
plt.show()

# violin plot (chol vs Target)
plt.figure(figsize=(6,4))
sns.violinplot(x='target',y='chol',data=df,palette=["#6C9BCF" , "#F67280"],inner="quartile")
plt.title("Cholestrol VS Heart Disease")
plt.show()
 
#Bar-plot sex vs disease
plt.figure(figsize=(6,4))
sns.barplot(x="sex",y="target",data=df,palette=["#4E79A7", "#F28E2B"])
plt.title("Gender vs Heart Disease Risk", fontsize=14, fontweight="bold")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Average Disease Rate")
plt.tight_layout()
plt.show()

# pair-plot
important_cols = ["age","chol","trestbps","thalach","oldpeak","target"]
sns.pairplot(df[important_cols].sample(800),hue="target",palette=["#2E86C1", "#F67280"],diag_kind="kde",plot_kws={"alpha":0.6})
plt.show()

# ------------------------------------------------------------ML-PART-STARTING-----------------------------------------------------------------#

x=df.drop("target",axis=1)
y=df['target']
print("features shape : ",x.shape)
print("target-shape : ",y.shape)

#Train-Test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("training data : ",x_train.shape)
print("Testing-data  :",x_test.shape)

# Decision - Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

dt_model=DecisionTreeClassifier( max_depth=4,min_samples_split=5, min_samples_leaf= 2, random_state=42)
dt_model.fit(x_train,y_train)
predict_dt=dt_model.predict(x_test)
ac_score=accuracy_score(predict_dt,y_test)
metrix=confusion_matrix(predict_dt,y_test)
print("Your cm matrix is ",metrix)
print("Your accuracy score is ",ac_score)
print(classification_report(y_test,predict_dt))

# SVM-model (Scalling)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
svm_model=SVC()
svm_model.fit(x_train_scaled,y_train)
predict=svm_model.predict(x_test_scaled)
ac=accuracy_score(y_test,predict)
matrix=confusion_matrix(y_test,predict)
print("SVM model : accuracy ",ac)
print("model matrix is ",matrix)

print("\n Decision tree report is : ",(y_test,predict_dt))
print("\n Svm Report is ")
print(classification_report(y_test,predict))

# Roc curve
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

#Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#SVM Model (probability=True VERY IMPORTANT)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(x_train_scaled, y_train)

#Probability for class 1
y_prob = svm_model.predict_proba(x_test_scaled)[:,1]

# ROC calculation
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

#Attractive Style
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, color="#ff4d6d", linewidth=3, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1], [0,1], linestyle='--', color="navy", linewidth=2)

plt.fill_between(fpr, tpr, alpha=0.3, color="#ff99ac")

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - SVM Model", fontsize=14, fontweight='bold')
plt.legend()
plt.show()
