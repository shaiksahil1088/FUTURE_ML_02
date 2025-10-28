import numpy as np 
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve,accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import seaborn as sns

df = pd.read_csv("churn_modelling.csv")
df.head(10)

df.shape

df.info

target = "Exited"
x = df.drop(columns=[target])
y = df[target]

x

y

x = x.copy()
for col in x.columns:
    if x[col].dtype == "object":
        x[col] = x[col].fillna("MISSING")
        x[col] = LabelEncoder().fit_transform(x[col])
    else:
        x[col] = x[col].fillna(x[col].median())

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=1),
    "XGBoost": XGBClassifier(
        n_estimators=300,learning_rate=0.05,use_label_encoder=False,eval_metric="logloss",random_state=42
    )
}
results = {}
scored_dfs = {}

for name,model in models.items():
    model.fit(x_scaled,y)
    y_pred = model.predict(x_scaled)
    y_prob = model.predict_proba(x_scaled)[:,1]

    acc = accuracy_score(y,y_pred)
    roc_auc = roc_auc_score(y,y_pred)
    cm = confusion_matrix(y,y_pred)
    report = classification_report(y,y_pred,output_dict=True)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "conf_matrix": cm,
        "report": report
    }

    scored_df = df.copy()
    scored_df[f"{name}_Predicted_Churn"] = y_pred
    scored_df[f"{name}_Churn_Probability"] = y_pred
    scored_dfs[name] = scored_df

best_model_name = max(results,key = lambda x: results[x]["roc_auc"])
best_model = results[best_model_name]["model"]

model_path = "final_best_model.pkl"
joblib.dump(best_model,model_path)

scored_csv_path = "scored_customers.csv"
scored_dfs[best_model_name].to_csv(scored_csv_path,index=False)

plt.figure()
for name , res in results.items():
    fpr , tpr , _ = roc_curve(y, res["model"].predict_proba(x_scaled)[:,1])
    plt.plot(fpr,tpr,label=f"{name} (AUC={res['roc_auc']:.2f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
roc_img_path = "roc_curves_comparison.png"
plt.savefig(roc_img_path)
plt.show()

for name, model in models.items():
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

best_model = models["Random Forest"] 
y_proba = best_model.predict_proba(x)[:,1]

plt.figure(figsize=(7,5))
sns.histplot(y_proba, bins=30, kde=True, color="orange")
plt.title("Predicted Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()

pdf_path = "final_churn_report.pdf"
doc= SimpleDocTemplate(pdf_path)
styles = getSampleStyleSheet()
elements = []

elements.append(Paragraph("Customer Churn Prediction - Model Comparison Report",styles["Title"]))
elements.append(Spacer(1,12))
elements.append(Paragraph(f"Target column: {target}",styles["Normal"]))
elements.append(Paragraph(f"Rows used for training: {len(df)}",styles["Normal"]))
elements.append(Spacer(1,12))

table_data = [["Model","Accuracy","ROC-AUC"]]
for name , res in results.items():
    table_data.append([name ,f"{res['accuracy']:.3f}",f"{res['roc_auc']:.3f}"])
comp_table = Table(table_data,colWidths=[200,100,100])
comp_table.setStyle(TableStyle([
    ("Background",[0,0],[-1,0],colors.grey),
    ("Grid",[0,0],[-1,-1],1,colors.black),
    ("Align",[0,0],[-1,-1],"CENTER"),
]))
elements.append(Paragraph("Model Performance Comparison:",styles["Heading2"]))
elements.append(comp_table)
elements.append(Spacer(1,12))

elements.append(Paragraph("ROC Curves:", styles["Heading2"]))
elements.append(Image(roc_img_path,width=400,height=300))

for name in ["Random Forest","XGBoost"]:
    if name in results:
        importances = results[name]["model"].feature_importances_
        feat_imp = sorted(zip(x.columns,importances),key=lambda x: x[1],reverse=True)[:10]
        feat_table = Table([[f,round(v,3)] for f , v in feat_imp],colWidths=[200,100])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(Paragraph(f"Top 10 Features Driving Churn - {name}:", styles["Heading2"]))
        elements.append(feat_table)
        elements.append(Spacer(1, 12))
doc.build(elements)

print("Training complete with 3 models.")
print(f"Best model: {best_model_name}")
print(f"Model saved to: {model_path}")
print(f"Scored data saved to: {scored_csv_path}")
print(f"PDF report saved to: {pdf_path}")