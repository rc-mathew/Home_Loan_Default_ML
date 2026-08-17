from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.features import build_feature_table, make_model_matrix, SELECTED_FEATURES
from src.metrics import evaluate_binary, find_f2_threshold


def save_plots(y_test, model_probs, results_df, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5))
    for name, prob in model_probs.items():
        fpr,tpr,_=roc_curve(y_test,prob); auc=roc_auc_score(y_test,prob)
        plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(artifact_dir/"roc_curves.png",dpi=140); plt.close()

    plt.figure(figsize=(7,5))
    for name, prob in model_probs.items():
        p,r,_=precision_recall_curve(y_test,prob); ap=average_precision_score(y_test,prob)
        plt.plot(r,p,label=f"{name} (PR-AUC={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(artifact_dir/"precision_recall_curves.png",dpi=140); plt.close()

    ax=results_df.set_index("Model")[['ROC_AUC','PR_AUC','F1','Recall']].plot(kind="bar",figsize=(8,5))
    ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    plt.tight_layout(); plt.savefig(artifact_dir/"model_comparison.png",dpi=140); plt.close()


def train(data_dir='data', model_dir='models', test_size=0.20, validation_size=0.20, seed=42):
    data_dir=Path(data_dir); model_dir=Path(model_dir); artifact_dir=ROOT/'artifacts'
    model_dir.mkdir(parents=True,exist_ok=True); artifact_dir.mkdir(parents=True,exist_ok=True)
    df=build_feature_table(data_dir)
    if 'TARGET' not in df: raise ValueError('application_train.csv must contain TARGET.')
    X=make_model_matrix(df); y=df.TARGET.astype('int8')
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=test_size,stratify=y,random_state=seed)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=validation_size, stratify=y_dev,random_state=seed)
    neg,pos=int((y_train==0).sum()),int((y_train==1).sum()); spw=neg/max(pos,1)
    prevalence=float(y.mean())
    (artifact_dir/'class_distribution.json').write_text(json.dumps({'positive_rate':prevalence,'negative_rate':1-prevalence,'train_positive':pos,'train_negative':neg,'scale_pos_weight':spw},indent=2))
    models={
      'logistic_regression':Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=200,class_weight='balanced',solver='lbfgs',random_state=seed))]),
      'xgboost':XGBClassifier(n_estimators=120,learning_rate=.05,max_depth=3,min_child_weight=20,subsample=.8,colsample_bytree=.7,reg_alpha=.1,reg_lambda=5,gamma=0,objective='binary:logistic',eval_metric='auc',tree_method='hist',max_bin=64,device='cpu',n_jobs=2,scale_pos_weight=spw,random_state=seed)
    }
    val_rows=[]; test_probs={}; fitted={}
    for name,model in models.items():
        print('Training',name); model.fit(X_train,y_train); fitted[name]=model
        pv=model.predict_proba(X_val)[:,1]; t,_=find_f2_threshold(y_val,pv)
        row=evaluate_binary(y_val,pv,t); row.update({'Model':name,'Threshold':t,'F2_optimized':True}); val_rows.append(row)
        print(row)
    val_df=pd.DataFrame(val_rows).sort_values(['PR_AUC','ROC_AUC'],ascending=False).reset_index(drop=True)
    best_name=val_df.iloc[0].Model; best_model=fitted[best_name]; best_t=float(val_df.iloc[0].Threshold)
    # Test only after model/threshold selection.
    pt=best_model.predict_proba(X_test)[:,1]; test_metrics=evaluate_binary(y_test,pt,best_t); test_metrics.update({'Model':best_name,'Threshold':best_t})
    pd.DataFrame([test_metrics]).to_csv(model_dir/'holdout_test_metrics.csv',index=False)
    pd.DataFrame(val_df).to_csv(model_dir/'validation_metrics.csv',index=False)
    save_plots(y_test,{n:fitted[n].predict_proba(X_test)[:,1] for n in fitted},val_df.rename(columns={'Threshold':'_'}),artifact_dir)
    # Refit selected model on development data for production artifact; keep test untouched.
    best_model.fit(X_dev,y_dev)
    model_path=model_dir/'home_loan_default_model.joblib'; joblib.dump(best_model,model_path,compress=3)
    metadata={'model_name':best_name,'target':'TARGET','n_features':len(SELECTED_FEATURES),'selected_features':SELECTED_FEATURES,'split':'60% train / 20% validation / 20% untouched holdout test','selection_metric':'PR-AUC then ROC-AUC on validation','threshold_selection':'F2 maximization on validation','test_metrics':test_metrics,'class_imbalance':{'positive_rate':prevalence,'scale_pos_weight':spw,'logistic_class_weight':'balanced'},'risk_segments':{'eligible_max_exclusive':.20,'manual_review_max_exclusive':.40},'temporal_validation':'Not claimed: the supplied application_train.csv has no true application/decision timestamp. Use scripts/temporal_validation.py with an approved timestamp when available.'}
    (model_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)); val_df.to_csv(model_dir/'model_metrics.csv',index=False)
    print('Saved',model_path); print('Untouched test metrics:',test_metrics); return val_df

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data'); p.add_argument('--model-dir',default='models'); p.add_argument('--test-size',type=float,default=.20); p.add_argument('--validation-size',type=float,default=.20); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); train(a.data_dir,a.model_dir,a.test_size,a.validation_size,a.seed)
