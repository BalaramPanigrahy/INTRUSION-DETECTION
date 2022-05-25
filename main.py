from flask import Flask, render_template, request, url_for, flash, redirect
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import shutil
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


app=Flask(__name__)
app.config['UPLOAD_FOLDER']=r"C:\Users\YMTS0519\Documents\intrusion\Dataset"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'


def load_data(path):
    cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class']
    df = pd.read_csv(path, header=None)
    df.columns = cols
    return df

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ss = StandardScaler()
from sklearn.decomposition import PCA
Final_PCA=None
# from sklearn.preprocessing import Normalizer
# nor=Normalizer()

def data_prep_train():
    global ohe, ss, Final_PCA
    ### one-hot-encoding data
    full_train = load_data(r"Uploaded_data/train.txt")
    ###for train dataset
    #object type encoding
    df_obj = full_train.drop(['class'], axis=1).select_dtypes('object')
    objs = df_obj.columns.values.tolist()
    df_1hot = ohe.fit_transform(df_obj)
    df_1hot=pd.DataFrame(df_1hot, columns=ohe.get_feature_names(['protocol_type', 'service', 'flag']))
    #numerical columns scaling
    all_cols=full_train.drop(['class'], axis=1).columns
    nums=[c for c in all_cols if c not in objs]
    df_nums=full_train[nums]
    df_scale=ss.fit_transform(df_nums)
    # df_scale=nor.fit_transform(df_nums)
    df_scale=pd.DataFrame(df_scale, columns=nums)
    #combining
    df_combined=pd.concat([df_1hot, df_scale], axis=1)

    #PCA
    for comp in range(3, df_combined.shape[1]):
        pca = PCA(n_components=comp, random_state=42)
        pca.fit(df_combined)
        comp_check = pca.explained_variance_ratio_
        print(comp_check)
        final_comp = comp
        if comp_check.sum() > 0.90:
            break

    Final_PCA = PCA(n_components=final_comp, random_state=42)
    Final_PCA.fit(df_combined)
    df_train_X = Final_PCA.transform(df_combined) #Final dataset for training
    df_train_y=full_train['class']
    return df_train_X, df_train_y

def data_prep_test():
    ### For Testing Dataset
    full_test = load_data(r"Uploaded_data/test.txt")

    #object type encoding
    df_obj = full_test.drop(['class'], axis=1).select_dtypes('object')
    objs = df_obj.columns.values.tolist()
    df_1hot = ohe.transform(df_obj)
    df_1hot=pd.DataFrame(df_1hot, columns=ohe.get_feature_names(['protocol_type', 'service', 'flag']))

    #numerical columns scaling
    all_cols=full_test.drop(['class'], axis=1).columns
    nums=[c for c in all_cols if c not in objs]
    df_nums=full_test[nums]
    df_scale=ss.transform(df_nums)
    # df_scale=nor.transform(df_nums)
    df_scale=pd.DataFrame(df_scale, columns=nums)

    #combining
    df_combined=pd.concat([df_1hot, df_scale], axis=1)
    df_test_X = Final_PCA.transform(df_combined) #Final dataset for testing
    df_test_y = full_test['class']
    return df_test_X, df_test_y

def support_vector_machines(train_X, train_y, test_X, test_y):
    svc=SVC()
    svc.fit(train_X, train_y)
    pred_y=svc.predict(test_X)
    acc_svc=accuracy_score(test_y, pred_y)
    flash(r"Support Vector Classifier Created Successfully", 'secondary')
    return acc_svc

def random_forest(train_X, train_y, test_X, test_y):
    rf = RandomForestClassifier()
    rf.fit(train_X, train_y)
    pred_y = rf.predict(test_X)
    acc_rf = accuracy_score(test_y, pred_y)
    flash("Random Forest Created Successfully", 'secondary')
    return acc_rf

import xgboost as xgb
def xgboost_algo(train_X, train_y, test_X, test_y):
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=1, n_neighbors=11)
    X, y = nm.fit_resample(train_X, train_y)
    xgb_model=xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, colsample_bytree=0.6, subsample=0.8, gamma=3, eta=0.3, min_child_weight=0.75)
    xgb_model.fit(X, y)
    pred_y = xgb_model.predict(test_X)
    acc_xgb = accuracy_score(test_y, pred_y)
    flash("XGBoost Created Successfully", 'secondary')
    return acc_xgb


def decision_tree(train_X, train_y, test_X, test_y):
    dt = DecisionTreeClassifier()
    dt.fit(train_X, train_y)
    pred_y = dt.predict(test_X)
    acc_dt = accuracy_score(test_y, pred_y)
    flash("Decision Tree Created Successfully", 'secondary')
    return acc_dt



def k_nearest_neighbor(train_X, train_y, test_X, test_y):
    knn = KNeighborsClassifier()
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    acc_knn = accuracy_score(test_y, pred_y)
    flash("K Nearest Neighbors Created Successfully", 'secondary')
    return acc_knn


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load', methods=["POST","GET"])
def load():
    if request.method=="POST":
        train_file=request.files['train']
        test_file=request.files['test']
        ext1=os.path.splitext(train_file.filename)[1]
        ext2 = os.path.splitext(test_file.filename)[1]
        if ext1.lower() == ".txt" and ext2.lower()=='.txt':
            try:
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            except:
                pass
            os.mkdir(app.config['UPLOAD_FOLDER'])
            train_file.save(os.path.join(app.config['UPLOAD_FOLDER'],'train.txt'))
            test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt'))
            flash('The data is loaded successfully','success')
            return render_template('load_dataset.html')
        else:
            flash('Please upload a txt type documents only','warning')
            return render_template('load_dataset.html')
    return render_template('load_dataset.html')

@app.route('/view', methods=['POST', 'GET'])
def view():
    if request.method=='POST':
        myfile=request.form['data']
        if myfile=='0':
            flash(r"Please select an option",'warning')
            return render_template('view_dataset.html')
        temp_df=load_data(os.path.join(app.config["UPLOAD_FOLDER"],myfile))
        # full_data=clean_data(full_data)
        return render_template('view_dataset.html', col=temp_df.columns.values, df=list(temp_df.values.tolist()))
    return render_template('view_dataset.html')

train_X=None; train_y =None;
test_X=None; test_y=None

@app.route('/train_model', methods=['GET','POST'])
def train_model():
    global train_X, train_y, test_X, test_y
    if request.method=="POST":
        model_no=int(request.form['algo'])
        if model_no==0:
            flash(r"You have not selected any model", "info")
        elif model_no==1:
            #support vector machine
            acc_svm = support_vector_machines(train_X, train_y, test_X, test_y)
            return render_template('train_model.html', acc=acc_svm, model=model_no)
        elif model_no==2:
            #random forest
            acc_rf = random_forest(train_X, train_y, test_X, test_y)
            return render_template('train_model.html', acc=acc_rf, model=model_no)
        elif model_no==3:
            #xgboost
            acc_xgb = xgboost_algo(train_X, train_y, test_X, test_y)
            return render_template('train_model.html', acc=acc_xgb, model=model_no)

        elif model_no==4:
            #decisiontree
            acc_dt = decision_tree(train_X, train_y, test_X, test_y)
            return render_template('train_model.html', acc=acc_dt, model=model_no)

        elif model_no==5:
            #knearestneighbor
            acc_knn = k_nearest_neighbor(train_X, train_y, test_X, test_y)
            return render_template('train_model.html', acc=acc_knn, model=model_no)

    else:
        train_X, train_y = data_prep_train()
        test_X, test_y = data_prep_test()
    return render_template('train_model.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        #Accepts all values
        print('----------------')
        f1=float(request.form['f1'])
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])
        f12 = float(request.form['f12'])
        f13 = float(request.form['f13'])
        f14 = float(request.form['f14'])
        f15 = float(request.form['f15'])
        f16 = float(request.form['f16'])
        f17 = float(request.form['f17'])
        f18 = float(request.form['f18'])
        f19 = float(request.form['f19'])
        f20 = float(request.form['f20'])
        f21 = float(request.form['f21'])
        f22 = float(request.form['f22'])
        f23 = float(request.form['f23'])
        f24 = float(request.form['f24'])
        f25 = float(request.form['f25'])
        f26 = float(request.form['f26'])
        f27 = float(request.form['f27'])
        f28 = float(request.form['f28'])
        f29 = float(request.form['f29'])
        f30 = float(request.form['f30'])
        f31 = float(request.form['f31'])
        f32 = float(request.form['f32'])
        f33 = float(request.form['f33'])
        f34 = float(request.form['f34'])
        f35 = float(request.form['f35'])
        f36 = float(request.form['f36'])
        f37 = float(request.form['f37'])
        f38 = float(request.form['f38'])
        f39 = float(request.form['f39'])
        f40 = float(request.form['f40'])
        f41 = float(request.form['f41'])

        ohe_vars=[f2, f3, f4]
        print(ohe_vars)
        scale_vars=[f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27,
                    f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41]
        print(scale_vars)

        ohe_trans = list(ohe.fit_transform([ohe_vars])[0])
        scale_trans = list(ss.fit_transform([scale_vars])[0])

        comb = ohe_trans + scale_trans

        mypca= Final_PCA.transform([comb])[0]
        print(mypca)

        #Model
        mymodel = pickle.load(open("model_xgb.pkl", "rb"))
        pred = mymodel.predict(pd.DataFrame(mypca).transpose())[0]
        print(pred)
        return render_template('prediction.html', pred=pred)
    return render_template('prediction.html')

if __name__=='__main__':
    app.run(debug=True)