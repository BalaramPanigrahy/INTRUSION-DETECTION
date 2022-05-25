import pandas as pd
from flask import Flask, render_template, request, flash
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from flask import *

from main import load_data

app= Flask(__name__)
app.config['UPLOAD_FOLDER']=r"Dataset"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

global data, x_train, x_test, y_train, y_test

df_train = pd.read_csv('Dataset/train.txt')
df_test = pd.read_csv('Dataset/test.txt')

cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class']

df_train.columns= cols
df_test.columns= cols

data = pd.concat([df_train, df_test], join= 'outer')

num_vars = data.select_dtypes(exclude= 'object')

ss = StandardScaler()
scaled_val= ss.fit_transform(num_vars)

col = num_vars.columns

scaled_val= pd.DataFrame(scaled_val, columns=col)

df=pd.DataFrame()
le = LabelEncoder()
df['protocol_type']= le.fit_transform(data['protocol_type'])
df['service']= le.fit_transform(data['service'])
df['flag']= le.fit_transform(data['flag'])

df= pd.concat([scaled_val,df], axis=1)
y= data['class']



from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='full')

pca.fit(df)
x_pca= pca.transform(df)

x= pd.DataFrame(x_pca, columns=['duration','src_bytes','dst_bytes','logged_in','count','srv_count','dst_host_count','protocol_type', 'service', 'flag'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


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
        temp_df= load_data(os.path.join(app.config["UPLOAD_FOLDER"],myfile))
        # full_data=clean_data(full_data)
        return render_template('view_dataset.html', col=temp_df.columns.values, df=list(temp_df.values.tolist()))
    return render_template('view_dataset.html')

x_train=None; y_train =None;
x_test=None; y_test=None

@app.route('/training', methods= ['GET','POST'])
def training():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if request.method== 'POST':
        model_no= int(request.form['algo'])

        if model_no==0:
            flash(r"You have not selected any model", "info")

        elif model_no == 1:
            model = SVC()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            svcr = accuracy_score(y_test, pred)
            msg = "Accuracy of SVM is :" + str(svcr)


        elif model_no== 2:
            cfr = RandomForestClassifier()
            model = cfr.fit(x_train, y_train)
            pred = model.predict(x_test)
            rfcr= accuracy_score(y_test, pred)
            msg= "Accuracy of Random Forest is :"+ str(rfcr)




        elif model_no== 3:
            xgc = xgb.XGBClassifier()
            model = xgc.fit(x_train, y_train)
            pred = model.predict(x_test)
            xgcr = accuracy_score(y_test, pred)
            msg = "Accuracy of XgBoost is :" + str(xgcr)



        elif model_no== 4:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            pred_y = dt.predict(x_test)
            acc_dt = accuracy_score(y_test, pred_y)
            msg = "Accuracy of Decision Tree is :" + str(acc_dt)



        elif model_no== 5:
            model = KNeighborsClassifier()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            accuracy_score(y_test, pred)
            kncr = accuracy_score(y_test, pred)
            msg = "Accuracy of KNeighbors is :" + str(kncr)
        return render_template('train_model.html', mag = msg)
    return render_template('train_model.html')


@app.route('/prediction', methods= ['GET', 'POST'])
def prediction():
    if request.method== "POST":
        duration= request.form['duration']
        print(duration)
        src_bytes= request.form['src_bytes']
        print(src_bytes)
        dst_bytes= request.form['dst_bytes']
        print(dst_bytes)
        logged_in= request.form['logged_in']
        print(logged_in)
        count= request.form['count']
        print(count)
        srv_count= request.form['srv_count']
        print(srv_count)
        dst_host_count= request.form['dst_host_count']
        print(dst_host_count)
        protocol_type= request.form['protocol_type']
        print(protocol_type)
        service= request.form['service']
        print(service)
        flag= request.form['flag']
        print(flag)

        di= {'duration' : [duration], 'src_bytes' : [src_bytes], 'dst_bytes' : [dst_bytes], 'logged_in' : [logged_in],
             'count' : [count],'srv_count' : [srv_count], 'dst_host_count' : [dst_host_count],
             'protocol_type' : [protocol_type], 'service' : [service],'flag' : [flag]}

        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        cfr = RandomForestClassifier()
        model = cfr.fit(x_train, y_train)
        output = model.predict(test)
        print(output)

        if output[0] == 'anomaly':
            msg = 'There is a possible <span style = color:red;>INTRUSION DETECTED</span></b> in the system'

        else:
            msg = 'The system is working normally <span style = color:green;>WITHOUT ANY INTRUSION(s)</span></b>'


        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')



if __name__=='__main__':
    app.run(debug=True)
