import numpy as np
import pandas as pd
#from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.externals import joblib

##prepare data
def split_data(data):
    train_x=data.drop(['service_type','current_service','user_id'],axis=1)
    train_y=pd.DataFrame(data['current_service'])
    return train_x,train_y
def data_prepare(path):##train_data的路径
    df = pd.read_csv(path)
    ##通过数据不同分布，将训练数据分为三种不同的类型分布
    dftype1_num3= df[df.service_type==1]
    dftype3_num1= df[df.service_type==3]
    dftype4_num11=df[df.service_type==4]
    
    dftype1_num3_train_x_y=split_data(dftype1_num3)
    dftype3_num1_train_x_y=split_data(dftype3_num1)
    dftype4_num11_train_x_y=split_data(dftype4_num11)
    
    return dftype1_num3,dftype3_num1_train_x_y,dftype4_num11_train_x_y##返回三种类型的训练数据
	
##train three model
def make_xgboost(data,path,model_name,num_class):##接受1.分类型的训练数据2.保存模型地址3.分类别的数目
    model_name=xgb.XGBClassifier(max_depth=6,learning_rate=0.01,n_estimators=1000,silent=True,objective='multi:softmax',
                        num_class=num_class,booster='gbtree',n_jobs=-1,nthread=-1,gamma=0,min_child_weight=1,max_delta_step=0,
                        subsample=0.7,colsample_bylevel=1,reg_alpha=0,reg_lambda=0)
    model_name.fit(data[0],data[1])
    joblib.dump(model_name,path)
##test 3 class model 
def test_3model(model3_path,model11_path,testcsv_path,savecsv_path):##接受1.训练好模型的地址2.测试数据的地址3.保存结果的地址
    df = pd.read_csv(testcsv_path)
    user= pd.DataFrame(df['user_id'])
    model3=joblib.load(model3_path)
    model11=joblib.load(model11_path)
    tmp_list=[]
    for i in df.index:
        if df.iloc[i]['service_type']==1:
            predict= model3.predict(pd.DataFrame(df.iloc[i].drop(['user_id','service_type']).values,columns=[0],
			index=['is_mix_service','online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 
			'month_traffic', 'many_over_bill', 'contract_type', 'contract_time', 'is_promise_low_consume', 'net_service', 
			'pay_times', 'pay_num', 'last_month_traffic', 'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 
			'service2_caller_time', 'gender', 'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee'],dtype=float).T)[0]
        if df.iloc[i]['service_type']==4:
            predict= model11.predict(pd.DataFrame(df.iloc[i].drop(['user_id','service_type']).values,columns=[0],index=['is_mix_service','online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic', 'many_over_bill', 'contract_type', 'contract_time', 'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic', 'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender', 'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee'],dtype=float).T)[0]
        if df.iloc[i]['service_type']==3:
            predict=99104722
        tmp_list.append(predict)
    add_predict=pd.DataFrame(tmp_list,columns=['predict'])
    user.join(add_predict).to_csv(savecsv_path)
#if __name__=='__main__':
data_class3,data_class1,data_class11=data_prepare('C:\\Users\\Lx\\Desktop\\x\\train.csv')
make_xgboost(data_class3,model_name='type1_class3',path='C:\\Users\\Lx\\Desktop\\x\\type1_class3.pkl',num_class=3)
make_xgboost(data_class11,model_name='type4_class11',path='C:\\Users\\Lx\\Desktop\\x\\type4_class11.pkl',num_class=11)
test_3model(model3_path='C:\\Users\\Lx\\Desktop\\x\\type1_class3.pkl',model11_path='C:\\Users\\Lx\\Desktop\\x\\type4_class11.pkl',testcsv_path='C:\\Users\\Lx\\Desktop\\x\\test.csv',savecsv_path='C:\\Users\\Lx\\Desktop\\x\\submit.csv')