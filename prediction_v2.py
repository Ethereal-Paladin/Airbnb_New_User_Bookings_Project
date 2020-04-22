import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import bisect
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler,OneHotEncoder
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from ndcg import ndcg_scorer
import ndcg





def pre_processing(training,test):

    # split date into year, month and day
    training['date_account_created_year'] = training['date_account_created'].apply(lambda x: x.split('-')[0]).astype('int')
    test['date_account_created_year'] = test['date_account_created'].apply(lambda x: x.split('-')[0]).astype('int')

    training['date_account_created_month'] = training['date_account_created'].apply(lambda x: x.split('-')[1]).astype('int')
    test['date_account_created_month'] = test['date_account_created'].apply(lambda x: x.split('-')[1]).astype('int')

    training['date_account_created_day'] = training['date_account_created'].apply(lambda x: x.split('-')[2]).astype('int')
    test['date_account_created_day'] = test['date_account_created'].apply(lambda x: x.split('-')[2]).astype('int')

    # split timestamp into year, month and day
    training['timestamp_year'] = training['timestamp_first_active'].astype(str).apply(lambda x: x[:4])
    test['timestamp_year'] = test['timestamp_first_active'].astype(str).apply(lambda x: x[:4])

    training['timestamp_month'] = training['timestamp_first_active'].astype(str).apply(lambda x: x[4:6])
    test['timestamp_month'] = test['timestamp_first_active'].astype(str).apply(lambda x: x[4:6])

    training['timestamp_day'] = training['timestamp_first_active'].astype(str).apply(lambda x: x[6:8])
    test['timestamp_day'] = test['timestamp_first_active'].astype(str).apply(lambda x: x[6:8])

    # fill nan str value with '-unknown-'
    cate = training.select_dtypes(include=object, exclude=None).columns.values.tolist()
    cate_test = test.select_dtypes(include=object, exclude=None).columns.values.tolist()
    training[cate] = training[cate].fillna('-unknown-')
    test[cate_test] = test[cate_test].fillna('-unknown-')

    # drop the column 'date_account_created','date_first_booking','timestamp_first_active'
    train_id = training['id']
    training = training.drop(['date_account_created','date_first_booking','timestamp_first_active'],axis=1)
    # training = training.drop(['date_account_created','timestamp_first_active'],axis=1)
    test_id = test['id']
    test = test.drop(['date_account_created','date_first_booking','timestamp_first_active'],axis=1)
    # test = test.drop(['date_account_created','timestamp_first_active'],axis=1)
    test_year = test['date_account_created_year']
    train_year = training['date_account_created_year']

    # convert string to number using LabelEncoder()
    # cate = training.select_dtypes(include=object, exclude=None).columns.difference(['id']).values.tolist()
    # cate_test = test.select_dtypes(include=object, exclude=None).columns.difference(['id']).values.tolist()
    # training[cate] = training[cate].astype(str)
    # test[cate_test] = test[cate_test].astype(str)
    #
    # target_label = {}
    #
    # le = LabelEncoder()
    # for i in cate:
    #     le.fit(training[i])
    #     le_classes = le.classes_.tolist()
    #     if '-unknown-' not in le_classes and i!='country_destination':
    #         bisect.insort_left(le_classes, '-unknown-')
    #         pass
    #     le.classes_ = le_classes
    #
    #     training[i] = le.transform(training[i])
    #     if i=='country_destination':
    #         target_label = {k: v for k, v in enumerate(le.classes_)}
    #         continue
    #         pass
    #     test[i] = test[i].map(lambda s: '-unknown-' if s not in le.classes_ else s)
    #     test[i] = le.transform(test[i])
    #     pass

    # handle target label
    le = LabelEncoder()
    le.fit(training['country_destination'])
    training['country_destination'] = le.transform(training['country_destination'])
    target_label = {k: v for k, v in enumerate(le.classes_)}

    # use OneHotEncoder() to handle the original string
    cate = training.select_dtypes(include=object, exclude=None).columns.difference(['id','country_destination']).values.tolist()
    cate_test = test.select_dtypes(include=object, exclude=None).columns.difference(['id']).values.tolist()
    training[cate] = training[cate].astype(str)
    test[cate_test] = test[cate_test].astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore').fit(training[cate])
    training_label = pd.DataFrame(ohe.transform(training[cate]).toarray())
    test_label = pd.DataFrame(ohe.transform(test[cate_test]).toarray())

    # drop original string features and appends the one hot features
    training = training.drop(cate,axis=1)
    test = test.drop(cate_test,axis=1)
    training = pd.concat([training,training_label],axis=1)
    test = pd.concat([test,test_label],axis=1)

    # fill nan numeric value with 0
    training = training.fillna(0)
    test = test.fillna(0)

    # drop target from training dataset
    X_train = training.drop(['country_destination'],axis=1)
    Y_train = training['country_destination']
    X_test = test

    # use the information of session dataset
    X_train,X_test = session_process(X_train,X_test)

    # drop the column 'id'
    X_train = X_train.drop(['id'],axis=1)
    X_test = X_test.drop(['id'],axis=1)

    # scale all the features with StandardScaler()
    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)

    return X_train,Y_train,X_test,test_id,test_year,target_label,train_id,train_year

def session_process(X_train,X_test):

    session =  pd.read_csv("sessions.csv")

    # change column name
    session['id'] = session['user_id']
    session = session.drop(['user_id'],axis=1)

    # fill nan str value with 'NULL'
    session[['action','action_type','action_detail','device_type','id']]=session[['action','action_type','action_detail','device_type','id']].fillna('NULL')

    # fill nan numeric value with 0
    session['secs_elapsed'] = session['secs_elapsed'].fillna(0)

    # create feature: sum of secs_elapsed group by id
    session_by_id_time = session.groupby(['id'])['secs_elapsed'].sum().astype(int).reset_index()

    # create features: counts of features
    session_num = session.groupby(['id'])['action'].count().to_frame().rename(columns={'action':'num_of_action'}).reset_index()

    session_action_by_id = session.groupby(['id'])['action'].value_counts().unstack().fillna(0).astype(int).reset_index()
    session_action_by_id = session_action_by_id.rename(columns={'NULL':'action_NULL'})

    session_action_type_by_id = session.groupby(['id'])['action_type'].value_counts().unstack().fillna(0).astype(int).reset_index()
    session_action_type_by_id = session_action_type_by_id.rename(columns={'NULL':'action_type_NULL'})
    session_action_type_by_id = session_action_type_by_id.rename(columns={'-unknown-':'action_type_-unknown-'})

    session_action_detail_by_id = session.groupby(['id'])['action_detail'].value_counts().unstack().fillna(0).astype(int).reset_index()
    session_action_detail_by_id = session_action_detail_by_id.rename(columns={'NULL':'action_detail_NULL'})
    session_action_detail_by_id = session_action_detail_by_id.rename(columns={'-unknown-':'action_detail_-unknown-'})

    session_device_type_by_id = session.groupby(['id'])['device_type'].value_counts().unstack().fillna(0).astype(int).reset_index()
    session_device_type_by_id = session_device_type_by_id.rename(columns={'-unknown-':'device_type_-unknown-'})

    # create feature: the number of each options in that feature of each id, where each option is over 100000 occurrences in the session table
    # session_action_num = session.groupby(['action']).count().reset_index().rename(columns={'id':'num'})[['action','num']]
    # action_option = session_action_num[session_action_num['num'] >100000]['action'].to_list()
    # session_id_action = session[session['action'].apply(lambda x: True if (x in action_option) else False)]
    # session_action_by_id = session_id_action.groupby(['id'])['action'].value_counts().unstack().fillna(0).astype(int).reset_index()

    # session_action_detail_num = session.groupby(['action_detail']).count().reset_index().rename(columns={'id':'num'})[['action_detail','num']]
    # action_option = session_action_detail_num[session_action_detail_num['num'] >100000]['action_detail'].to_list()
    # action_option.remove('-unknown-')
    # action_option.remove('NULL')
    # session_id_action_detail = session[session['action_detail'].apply(lambda x: True if (x in action_option) else False)]
    # session_action_detail_by_id = session_id_action_detail.groupby(['id'])['action_detail'].value_counts().unstack().fillna(0).astype(int).reset_index()


    # merge session features
    results = pd.merge(session_by_id_time, session_num, how='left', on=['id'])
    results = pd.merge(results, session_action_by_id, how='left', on=['id'])
    results = pd.merge(results, session_action_type_by_id, how='left', on=['id'])
    results = pd.merge(results, session_action_detail_by_id, how='left', on=['id'])
    results = pd.merge(results, session_device_type_by_id, how='left', on=['id'])


    # use id to combine session data with X_train, X_test
    X_train = pd.merge(X_train, results, how='left', on=['id']).fillna(0, downcast='infer')
    X_test = pd.merge(X_test, results, how='left', on=['id']).fillna(0, downcast='infer')

    return X_train,X_test


def validate_models(X_validate,Y_validate):

    # validate models
    models = []

    log = LogisticRegression(random_state=0,n_jobs=-1)
    models.append(log)
    gnb = GaussianNB()
    models.append(gnb)
    # svc = SVC(random_state=0,probability=True,kernel='linear',verbose=1)
    # models.append(svc)
    rf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=1)
    models.append(rf)
    knc = KNeighborsClassifier(n_jobs=-1)
    models.append(knc)
    xgb = XGBClassifier(random_state=0,n_jobs=-1,verbosity=2)
    models.append(xgb)
    nn = MLPClassifier(random_state=0)
    models.append(nn)

    # use cv test to choose models
    sorted_models = sorted(models,key=lambda x:statistics.mean(cross_val_score(x,X_validate, Y_validate,cv=10,n_jobs=-1,verbose=1)),reverse=True)
    sorted_models_name = [x.__class__.__name__ for x in sorted_models]
    print('acc without tuning high -> low: '+str(sorted_models_name))

    pass


def feature_selection(X_validate,Y_validate):

    # feature selection
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,random_state=0,verbose=1,max_iter=5000).fit(X_validate, Y_validate)
    model = SelectFromModel(lsvc, prefit=True)

    cols_select_tmp = model.get_support()
    cols_select_idx = []
    for i,val in enumerate(cols_select_tmp):
        if val:
            cols_select_idx.append(i)
            pass
        pass
    print(cols_select_idx)
    return cols_select_tmp,cols_select_idx


def tune_hyper_parameters(param_grid,model,X_train,Y_train):

    # search optimal hyper-parameters for the model
    grid_cv = GridSearchCV(model,param_grid=param_grid, cv=10, verbose=1, n_jobs=-1)
    grid_cv.fit(X_train, Y_train)
    model_optimal = grid_cv.best_estimator_
    print(grid_cv.best_params_)
    # print('params: ',grid_cv.cv_results_['params'],', mean_test_score: ',grid_cv.cv_results_['mean_test_score'])

    return model_optimal


def run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year):

    # acc = statistics.mean(cross_val_score(model,X_train, Y_train,cv=10,n_jobs=-1,verbose=1,scoring=ndcg_scorer))
    # print(model.__class__.__name__+' ndcg score: '+str(acc))

    X_train_2,X_test_2,Y_train_2,Y_test_2 = train_test_split(X_train,Y_train,test_size=0.3,random_state=0)

    # ndcg score
    model.fit(X_train_2,Y_train_2)
    y_pred = model.predict_proba(X_test_2)
    score = ndcg.ndcg_score(Y_test_2,y_pred)
    print(model.__class__.__name__+' ndcg score: '+str(score))

    # auc score
    # model.fit(X_train_2,Y_train_2)
    # y_pred = model.predict(X_test_2)
    # fpr, tpr, thresholds = metrics.roc_curve(Y_test_2, y_pred, pos_label=2)
    # print(model.__class__.__name__+' AUC: '+str(metrics.auc(fpr, tpr)))

    model.fit(X_train,Y_train)

    result_num = model.predict(X_test)

    result = []
    for x in result_num:
        result.append(target_label[x])
        pass

    # output = pd.DataFrame({ 'id' : test_id, 'country': result})
    # output.to_csv('submission_'+model.__class__.__name__+'.csv', index = False)
    iv = pd.DataFrame({ 'id' : test_id, 'country': result, 'Year': test_year})
    iv.to_csv('iv.csv', index = False)

    # Taking the 5 classes with highest probabilities
    result_num = model.predict_proba(X_test)
    result = []
    ids=[]
    test_years = []
    for i in range(len(test_id)):
        cls_tmp = (np.argsort(result_num[i])[::-1])[:5].tolist()
        for tmp in cls_tmp:
            result.append(target_label[tmp])
            ids.append(test_id[i])
            test_years.append(test_year[i])
            pass
        pass

    output = pd.DataFrame({ 'id' : ids, 'country': result})
    output.to_csv('submission_'+model.__class__.__name__+'.csv', index = False)
    # iv = pd.DataFrame({ 'id' : ids, 'country': result, 'Year': test_years})
    # iv.to_csv('iv.csv', index = False)

    pass


if __name__ == '__main__':

    # load data
    training = pd.read_csv("train_users_2.csv")
    test = pd.read_csv("test_users.csv")
    # session = pd.read_csv("sessions.csv")
    # merge session to training and test by id
    # training = pd.concat([training,session],join="outer",axis=1,keys='id')
    # test = pd.concat([test,session],join="outer",axis=1,keys='id')

    X_train,Y_train,X_test,test_id,test_year,target_label,train_id,train_year = pre_processing(training,test)
    print(X_train.shape)

    X_tmp,X_validate,Y_tmp,Y_validate = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)

    # feature selection
    cols_select,cols_select_idx = feature_selection(X_validate,Y_validate)
    X_train = X_train[:,cols_select_idx]
    X_test = X_test[:,cols_select_idx]
    X_validate = X_validate[:,cols_select_idx]
    print(X_train.shape)
    print('finish feature selection')

    # validate_models(X_validate,Y_validate)
    # print('finish models validation')

    # GridSearchCV tune hyper-parameters for RF
    # model = RandomForestClassifier(random_state=0,n_jobs=-1)
    # param_grid = {
    #     'n_estimators': list(range(50,160,10)),
    #     'max_depth': list(range(1,30,2))
    # }
    # tune_hyper_parameters(param_grid,model,X_validate,Y_validate)

    # GridSearchCV tune hyper-parameters for KNN
    # model = KNeighborsClassifier(n_jobs=-1)
    # param_grid = {'n_neighbors':range(1,60,1)}
    # tune_hyper_parameters(param_grid,model,X_validate,Y_validate)

    # GridSearchCV tune hyper-parameters for XGBoost
    # model = XGBClassifier(random_state=0,n_jobs=-1,objective='multi:softprob',learning_rate=0.2,max_depth=6,n_estimators=35)
    # param_grid = {
    #     'n_estimators': list(range(50,110,10)),
    #     'max_depth': list(range(6,12,2))
    # }
    # tune_hyper_parameters(param_grid,model,X_validate,Y_validate)

    # KNN
    model = KNeighborsClassifier(n_jobs=-1,n_neighbors=58)
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # Logistic
    model = LogisticRegression(random_state=0,n_jobs=-1)
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # BN
    model = GaussianNB()
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # RF
    model = RandomForestClassifier(random_state=0,n_estimators=130,max_depth=21,n_jobs=-1,verbose=1)
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # NN
    model = MLPClassifier(random_state=0)
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # SVM
    # model = SVC(random_state=0,probability=True,kernel='linear',verbose=1)
    # run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    # XGBoost
    model = XGBClassifier(random_state=0,n_jobs=-1,objective='multi:softprob',learning_rate=0.2,max_depth=6,n_estimators=60)
    run_model(model,X_train,Y_train,X_test,target_label,test_id,test_year)

    pass




