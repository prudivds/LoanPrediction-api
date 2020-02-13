import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)
model = pickle.load(open('DecisionTree.pkl', 'rb'))

class PreProcessing(BaseEstimator, TransformerMixin):


    def __init__(self):
        pass

    
    ## Here this model will be called with a variable (df), which is a dataframe
    def datapreprocessing(self, df):
        
        df = df.reset_index(drop=True)
        
        df.loc[df['Gender'].isnull(),'Gender']='Trasgender'
        df.loc[df['Dependents'].isnull(),'Dependents']='0'
        df.loc[df['Education'].isnull(),'Education']='No Education'
        df.loc[df['Married'].isnull(),'Married']='No'
        df.loc[df['Self_Employed'].isnull(),'Self_Employed']='No'
        df.loc[df['Loan_Amount_Term'].isnull(),'Loan_Amount_Term']=0
        df.loc[df['Credit_History'].isnull(),'Credit_History']=0
        df.loc[df['LoanAmount'].isnull(),'LoanAmount']=146.867     
        df['Credit_History']=df['Credit_History'].astype(str)
        
        
        ## Creating a Instance for the minmax Pickle File
        MinMaxpickle_in = open("minmax_pickle.pkl","rb")
        MinmaxScaler_dict = pickle.load(MinMaxpickle_in)
        
        ## Applying the pickle file
        df['ApplicantIncome']=pd.DataFrame(MinmaxScaler_dict['ApplicantIncome'].transform(df[['ApplicantIncome']]))
        
        ## Creating a Instance for the minmax Pickle File
        standardscaler_in = open("standardscaler_pickle.pkl","rb")
        standardscaler_dict = pickle.load(standardscaler_in)
        
        ## Applying the pickle file
        df['LoanAmount']=pd.DataFrame(standardscaler_dict['LoanAmount'].transform(df[['LoanAmount']]))
        
        ## Creating a Instance for the LabelEncoder Pickle File
        Labelencoderpickle_in = open("labelencoder_pickle.pkl","rb")
        Labelencoder_dict = pickle.load(Labelencoderpickle_in)
        
        ## Applying the pickle file
        df['Education']=pd.DataFrame(Labelencoder_dict['Education'].transform(df[['Education']]))
        df['Property_Area']=pd.DataFrame(Labelencoder_dict['Property_Area'].transform(df[['Property_Area']]))
        df['Credit_History']=pd.DataFrame(Labelencoder_dict['Credit_History'].transform(df[['Credit_History']]))
        df['Dependents']=pd.DataFrame(Labelencoder_dict['Dependents'].transform(df[['Dependents']]))
        
        ## Creating a Instance for the binarizer Pickle File
        Onehotpickle_in = open("binarizer_pickle.pkl","rb")
        Onehot_dict = pickle.load(Onehotpickle_in)
        
        ## Applying the pickle file
        d1=pd.DataFrame(Onehot_dict['Self_Employed'].transform(df[['Self_Employed']]))
        d1.columns=['Self_Employed_0']
        d2=pd.DataFrame(Onehot_dict['Married'].transform(df[['Married']]))
        d2.columns=['Married_0']
        d3=pd.DataFrame(Onehot_dict['Gender'].transform(df[['Gender']]))
        d3.columns=['Gender_0','Gender_1','Gender_2']
    
        df=df.drop(['Self_Employed','Married','Gender'],axis=1)
        df=pd.concat([df,d1,d2,d3],axis=1)
        
        return df.as_matrix()

    def definingvalues(self, df, y=None,**fit_params):
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self
    
    def encodingTargetVariable(self,df):
        Labelencoderpickle_in = open("labelencoder_pickle.pkl","rb")
        Labelencoder_dict = pickle.load(Labelencoderpickle_in)
        
        ## Applying the pickle file
        df['Loan_Status']=pd.DataFrame(Labelencoder_dict['Loan_Status'].transform(df[['Loan_Status']]))
        
        return df.as_matrix()

preprocess = PreProcessing()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    df=pd.DataFrame(final_features,columns=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
    data_transformed_xtrain = preprocess.datapreprocessing(df)
    
    pred_var = ['Dependents', 'Education', 'ApplicantIncome',
       'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area', 'Self_Employed_0', 'Married_0', 'Gender_0', 'Gender_1','Gender_2']


    data_transformed_xtrain=pd.DataFrame(data_transformed_xtrain,columns=pred_var)
    
    prediction = model.predict(data_transformed_xtrain)

    output = round(prediction[0], 2)
    
    if output==1:
        output='Loan Approved'
    else:
        output='Loan Rejected'

    return render_template('index.html', prediction_text='LOAN STATUS (Yes/NO)  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)