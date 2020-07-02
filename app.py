import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score



def main():
    def load_data():
          data = pd.read_csv("mushrooms.csv")
          label = LabelEncoder()
          for col in data.columns:
              data[col]=label.fit_transform(data[col])
          return data
    activities = ['EDA','Plots','Model Building','About']
    choice = st.sidebar.radio('Select Activities', activities)
    data = st.file_uploader("Upload a Dataset",type=['csv','txt'],encoding='latin-1')

#EDA    
    if choice == 'EDA':
      st.subheader("Exploratory Data Analytics")

      #data = st.file_uploader("Upload a Dataset",type=['csv','txt'])
      if data is not None:
        df = pd.read_csv(data)
        st.write(df.head())

      if st.checkbox('Show Shape'):
        st.write(df.shape)

      if st.checkbox('Show Columns'):
        all_columns = df.columns.tolist()
        st.write(df.columns.tolist())

      if st.checkbox('Summary'):
        st.write(df.describe())

      if st.checkbox('Show Selected Columns'):
        selected_columns = st.multiselect('Select Columns', all_columns)
        new_df = df[selected_columns]
        st.write(new_df)

      if st.checkbox('Show Value Counts'):
        st.write(df.iloc[:,-1].value_counts())

      if st.checkbox("Correlation Plot(Seaborn)"):
        try:
          st.write(sns.heatmap(df.corr(), annot = True))
          st.pyplot()
        except:
          st.write("This operation isn't available on the selected dataset")

      if st.checkbox("Pie Plot"):
        all_columns = df.columns.tolist()
        all_columns.insert(0,'')
        try:
          column_to_plot = st.selectbox("Select a Column", all_columns)
          if column_to_plot is not None:
            pie_plot = df[column_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
            st.write(pie_plot)
            st.pyplot()
        except:
          st.write("Select a column")

#MODEL
    

#PLOT
    
    elif choice == 'Plots':
              
      st.subheader("Data Visualization")
      def label_encode(le):
          LE =LabelEncoder()
          df[le] = LE.fit_transform(df[le])
          st.write(df.head())
      #data = st.file_uploader("Upload a Dataset", type = ["csv", "txt"])
      if data is not None:
        df = pd.read_csv(data)
        st.write(df.head())
      all_columns = df.columns.tolist()
      all_columns.insert(0,'')
      if st.checkbox("Encode Data"):
          le = st.selectbox("Select a column",all_columns)
          label_encode(le)
              
      
      if st.checkbox("Show Value Counts"):
        
        all_columns.insert(0,'')
        column_to_plot = st.selectbox("Select a Column", all_columns)
        pie_plot = df[column_to_plot].value_counts().plot(kind = "bar")
    
        #st.pyplot()
        #st.write(df.iloc[:,0].value_counts().plot(kind="bar"))
        st.pyplot()
      #st.write(df.iloc[:,1])
      all_columns_names = df.columns.tolist()
      all_columns_names.insert(0,'')
      type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
      selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

      if st.button("Generate Plot"):
          st.success("Generating Plot of {} for {}".format(type_of_plot,selected_columns_names))

          try:
              if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

              if type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

              if type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

              elif type_of_plot:
                cust_data = df[selected_columns_names]
                cust_plot = cust_data.plot(kind = type_of_plot)
                st.write(cust_plot)
                st.pyplot()

          except:
              st.write("[Error] Label Encoding required")
      
        

      

      
      
if __name__ == "__main__":
  main()











        
        






        
