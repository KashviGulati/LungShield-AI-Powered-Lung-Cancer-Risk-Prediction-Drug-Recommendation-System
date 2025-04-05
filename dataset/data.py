
import pandas as pd

survey_path = "D:\\Cancer risk prediction\\survey lung cancer.csv"
drug_path = "D:\\Cancer risk prediction\\Cell_Lines_Details.xlsx"

survey_df = pd.read_csv(survey_path)
drug_df = pd.read_excel(drug_path)


survey_df['LUNG_CANCER'] = survey_df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})


survey_df['GENDER'] = survey_df['GENDER'].map({'M': 1, 'F': 0})


lung_cancer_drugs = drug_df[drug_df["Cancer Type\n(matching TCGA label)"] == "LUAD"]  
lung_cancer_drugs = lung_cancer_drugs[['Sample Name', 'GDSC\nTissue descriptor 1', 'Cancer Type\n(matching TCGA label)']]



import numpy as np

lung_cancer_patients = survey_df[survey_df['LUNG_CANCER'] == 1].copy()

lung_cancer_patients['Recommended Drug'] = np.random.choice(lung_cancer_drugs['Sample Name'].values, size=len(lung_cancer_patients))


merged_df = survey_df.merge(lung_cancer_patients[['AGE', 'Recommended Drug']], on='AGE', how='left')

merged_df.to_csv("Integrated_Lung_Cancer_Data.csv", index=False)

print("Dataset successfully processed and saved as 'Integrated_Lung_Cancer_Data.csv'")
