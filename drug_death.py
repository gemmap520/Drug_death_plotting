import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#############################################################
# drug_death로 파일을 선언
drug_death = pd.read_csv('drug_deaths.csv')

drug_columns = [col for col in drug_death.columns if drug_death[col].dtype != 'O' and col not in ['Unnamed: 0', 'DateType', 'Age']]
drug_death['Date'] = pd.to_datetime(drug_death['Date'], errors='coerce')
drug_death['Year'] = drug_death['Date'].dt.year

unique_values = drug_death['ResidenceState'].unique()
print(unique_values)
#############################################################
# 1. Drug Death by certain drug: Drug vs. Number of Death
drug_counts = drug_death[drug_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
drug_counts.plot(kind='bar')
plt.title('Drug Death by certain drug')
plt.xlabel('Drug')
plt.ylabel('Number of Death')
plt.xticks(rotation=18, fontsize=7)

#############################################################
# 2. Drug Death By Year: Year vs. Number of Deaths
deaths_by_year = drug_death.groupby('Year').size()

plt.figure(figsize=(10, 6))  
deaths_by_year.plot(kind='line', marker='o') 
plt.title('Drug Death By Year')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True) 

#############################################################
# 3. Drug Deaths by Age Group: Age Group vs. Number of Deaths
bins = [0, 18, 30, 45, 60, 75, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-100']
drug_death['AgeGroup'] = pd.cut(drug_death['Age'], bins=bins, labels=labels, right=False)

deaths_by_agegroup = drug_death['AgeGroup'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
deaths_by_agegroup.plot(kind='bar')
plt.title('Drug Deaths by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=0)

#############################################################
# 4. Drug Death by Sex: Sex vs. Number of Deaths
deaths_by_sex = drug_death['Sex'].value_counts()

plt.figure(figsize=(10, 6))
deaths_by_sex.plot(kind='bar')
plt.title('Drug Deaths by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=0)

#############################################################
# 5. Drug Death by Race: Race vs. Number of Deaths
deaths_by_race = drug_death['Race'].value_counts()

plt.figure(figsize=(10, 6))
deaths_by_race.plot(kind='bar')
plt.title('Drug Deaths by Race')
plt.xlabel('Race', fontsize=9)
plt.ylabel('Number of Deaths')
plt.xticks(rotation=14, fontsize=6)

#############################################################
# 6-1. Drug Death by City: City vs. Number of Deaths
deaths_by_city = drug_death['ResidenceCity'].value_counts()

top_n = 8
deaths_by_city[:top_n].plot(kind='bar', figsize=(10, 6))
plt.title('Drug Death by City')
plt.xlabel('City')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=20, fontsize=7)

#############################################################
# 6-2. Drug Death by State: State vs. Number of Deaths
deaths_by_states = drug_death['ResidenceState'].value_counts()

top_n = 8
deaths_by_states[:top_n].plot(kind='bar', figsize=(10, 6))
plt.title('Drug Death by State')
plt.xlabel('State')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=20)

#############################################################
# 6-3. Drug Death by County: County vs. Number of Deaths
deaths_by_County = drug_death['ResidenceCounty'].value_counts()

top_n = 8
deaths_by_County[:top_n].plot(kind='bar', figsize=(10, 6))
plt.title('Drug Death by County')
plt.xlabel('County')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=20, fontsize=7)

#############################################################
# 7. Co-occurrence of Drugs: Heroin, Cocaine, Fentanyl_Analogue, Methadone
drug_death['Fentanyl_Analogue'] = drug_death['Fentanyl_Analogue'].astype(int)
drugs = ['Heroin', 'Cocaine', 'Fentanyl_Analogue', 'Methadone']
co_occurrence_matrix = drug_death[drugs].T.dot(drug_death[drugs])

plt.figure(figsize=(8, 6))
sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Co-occurrence of Drugs')

#############################################################
# 8-1. Drug Deaths by Year for Selected Drugs
selected_drugs = ['Heroin', 'Cocaine', 'Fentanyl_Analogue', 'Oxycodone', 'Ethanol',
                  'Methadone', 'Amphet', 'Tramad', 'Benzodiazepine', 'Hydrocodone']

yearly_drug_data = drug_death.groupby('Year')[selected_drugs].sum()

plt.figure(figsize=(14, 8))
for drug in selected_drugs:
    plt.plot(yearly_drug_data.index, yearly_drug_data[drug], label=drug)

plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Drug Deaths by Year for Selected Drugs')
plt.legend(title='Drugs')
plt.grid(True)

#############################################################
# 8-2. Heatmap: correlation matrix for the selected drugs
for drug in selected_drugs:
    drug_death[drug] = pd.to_numeric(drug_death[drug], errors='coerce').fillna(0)

drug_correlation = drug_death[selected_drugs].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(drug_correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Selected Drugs')
