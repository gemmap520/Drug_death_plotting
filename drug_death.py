import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#############################################################
# drug_death로 파일을 선언
drug_death = pd.read_csv('drug_deaths.csv')
print(drug_death.columns)
#############################################################
# Identifying drug-related columns by filtering out non-drug columns
drug_columns = [col for col in drug_death.columns if drug_death[col].dtype != 'O' and col not in ['Unnamed: 0', 'DateType', 'Age']]

# Summing up the occurrences of each drug
drug_counts = drug_death[drug_columns].sum().sort_values(ascending=False)

# Creating a bar plot for the drug counts
plt.figure(figsize=(10, 8))
drug_counts.plot(kind='bar')
plt.title('Number of Death from certain drug')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

#plt.show()
#############################################################
# Convert 'Date' to datetime format
drug_death['Date'] = pd.to_datetime(drug_death['Date'], errors='coerce')

# Extract year from 'Date' and create a new column 'Year'
drug_death['Year'] = drug_death['Date'].dt.year

# Aggregate deaths by year
deaths_by_year = drug_death.groupby('Year').size()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
deaths_by_year.plot(kind='line', marker='o')  # Line plot with markers
plt.title('Drug-Related Deaths Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True)  # Optional: Add grid for better readability
plt.tight_layout()  # Adjust layout to make room for the labels

#plt.show()
#############################################################
# Create age groups
bins = [0, 18, 30, 45, 60, 75, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-100']
drug_death['AgeGroup'] = pd.cut(drug_death['Age'], bins=bins, labels=labels, right=False)

# Plot 1 : Age Group vs. Number of Deaths
# Aggregate deaths by age group
deaths_by_agegroup = drug_death['AgeGroup'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
deaths_by_agegroup.plot(kind='bar')
plt.title('Drug-Related Deaths by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
#plt.show()

# Plot 2 : Sex vs. Number of Deaths
# Aggregate deaths by sex
deaths_by_sex = drug_death['Sex'].value_counts()

plt.figure(figsize=(10, 6))
deaths_by_sex.plot(kind='bar')
plt.title('Drug-Related Deaths by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=0)
#plt.show()

# Plot 3 : Race vs. Number of Deaths
# Aggregate deaths by race
deaths_by_race = drug_death['Race'].value_counts()

plt.figure(figsize=(10, 6))
deaths_by_race.plot(kind='bar')
plt.title('Drug-Related Deaths by Race')
plt.xlabel('Race')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
plt.show()
#############################################################
# Plot: Geographical Analysis
deaths_by_country = drug_death['DeathCounty'].value_counts()

# Simple bar plot for the top N cities
top_n = 10
deaths_by_country[:top_n].plot(kind='bar', figsize=(10, 6))
plt.title('Top N Counties by Drug-Related Deaths')
plt.xlabel('County')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
#plt.show()
#############################################################
# Plot: Co-occurrence of Drugs
# Convert 'Fentanyl' to numeric, coercing errors to NaN
drug_death['Fentanyl'] = pd.to_numeric(drug_death['Fentanyl'], errors='coerce')

# Fill NaN values with 0 (or another appropriate value)
drug_death['Fentanyl'] = drug_death['Fentanyl'].fillna(0)

# Convert to integer
drug_death['Fentanyl'] = drug_death['Fentanyl'].astype(int)

drugs = ['Heroin', 'Cocaine', 'Fentanyl', 'Methadone']
print(drug_death[drugs].dtypes)

# Calculate co-occurrence matrix
co_occurrence_matrix = drug_death[drugs].T.dot(drug_death[drugs])

# Plot the matrix
plt.figure(figsize=(8, 6))
sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Drug Co-occurrence Matrix')
#plt.show()

#############################################################
# Plot: Manner of Death Analysis
# Standardize the 'MannerofDeath' entries by converting to lower case and stripping whitespace
drug_death['MannerofDeath'] = drug_death['MannerofDeath'].str.lower().str.strip()

# Now calculate the counts of each manner of death again
manner_of_death_counts = drug_death['MannerofDeath'].value_counts()

# Plot the cleaned and standardized counts of each manner of death
plt.figure(figsize=(10, 6))
manner_of_death_counts.plot(kind='bar')
plt.title('Manner of Death in Drug-Related Deaths')
plt.xlabel('Manner of Death')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust the layout to ensure everything fits without overlap

# Show the plot
plt.show()

#############################################################
# Plot: Comparative Analysis Before and After Policy Changes
# Assuming 'Date' has been converted to datetime and 'Year' extracted
policy_change_year = 2018

before_policy_change = drug_death[drug_death['Year'] < policy_change_year]['Drug'].value_counts()
after_policy_change = drug_death[drug_death['Year'] >= policy_change_year]['Drug'].value_counts()

# Simple comparison plot (adjust based on specific drug or analysis focus)
plt.figure(figsize=(10, 6))
(before_policy_change - after_policy_change).plot(kind='bar')
plt.title('Change in Drug-Related Deaths Before and After Policy Change')
plt.xlabel('Drug')
plt.ylabel('Change in Number of Deaths')
plt.xticks(rotation=45)
plt.show()

#############################################################
# Plot: Fentanyl and Its Analogs Analysis
# Assuming 'Year' column exists and 'Fentanyl' indicator is present
fentanyl_deaths_by_year = drug_death[drug_death['Fentanyl'] == 1].groupby('Year').size()

fentanyl_deaths_by_year.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Fentanyl-Related Deaths Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True)
plt.show()

#############################################################
# Plot: Fentanyl and Its Analogs Analysis
# Simple comparison of deaths with and without Naloxone
naloxone_effectiveness = drug_death['NaloxoneAdministered'].value_counts()

naloxone_effectiveness.plot(kind='bar', figsize=(10, 6))
plt.title('Drug-Related Deaths with and without Naloxone Administration')
plt.xlabel('Naloxone Administered')
plt.ylabel('Number of Deaths')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.show()

