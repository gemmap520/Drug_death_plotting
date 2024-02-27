import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pydeck as pdk

#############################################################
# Load drug_deaths
def load_data():
    data = pd.read_csv('drug_deaths.csv')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    return data

drug_death = load_data()

#############################################################
# Filter by Drugs and Year
drug_list = ['Heroin', 'Cocaine', 'Fentanyl_Analogue', 'Oxycodone', 'Oxymorphone', 
                    'Ethanol', 'Hydrocodone', 'Benzodiazepine', 'Methadone', 'Amphet', 
                    'Tramad', 'Hydromorphone']

selected_drugs = st.sidebar.multiselect('Select drugs:', drug_list)
selected_years = st.sidebar.slider('Select year range:', 
                                   int(drug_death['Year'].min()), 
                                   int(drug_death['Year'].max()), 
                                   (int(drug_death['Year'].min()), 
                                    int(drug_death['Year'].max())))
drug_death_filtered = drug_death[(drug_death['Year'] >= selected_years[0]) & (drug_death['Year'] <= selected_years[1])]

#############################################################
# DASHBOARD
st.title('Drug Deaths Dashboard')
#############################################################
# Define plotting functions
#############################################################
# 1. Drug Death by certain drug: Drug vs. Number of Death
def plot_death_by_drug():
    st.header('Drug Deaths by Certain Drugs')
    if selected_drugs:
        death_by_drug = drug_death_filtered[selected_drugs].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        death_by_drug.plot(kind='bar')
        plt.title('Drug Death by Certain Drug')
        plt.xlabel('Drug')
        plt.ylabel('Number of Death')
        plt.xticks(rotation=45)
        st.pyplot(plt)

#############################################################
# 2. Drug Death By Year: Year vs. Number of Deaths
def plot_death_by_year():
    st.header('Drug Death By Year')
    if selected_drugs:
        deaths_by_year = drug_death.groupby('Year').size()
        plt.figure(figsize=(10, 6))  
        deaths_by_year.plot(kind='line', marker='o') 
        plt.title('Drug Death By Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Deaths')
        plt.grid(True) 
        st.pyplot(plt)

#############################################################
# 3. Drug Deaths by Age Group: Age Group vs. Number of Deaths
def plot_deaths_by_age_group():
    st.header('Drug Deaths by Age Group')
    if selected_drugs:
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
        st.pyplot(plt)

#############################################################
# 4. Drug Death by Sex: Sex vs. Number of Deaths
def plot_deaths_by_sex():
    st.header('Drug Deaths by Sex')
    if selected_drugs:
        deaths_by_sex = drug_death['Sex'].value_counts()

        plt.figure(figsize=(10, 6))
        deaths_by_sex.plot(kind='bar')
        plt.title('Drug Deaths by Sex')
        plt.xlabel('Sex')
        plt.ylabel('Number of Deaths')
        plt.xticks(rotation=0)
        st.pyplot(plt)

#############################################################
# 5. Drug Death by Race: Race vs. Number of Deaths
def plot_deaths_by_race():
    st.header('Drug Deaths by Race')
    if selected_drugs:
        deaths_by_race = drug_death['Race'].value_counts()

        plt.figure(figsize=(10, 6))
        deaths_by_race.plot(kind='bar')
        plt.title('Drug Deaths by Race')
        plt.xlabel('Race', fontsize=9)
        plt.ylabel('Number of Deaths')
        plt.xticks(rotation=14, fontsize=6)
        st.pyplot(plt)

#############################################################
# 6-1. Drug Death by City: City vs. Number of Deaths
def plot_deaths_by_city():
    st.header('Drug Deaths by (Residence)City')
    if selected_drugs:
        deaths_by_city = drug_death['ResidenceCity'].value_counts()

        top_n = 8
        deaths_by_city[:top_n].plot(kind='bar', figsize=(10, 6))
        plt.title('Drug Death by City')
        plt.xlabel('City')
        plt.ylabel('Number of Deaths')
        plt.xticks(rotation=20, fontsize=7)
        st.pyplot(plt)

#############################################################
# 6-2. Drug Death by State: State vs. Number of Deaths
def plot_deaths_by_state():
    st.header('Drug Deaths by (Residence)State')
    if selected_drugs:
        deaths_by_states = drug_death['ResidenceState'].value_counts()

        top_n = 8
        deaths_by_states[:top_n].plot(kind='bar', figsize=(10, 6))
        plt.title('Drug Death by State')
        plt.xlabel('State')
        plt.ylabel('Number of Deaths')
        plt.xticks(rotation=20)
        st.pyplot(plt)

#############################################################
# 6-3. Drug Death by County: County vs. Number of Deaths
def plot_deaths_by_county():
    st.header('Drug Deaths by (Residence)County')
    if selected_drugs:
        deaths_by_County = drug_death['ResidenceCounty'].value_counts()

        top_n = 8
        deaths_by_County[:top_n].plot(kind='bar', figsize=(10, 6))
        plt.title('Drug Death by County')
        plt.xlabel('County')
        plt.ylabel('Number of Deaths')
        plt.xticks(rotation=20, fontsize=7)
        st.pyplot(plt)

#############################################################
# 7. Co-occurrence of Drugs: Heroin, Cocaine, Fentanyl_Analogue, Methadone
def plot_cooccurence_of_drug():
    st.header('Co-occurrence of Drugs')
    if selected_drugs:     
        # Convert float to integer
        drug_death['Fentanyl_Analogue'] = drug_death['Fentanyl_Analogue'].astype(int)
        drugs = ['Heroin', 'Cocaine', 'Fentanyl_Analogue', 'Methadone']
        
        co_occurrence_matrix = drug_death[drugs].T.dot(drug_death[drugs])

        # Plot the matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.title('Co-occurrence of Drugs')
        st.pyplot(plt)

#############################################################
# 8-1. Drug Deaths by Year for Selected Drugs
def plot_death_by_year_for_drugs():
    st.header('Drug Deaths by Year for Selected Drugs')
    if selected_drugs:     
        yearly_drug_data = drug_death.groupby('Year')[selected_drugs].sum()
        
        plt.figure(figsize=(14, 8))
        for drug in selected_drugs:
            plt.plot(yearly_drug_data.index, yearly_drug_data[drug], label=drug)

        plt.xlabel('Year')
        plt.ylabel('Number of Deaths')
        plt.title('Drug Deaths by Year for Selected Drugs')
        plt.legend(title='Drugs')
        plt.grid(True)
        st.pyplot(plt)

#############################################################
# 8-2. Heatmap: correlation matrix for the selected drugs
def plot_heatmap_for_drugs():
    st.header('Correlation Matrix of Selected Drugs')
    if selected_drugs:   
        for drug in selected_drugs:
            drug_death[drug] = pd.to_numeric(drug_death[drug], errors='coerce').fillna(0)

        drug_correlation = drug_death[selected_drugs].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(drug_correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Selected Drugs')
        st.pyplot(plt)

#############################################################
# Drug Deaths by State with PyDeck
death_counts = drug_death['ResidenceState'].value_counts()

state_coordinates = {
    'CT': [41.6032, -73.0877], 'NY': [40.7128, -74.0060], 'PA': [41.2033, -77.1945], 
    'MA': [42.4072, -71.3824], 'FL': [27.9944, -81.7603], 'TN': [35.5175, -86.5804], 
    'GA': [32.1656, -82.9001], 'CA': [36.7783, -119.4179], 'ME': [45.2538, -69.4455], 
    'OK': [35.4676, -97.5164], 'VT': [44.5588, -72.5778], 'MI': [44.3148, -85.6024], 
    'RI': [41.5801, -71.4774], 'NH': [43.1939, -71.5724], 'NJ': [40.0583, -74.4057], 
    'SD': [43.9695, -99.9018], 'OH': [40.4173, -82.9071], 'IL': [40.6331, -89.3985], 
    'SC': [33.8361, -81.1637], 'TX': [31.9686, -99.9018], 'AL': [32.3182, -86.9023], 
    'NC': [35.7596, -79.0193], 'MD': [39.0458, -76.6413], 'CO': [39.5501, -105.7821], 
    'LA': [30.9843, -91.9623], 'MN': [46.7296, -94.6859],
}
state_deaths_mapped = pd.DataFrame([
    (state, deaths, *state_coordinates[state])
    for state, deaths in death_counts.items() if state in state_coordinates
], columns=['State', 'NumberOfDeaths', 'Latitude', 'Longitude'])
state_deaths_mapped['Radius'] = state_deaths_mapped['NumberOfDeaths'].apply(lambda x: max(min(x * 1000, 60000), 40000))

def plot_deaths_by_state_pydeck():
    st.header('Drug Deaths by State with PyDeck')

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=state_deaths_mapped,
        get_position='[Longitude, Latitude]',
        get_color='[200, 30, 0, 160]',  # RGBA color format
        get_radius='Radius',  # Use the 'Radius' column for circle size
    )
    view_state = pdk.ViewState(
        latitude=state_deaths_mapped['Latitude'].mean(),
        longitude=state_deaths_mapped['Longitude'].mean(),
        zoom=4,
        pitch=0,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
    )
    st.pydeck_chart(r)

#############################################################
# Define visualization options
visualization_options = {
    'Drug Death by Drug': plot_death_by_drug,
    'Drug Death by Year': plot_death_by_year,
    'Drug Deaths by Age Group': plot_deaths_by_age_group,
    'Drug Deaths by Sex': plot_deaths_by_sex,
    'Drug Deaths by Race': plot_deaths_by_race,
    'Drug Deaths by (Residence)City': plot_deaths_by_city,
    'Drug Deaths by (Residence)State': plot_deaths_by_state,
    'Drug Deaths by (Residence)County': plot_deaths_by_county,
    'Co-occurrence of Drugs': plot_cooccurence_of_drug,
    'Drug Deaths by Year for Selected Drugs': plot_death_by_year_for_drugs,
    'Correlation Matrix of Selected Drugs': plot_heatmap_for_drugs,
}

selected_visualization = st.sidebar.selectbox('Choose a visualization:', list(visualization_options.keys()))

if selected_visualization == 'Drug Death by Drug':
    plot_death_by_drug()
elif selected_visualization == 'Drug Death by Year':
    plot_death_by_year()
elif selected_visualization == 'Drug Deaths by Age Group':
    plot_deaths_by_age_group()
elif selected_visualization == 'Drug Deaths by Sex': 
    plot_deaths_by_sex()
elif selected_visualization == 'Drug Deaths by Race': 
    plot_deaths_by_race()
elif selected_visualization == 'Drug Deaths by (Residence)City': 
    plot_deaths_by_city()
elif selected_visualization == 'Drug Deaths by (Residence)State': 
    plot_deaths_by_state()
    plot_deaths_by_state_pydeck()
elif selected_visualization == 'Drug Deaths by (Residence)County': 
    plot_deaths_by_county()
elif selected_visualization == 'Co-occurrence of Drugs': 
    plot_cooccurence_of_drug()
elif selected_visualization == 'Drug Deaths by Year for Selected Drugs': 
    plot_death_by_year_for_drugs()
elif selected_visualization == 'Correlation Matrix of Selected Drugs': 
    plot_heatmap_for_drugs()