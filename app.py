"""_summary_
Concrete Mix Optimizer Web Application
Summary
This web application, developed with Streamlit, assists in optimizing and predicting the properties of Engineered Cementitious Composites (ECC) mixtures. By utilizing machine learning models and multi-objective optimization algorithms, it aids in formulating mixtures that meet specific requirements in terms of strength, cost, and CO2 emissions.

Features
User Inputs: Enables users to specify both the bounds for generative design inputs (like Cement, Clinker, Slag, Fly Ash, etc.) and the fixed values for desired compressive strength, expected date, and the number of iterations for the optimization process.

Predictions: Facilitates making predictions using the current values of input variables with a straightforward user interface.

Visualizations: Once optimization is executed, the app provides visual insights through various plots, like pairwise relationships of objectives and a 3D scatter plot to visualize the trade-offs between all three objectives (strength deviation, CO2, and cost). A 3x3 subplot grid is also generated to visualize each input feature against cost, with points colored by strength and sized by CO2 emissions.

Downloadable Results: Users can download the results and visualizations for further analysis and reporting.

Technology Stack
Streamlit: For web application framework.
Pandas: For data manipulation.
NumPy: For numerical operations.
Matplotlib and Seaborn: For data visualization.
Scikit-learn: For data scaling.
Platypus: For multi-objective optimization.
Joblib: For loading pre-trained machine learning models.
Usage
The application can be run locally using Streamlit. Ensure to have all the necessary Python libraries installed and simply use the command:

shell
Copy code
streamlit run app.py
This command will open a new tab in your web browser with the app running.

Author
Khuong Le Nguyen
University of Transport Technology, Vietnam
University of Canberra, Australia
Email: lekhuong.jmi@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from platypus import NSGAII, Problem, Real

# Load the data
data_path = './Data_for_Opt.csv'
data = pd.read_csv(data_path)

# Define features and targets
features = data.drop(columns=['Age', 'Strength (MPa)', 'CO2 (kg)', 'Cost (USD)'])
targets = data[['Strength (MPa)', 'CO2 (kg)', 'Cost (USD)']]
data_description = data.describe()
min_max_values = data_description.loc[['min', 'max']].transpose()
expected_Strength=22
expected_date = 14
class ConcreteOptimizer:
    def __init__(self, models_path, min_max_values, feature_columns):
        self.models = self.load_models(models_path)
        self.min_max_values = min_max_values
        self.feature_columns = feature_columns

    @staticmethod
    def load_models(path):
        model_names = ['Strength', 'CO2', 'Cost']
        models = {name: joblib.load(os.path.join(path, f"RandomForest_{name}_model.pkl")) for name in model_names}
        return models

    def optimize_concrete_mix(self, x):
        cement, clinker, slag, flyash, limestones, gypsum, water, chem_admix, coarseagg, fineagg = x
        input_features = [cement, clinker, slag, flyash, limestones, gypsum, water, chem_admix, coarseagg, fineagg, expected_date]
        
        strength = self.models['Strength'].predict([input_features])[0]
        co2 = self.models['CO2'].predict([input_features])[0]
        cost = self.models['Cost'].predict([input_features])[0]

        objectives = [abs(strength - expected_Strength), co2, cost]
        constraints = [expected_Strength - strength]
        
        return objectives, constraints

    def run_optimization(self, iterations):
        problem = Problem(10, 3, 1)
        problem.types[:] = [Real(self.min_max_values.loc[feature, 'min'], self.min_max_values.loc[feature, 'max']) for feature in self.feature_columns]
        problem.function = self.optimize_concrete_mix
        problem.constraints[:] = "<=0"

        algorithm = NSGAII(problem)
        algorithm.run(iterations)

        optimal_solutions = np.array([[s.objectives[0], s.objectives[1], s.objectives[2]] for s in algorithm.result if s.feasible])
        scaler = MinMaxScaler()
        scaled_optimal_solutions = scaler.fit_transform(optimal_solutions)
        optimal_features = np.array([s.variables for s in algorithm.result if s.feasible])

        return optimal_solutions, scaled_optimal_solutions, optimal_features

def main():
    st.title("Concrete Mix Optimizer")

    with st.sidebar:
        st.header("Input Variables:")
        cement = st.slider("Cement", float(min_max_values.loc['Cement', 'min']), float(min_max_values.loc['Cement', 'max']), value=300.0)
        clinker = st.slider("Clinker", float(min_max_values.loc['Clinker', 'min']), float(min_max_values.loc['Clinker', 'max']), value=300.0)
        slag = st.slider("Slag", float(min_max_values.loc['Slag', 'min']), float(min_max_values.loc['Slag', 'max']), value=300.0)
        flyash = st.slider("FlyAsh", float(min_max_values.loc['FlyAsh', 'min']), float(min_max_values.loc['FlyAsh', 'max']), value=300.0)
        limestones = st.slider("Limestones", float(min_max_values.loc['Limestones', 'min']), float(min_max_values.loc['Limestones', 'max']), value=300.0)
        gypsum = st.slider("Gypsum", float(min_max_values.loc['Gypsum', 'min']), float(min_max_values.loc['Gypsum', 'max']), value=300.0)
        water = st.slider("Water", float(min_max_values.loc['Water', 'min']), float(min_max_values.loc['Water', 'max']), value=300.0)
        chem_admix = st.slider("Chemical Admixture", float(min_max_values.loc['Chemical Admixture', 'min']), float(min_max_values.loc['Chemical Admixture', 'max']), value=300.0)
        coarseagg = st.slider("Coarse Aggregate", float(min_max_values.loc['coarseagg', 'min']), float(min_max_values.loc['coarseagg', 'max']), value=300.0)
        fineagg = st.slider("Fine Aggregate", float(min_max_values.loc['fineagg', 'min']), float(min_max_values.loc['fineagg', 'max']), value=300.0)
        st.header("Optimization Parameters:")
        expected_Strength = st.number_input("Desired Compressive Strength (MPa)", min_value=0, max_value=100, value=22)
        expected_date = st.number_input("Expected Date (days)", min_value=1, max_value=100, value=14)
        iterations = st.number_input("Iterations", min_value=1, max_value=10000, value=1000)

    if st.button("Run Optimization"):
        optimizer = ConcreteOptimizer(models_path='./Models', min_max_values=min_max_values, feature_columns=features.columns)
        optimal_solutions, scaled_optimal_solutions, optimal_features = optimizer.run_optimization(iterations)

        st.subheader("Feature vs Cost Analysis:")
        
        # Set global fontsize
        plt.rcParams.update({'font.size': 16})

        # Creating 3x3 subplot
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        # fig.suptitle('Feature vs Cost Analysis')
        axs = axs.flatten()

        # Assume that optimal_features and optimal_solutions are available
        optimal_df = pd.DataFrame(optimal_features, columns=features.columns)
        optimal_df['Strength Deviation (MPa)'] = optimal_solutions[:, 0]
        optimal_df['CO2 (kg)'] = optimal_solutions[:, 1]
        optimal_df['Cost (USD)'] = optimal_solutions[:, 2]

        # Scatter plots of each feature vs cost, color=strength, size=CO2
        scaling_factor = 250  # adjust this as per your visual preference
        optimal_df['CO2_normalized'] = ((optimal_df['CO2 (kg)'] - optimal_df['CO2 (kg)'].min()) /
                                        (optimal_df['CO2 (kg)'].max() - optimal_df['CO2 (kg)'].min())) * scaling_factor

        for i, feature in enumerate(features.columns[:9]):  # Limiting to the first 9 features
            scatter = axs[i].scatter(optimal_df['Cost (USD)'], optimal_df[feature], c=optimal_df['Strength Deviation (MPa)'], cmap='viridis', s=optimal_df['CO2_normalized'], alpha=0.7)
            # axs[i].set_title(feature)
            axs[i].set_xlabel('Cost (USD)')
            axs[i].set_ylabel(feature)
            fig.colorbar(scatter, ax=axs[i], label='Strength Deviation (MPa)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
        
        st.subheader("Optimal Solutions:")
        st.write(pd.DataFrame(optimal_solutions, columns=['Strength Deviation (MPa)', 'CO2 (kg)', 'Cost (USD)']))

        st.subheader("Optimal Features:")
        st.write(pd.DataFrame(optimal_features, columns=features.columns))

        st.subheader("Scaled Optimal Solutions:")
        st.write(pd.DataFrame(scaled_optimal_solutions, columns=['Strength Deviation (MPa)', 'CO2 (kg)', 'Cost (USD)']))
        
        # fig, ax = plt.subplots(figsize=(12, 8))
        # sns.pairplot(pd.DataFrame(optimal_solutions, columns=['Strength Deviation (MPa)', 'CO2 (kg)', 'Cost (USD)']), diag_kind='kde')
        # st.pyplot(fig)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(optimal_solutions[:,1], optimal_solutions[:,2], optimal_solutions[:,0])
        ax.set_xlabel('CO2 (kg)')
        ax.set_ylabel('Cost (USD)')
        ax.set_zlabel('Strength Deviation (MPa)')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
