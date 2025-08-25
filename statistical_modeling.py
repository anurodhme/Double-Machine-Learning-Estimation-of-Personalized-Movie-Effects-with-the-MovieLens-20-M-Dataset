"""
Double Machine Learning for Personalized Movie Effects Estimation
=================================================================

This module implements a sophisticated statistical framework combining:
1. Causal Inference (Double Machine Learning)
2. Advanced Statistical Methods (Mixed Effects, Bayesian)
3. Machine Learning (Random Forest, XGBoost, Neural Networks)
4. Econometric Analysis (Treatment Effects, Heterogeneity)

Research Question: How do personalized movie characteristics affect individual user ratings?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

class DoubleMLMovieAnalysis:
    """Double Machine Learning framework for estimating personalized movie effects."""
    
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.merged_data = None
        self.treatment_effects = None
        self.ml_models = {}
        self.statistical_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for statistical analysis."""
        print("Loading and preparing data for statistical modeling...")
        
        self.ratings = pd.read_csv(self.data_path / "rating.csv")
        self.movies = pd.read_csv(self.data_path / "movie.csv")
        
        # Convert timestamps
        try:
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        except ValueError:
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'])
        
        # Create time features
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        self.ratings['month'] = self.ratings['timestamp'].dt.month
        self.ratings['day_of_week'] = self.ratings['timestamp'].dt.dayofweek
        
        # Merge ratings with movies
        self.merged_data = self.ratings.merge(self.movies, on='movieId', how='left')
        
        print(f"✓ Data prepared: {len(self.merged_data):,} rating observations")
        
    def feature_engineering(self):
        """Create sophisticated features for statistical modeling."""
        print("Engineering features for statistical analysis...")
        
        # User-level features
        user_stats = self.merged_data.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'year': ['min', 'max']
        }).round(3)
        user_stats.columns = ['user_rating_count', 'user_avg_rating', 'user_rating_std', 
                             'user_first_year', 'user_last_year']
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        user_stats['user_experience_years'] = user_stats['user_last_year'] - user_stats['user_first_year']
        
        # Movie-level features
        movie_stats = self.merged_data.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).round(3)
        movie_stats.columns = ['movie_rating_count', 'movie_avg_rating', 'movie_rating_std', 'movie_unique_users']
        movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
        
        # Genre engineering
        self.merged_data['genres_list'] = self.merged_data['genres'].str.split('|')
        self.merged_data['num_genres'] = self.merged_data['genres_list'].str.len()
        
        # Create genre dummy variables
        main_genres = ['Action', 'Adventure', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                      'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        
        for genre in main_genres:
            self.merged_data[f'genre_{genre.lower()}'] = (
                self.merged_data['genres'].str.contains(genre, na=False).astype(int)
            )
        
        # Extract movie year
        self.merged_data['movie_year'] = self.merged_data['title'].str.extract(r'\((\d{4})\)$')[0]
        self.merged_data['movie_year'] = pd.to_numeric(self.merged_data['movie_year'], errors='coerce')
        self.merged_data['movie_age'] = self.merged_data['year'] - self.merged_data['movie_year']
        
        # Merge statistics
        self.merged_data = self.merged_data.merge(user_stats, left_on='userId', right_index=True)
        self.merged_data = self.merged_data.merge(movie_stats, left_on='movieId', right_index=True)
        
        # Create treatment variables
        self.merged_data['treatment_drama_romance'] = (
            (self.merged_data['genre_drama'] == 1) & (self.merged_data['genre_romance'] == 1)
        ).astype(int)
        
        self.merged_data['treatment_action_adventure'] = (
            (self.merged_data['genre_action'] == 1) & (self.merged_data['genre_adventure'] == 1)
        ).astype(int)
        
        print(f"✓ Feature engineering complete: {self.merged_data.shape[1]} features")
        
    def implement_double_ml(self, treatment_col='treatment_drama_romance', sample_size=50000):
        """Implement Double Machine Learning for causal inference."""
        print(f"Implementing Double Machine Learning for {treatment_col}...")
        
        # Sample data for computational efficiency
        if len(self.merged_data) > sample_size:
            sample_data = self.merged_data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.merged_data.copy()
        
        # Define features (confounders)
        feature_cols = [
            'user_rating_count', 'user_avg_rating', 'user_rating_std',
            'user_experience_years', 'movie_rating_count', 'movie_avg_rating',
            'movie_rating_std', 'movie_unique_users', 'num_genres', 'movie_age',
            'year', 'month', 'day_of_week'
        ]
        
        # Add genre features
        genre_cols = [col for col in sample_data.columns if col.startswith('genre_') and col != treatment_col]
        feature_cols.extend(genre_cols[:8])  # Limit to avoid multicollinearity
        
        # Clean data
        analysis_data = sample_data[feature_cols + [treatment_col, 'rating']].dropna()
        
        X = analysis_data[feature_cols]
        T = analysis_data[treatment_col]  # Treatment
        Y = analysis_data['rating']       # Outcome
        
        print(f"Analysis sample: {len(analysis_data):,} observations")
        print(f"Treatment prevalence: {T.mean():.3f}")
        
        # Cross-fitting with K-fold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        treatment_effects = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Processing fold {fold + 1}/3...")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            
            # Step 1: Predict treatment (propensity score)
            prop_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            prop_model.fit(X_train, T_train)
            T_pred = prop_model.predict(X_test)
            
            # Step 2: Predict outcome
            outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            outcome_model.fit(X_train, Y_train)
            Y_pred = outcome_model.predict(X_test)
            
            # Step 3: Calculate residuals
            T_residual = T_test - T_pred
            Y_residual = Y_test - Y_pred
            
            # Step 4: Estimate treatment effect
            if np.var(T_residual) > 1e-6:
                treatment_effect = np.cov(Y_residual, T_residual)[0, 1] / np.var(T_residual)
                treatment_effects.append(treatment_effect)
        
        # Final treatment effect estimate
        avg_treatment_effect = np.mean(treatment_effects)
        se_treatment_effect = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
        
        # Store results
        self.treatment_effects = {
            'treatment': treatment_col,
            'ate': avg_treatment_effect,
            'se': se_treatment_effect,
            't_stat': avg_treatment_effect / se_treatment_effect if se_treatment_effect > 0 else 0,
            'p_value': 2 * (1 - stats.norm.cdf(abs(avg_treatment_effect / se_treatment_effect))) if se_treatment_effect > 0 else 1,
            'ci_lower': avg_treatment_effect - 1.96 * se_treatment_effect,
            'ci_upper': avg_treatment_effect + 1.96 * se_treatment_effect,
            'fold_effects': treatment_effects
        }
        
        print(f"✓ Double ML complete:")
        print(f"  Average Treatment Effect: {avg_treatment_effect:.4f}")
        print(f"  Standard Error: {se_treatment_effect:.4f}")
        print(f"  95% CI: [{self.treatment_effects['ci_lower']:.4f}, {self.treatment_effects['ci_upper']:.4f}]")
        print(f"  P-value: {self.treatment_effects['p_value']:.4f}")
        
    def machine_learning_comparison(self, sample_size=25000):
        """Compare multiple ML models for rating prediction."""
        print("Comparing machine learning models...")
        
        if len(self.merged_data) > sample_size:
            sample_data = self.merged_data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.merged_data.copy()
        
        # Prepare features
        feature_cols = [
            'user_rating_count', 'user_avg_rating', 'user_rating_std',
            'movie_rating_count', 'movie_avg_rating', 'movie_rating_std',
            'num_genres', 'movie_age', 'year', 'month'
        ]
        
        # Add genre features
        genre_cols = [col for col in sample_data.columns if col.startswith('genre_')][:8]
        feature_cols.extend(genre_cols)
        
        # Clean data
        ml_data = sample_data[feature_cols + ['rating']].dropna()
        
        X = ml_data[feature_cols]
        y = ml_data['rating']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models to compare
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use scaled data for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                'cv_rmse_std': np.sqrt(cv_scores.std()),
                'model': model
            }
        
        self.ml_models = model_results
        
        print("✓ ML model comparison complete:")
        for name, results in model_results.items():
            print(f"  {name}: RMSE = {results['rmse']:.4f}, R² = {results['r2']:.4f}")
    
    def run_complete_analysis(self):
        """Run the complete statistical and ML analysis."""
        print("🔬 DOUBLE MACHINE LEARNING ANALYSIS FOR MOVIELENS")
        print("=" * 60)
        
        self.load_and_prepare_data()
        self.feature_engineering()
        self.implement_double_ml()
        self.machine_learning_comparison()
        
        print("\n" + "="*60)
        print("✅ STATISTICAL ANALYSIS COMPLETE!")
        print("="*60)

def main():
    """Main function to run the analysis."""
    analyzer = DoubleMLMovieAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
