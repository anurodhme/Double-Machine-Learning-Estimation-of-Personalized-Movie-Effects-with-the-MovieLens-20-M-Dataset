"""
Exploratory Data Analysis for MovieLens 20M Dataset
This script provides comprehensive analysis of the MovieLens dataset including:
- Basic dataset statistics
- Rating distributions
- User behavior patterns
- Movie popularity analysis
- Genre analysis
- Temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class MovieLensAnalyzer:
    def __init__(self, data_path="data/raw"):
        """Initialize the analyzer with data path."""
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.tags = None
        self.links = None
        
    def load_data(self):
        """Load all dataset files."""
        print("Loading MovieLens dataset...")
        
        # Load main datasets
        self.ratings = pd.read_csv(self.data_path / "rating.csv")
        self.movies = pd.read_csv(self.data_path / "movie.csv")
        self.tags = pd.read_csv(self.data_path / "tag.csv")
        self.links = pd.read_csv(self.data_path / "link.csv")
        
        # Convert timestamps (handle both Unix timestamps and datetime strings)
        try:
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        except ValueError:
            self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'])
        
        try:
            self.tags['timestamp'] = pd.to_datetime(self.tags['timestamp'], unit='s')
        except ValueError:
            self.tags['timestamp'] = pd.to_datetime(self.tags['timestamp'])
        
        print("✓ Data loaded successfully!")
        
    def basic_statistics(self):
        """Display basic dataset statistics."""
        print("\n" + "="*50)
        print("BASIC DATASET STATISTICS")
        print("="*50)
        
        print(f"Ratings Dataset:")
        print(f"  • Total ratings: {len(self.ratings):,}")
        print(f"  • Unique users: {self.ratings['userId'].nunique():,}")
        print(f"  • Unique movies: {self.ratings['movieId'].nunique():,}")
        print(f"  • Rating range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"  • Time period: {self.ratings['timestamp'].min().strftime('%Y-%m-%d')} to {self.ratings['timestamp'].max().strftime('%Y-%m-%d')}")
        
        print(f"\nMovies Dataset:")
        print(f"  • Total movies: {len(self.movies):,}")
        print(f"  • Movies with ratings: {self.movies['movieId'].isin(self.ratings['movieId']).sum():,}")
        
        print(f"\nTags Dataset:")
        print(f"  • Total tags: {len(self.tags):,}")
        print(f"  • Unique tags: {self.tags['tag'].nunique():,}")
        print(f"  • Users who tagged: {self.tags['userId'].nunique():,}")
        
        # Rating distribution
        print(f"\nRating Distribution:")
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = (count / len(self.ratings)) * 100
            print(f"  • {rating}: {count:,} ({percentage:.1f}%)")
    
    def analyze_user_behavior(self):
        """Analyze user rating patterns."""
        print("\n" + "="*50)
        print("USER BEHAVIOR ANALYSIS")
        print("="*50)
        
        # Ratings per user
        user_ratings = self.ratings.groupby('userId').size()
        
        print(f"Ratings per User:")
        print(f"  • Mean: {user_ratings.mean():.1f}")
        print(f"  • Median: {user_ratings.median():.1f}")
        print(f"  • Min: {user_ratings.min()}")
        print(f"  • Max: {user_ratings.max():,}")
        print(f"  • Users with 1 rating: {(user_ratings == 1).sum():,}")
        print(f"  • Users with 100+ ratings: {(user_ratings >= 100).sum():,}")
        print(f"  • Users with 1000+ ratings: {(user_ratings >= 1000).sum():,}")
        
        # User rating patterns
        user_avg_ratings = self.ratings.groupby('userId')['rating'].mean()
        print(f"\nUser Average Ratings:")
        print(f"  • Mean user avg rating: {user_avg_ratings.mean():.2f}")
        print(f"  • Std user avg rating: {user_avg_ratings.std():.2f}")
        
    def analyze_movie_popularity(self):
        """Analyze movie popularity and rating patterns."""
        print("\n" + "="*50)
        print("MOVIE POPULARITY ANALYSIS")
        print("="*50)
        
        # Ratings per movie
        movie_ratings = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        movie_ratings.columns = ['num_ratings', 'avg_rating', 'rating_std']
        
        print(f"Ratings per Movie:")
        print(f"  • Mean: {movie_ratings['num_ratings'].mean():.1f}")
        print(f"  • Median: {movie_ratings['num_ratings'].median():.1f}")
        print(f"  • Movies with 1 rating: {(movie_ratings['num_ratings'] == 1).sum():,}")
        print(f"  • Movies with 100+ ratings: {(movie_ratings['num_ratings'] >= 100).sum():,}")
        print(f"  • Movies with 1000+ ratings: {(movie_ratings['num_ratings'] >= 1000).sum():,}")
        
        # Most popular movies
        popular_movies = movie_ratings.nlargest(10, 'num_ratings')
        movie_titles = self.movies.set_index('movieId')['title']
        
        print(f"\nTop 10 Most Rated Movies:")
        for i, (movie_id, row) in enumerate(popular_movies.iterrows(), 1):
            title = movie_titles.get(movie_id, "Unknown")
            print(f"  {i:2d}. {title[:50]:<50} ({row['num_ratings']:,} ratings, avg: {row['avg_rating']:.1f})")
        
        # Highest rated movies (with minimum ratings)
        well_rated = movie_ratings[movie_ratings['num_ratings'] >= 100].nlargest(10, 'avg_rating')
        print(f"\nTop 10 Highest Rated Movies (100+ ratings):")
        for i, (movie_id, row) in enumerate(well_rated.iterrows(), 1):
            title = movie_titles.get(movie_id, "Unknown")
            print(f"  {i:2d}. {title[:50]:<50} (avg: {row['avg_rating']:.2f}, {row['num_ratings']:,} ratings)")
    
    def analyze_genres(self):
        """Analyze movie genres."""
        print("\n" + "="*50)
        print("GENRE ANALYSIS")
        print("="*50)
        
        # Extract all genres
        all_genres = []
        for genres in self.movies['genres'].dropna():
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"Genre Distribution (Top 15):")
        for i, (genre, count) in enumerate(genre_counts.head(15).items(), 1):
            percentage = (count / len(self.movies)) * 100
            print(f"  {i:2d}. {genre:<15} {count:,} movies ({percentage:.1f}%)")
        
        # Genre ratings analysis
        genre_ratings = {}
        for genre in genre_counts.head(10).index:
            genre_movies = self.movies[self.movies['genres'].str.contains(genre, na=False)]['movieId']
            genre_rating_data = self.ratings[self.ratings['movieId'].isin(genre_movies)]['rating']
            genre_ratings[genre] = {
                'avg_rating': genre_rating_data.mean(),
                'num_ratings': len(genre_rating_data),
                'movies': len(genre_movies)
            }
        
        print(f"\nTop 10 Genres - Rating Statistics:")
        print(f"{'Genre':<15} {'Avg Rating':<12} {'Total Ratings':<15} {'Movies':<10}")
        print("-" * 55)
        for genre, stats in sorted(genre_ratings.items(), key=lambda x: x[1]['avg_rating'], reverse=True):
            print(f"{genre:<15} {stats['avg_rating']:.2f}        {stats['num_ratings']:,}           {stats['movies']:,}")
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in ratings."""
        print("\n" + "="*50)
        print("TEMPORAL ANALYSIS")
        print("="*50)
        
        # Ratings by year
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        yearly_ratings = self.ratings.groupby('year').size()
        
        print(f"Rating Activity by Year:")
        print(f"  • Peak year: {yearly_ratings.idxmax()} ({yearly_ratings.max():,} ratings)")
        print(f"  • Most recent year: {yearly_ratings.index.max()} ({yearly_ratings.iloc[-1]:,} ratings)")
        print(f"  • Average ratings per year: {yearly_ratings.mean():.0f}")
        
        # Ratings by day of week
        self.ratings['day_of_week'] = self.ratings['timestamp'].dt.day_name()
        daily_ratings = self.ratings.groupby('day_of_week').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_ratings = daily_ratings.reindex(day_order)
        
        print(f"\nRating Activity by Day of Week:")
        for day, count in daily_ratings.items():
            percentage = (count / len(self.ratings)) * 100
            print(f"  • {day:<10} {count:,} ({percentage:.1f}%)")
    
    def create_visualizations(self):
        """Create key visualizations."""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Create figures directory
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Rating distribution
        plt.figure(figsize=(10, 6))
        self.ratings['rating'].hist(bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Ratings per user distribution
        plt.figure(figsize=(12, 6))
        user_ratings = self.ratings.groupby('userId').size()
        plt.hist(user_ratings, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Ratings per User', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'ratings_per_user.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Ratings over time
        plt.figure(figsize=(14, 8))
        monthly_ratings = self.ratings.set_index('timestamp').resample('M').size()
        monthly_ratings.plot(kind='line', color='green', linewidth=2)
        plt.title('Rating Activity Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Ratings per Month')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'ratings_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top genres
        all_genres = []
        for genres in self.movies['genres'].dropna():
            all_genres.extend(genres.split('|'))
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        genre_counts.plot(kind='barh', color='orange', alpha=0.8)
        plt.title('Top 15 Movie Genres by Count', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Movies')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(viz_dir / 'top_genres.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved to 'visualizations/' directory")
        print("  • rating_distribution.png")
        print("  • ratings_per_user.png") 
        print("  • ratings_over_time.png")
        print("  • top_genres.png")
    
    def run_full_analysis(self):
        """Run the complete exploratory data analysis."""
        print("🎬 MOVIELENS 20M DATASET - EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.load_data()
        self.basic_statistics()
        self.analyze_user_behavior()
        self.analyze_movie_popularity()
        self.analyze_genres()
        self.analyze_temporal_patterns()
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("Check the 'visualizations/' directory for charts and graphs.")
        print("="*60)

def main():
    """Main function to run the analysis."""
    analyzer = MovieLensAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
