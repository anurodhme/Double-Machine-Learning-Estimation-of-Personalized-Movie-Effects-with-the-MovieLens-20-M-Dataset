# MovieLens Dataset Analysis Project

A Python project for downloading and analyzing the MovieLens 20M dataset for movie recommendation and statistical analysis.

## Project Overview

This project provides tools to download, organize, and analyze the MovieLens 20M dataset. The dataset contains movie ratings, tags, and metadata that can be used for building recommendation systems, performing statistical analysis, and exploring user behavior patterns.

## Dataset Information

### MovieLens 20M Dataset

The MovieLens 20M dataset is one of the most popular datasets for recommendation system research and movie analysis. It contains:

- **20 million ratings** from 138,000 users on 27,000 movies
- **465,000 tag applications** applied to movies by users
- **Movie metadata** including titles, genres, and release years
- **Genome data** with tag relevance scores for movies

### Dataset Files

The dataset includes the following CSV files:

- `rating.csv` - User ratings for movies (userId, movieId, rating, timestamp)
- `movie.csv` - Movie information (movieId, title, genres)
- `tag.csv` - User-generated tags for movies (userId, movieId, tag, timestamp)
- `link.csv` - Links to external movie databases (movieId, imdbId, tmdbId)
- `genome_scores.csv` - Tag relevance scores for movies
- `genome_tags.csv` - Tag descriptions for the genome data

### Data Source

- **Original Source**: [GroupLens Research](https://grouplens.org/datasets/movielens/)
- **Kaggle Dataset**: [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- **License**: The dataset is provided for research and educational purposes

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Download the Dataset

Run the download script to automatically fetch and organize the dataset:

```bash
python download_dataset.py
```

This will:
1. Download the MovieLens 20M dataset from Kaggle
2. Create a `data/raw/` directory structure
3. Copy all dataset files to the appropriate location

### Project Structure

```
Stats project/
├── data/
│   └── raw/           # Raw dataset files (ignored by git)
├── download_dataset.py # Dataset download script
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── .gitignore        # Git ignore rules
```

## Usage

After downloading the dataset, you can start analyzing the data using your preferred Python data analysis tools (pandas, numpy, matplotlib, etc.).

## Dataset Statistics

- **Ratings**: 20,000,263 ratings
- **Users**: 138,493 users
- **Movies**: 27,278 movies
- **Tags**: 465,564 tag applications
- **Rating Scale**: 0.5 to 5.0 stars (in half-star increments)
- **Time Period**: Ratings from 1995 to 2015

## Citation

If you use this dataset in your research, please cite:

```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
```

## License

This project is for educational and research purposes. The MovieLens dataset is provided by GroupLens Research and is subject to their terms of use.
