# NBA Game Prediction Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdoulayeSeydi/nba-predictor/blob/main/NBA_Predictor_Development.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdoulayeSeydi/nba-predictor/blob/main/NBA_Prediction_Summary.ipynb)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-complete-success.svg)](https://github.com/AbdoulayeSeydi/nba-game-predictor)

## Overview

This project builds a machine learning system to predict NBA game outcomes using historical team performance data. The model uses proper time-series validation, rolling statistics, and ensemble methods to achieve reliable predictions while avoiding common pitfalls like data leakage.

**What this project does:**
- Predicts NBA game winners using team performance metrics
- Analyzes the impact of rest, travel, and schedule factors on outcomes
- Tests whether ensemble modeling improves prediction accuracy
- Computes calibration metrics to ensure probability estimates are reliable

**What this project does NOT do:**
- Predict player performance or individual statistics
- Account for injuries, trades, or roster changes at the player level
- Make betting recommendations
- Claim to predict upsets or "sure things"

## Key Findings

### Overall Model Performance (Test Set: 2023-24 Season)
- **Accuracy**: 68.5% (843/1,230 games correct)
- **Log Loss**: 0.5847 (lower is better)
- **Brier Score**: 0.2134 (lower is better)
- **Status**: Outperforms Vegas implied probabilities by 2.3%

### Model Comparison
| Model | Accuracy | Log Loss | Brier Score |
|-------|----------|----------|-------------|
| Logistic Regression | 67.2% | 0.6012 | 0.2201 |
| XGBoost | 68.1% | 0.5891 | 0.2156 |
| **Ensemble (50/50)** | **68.5%** | **0.5847** | **0.2134** |

**Change**: Ensemble improves accuracy by +1.3% over baseline logistic regression

### Feature Importance Analysis
**Top 5 Most Predictive Features:**
1. **Net Rating (L10)**: +0.42 coefficient (dominant predictor)
2. **Home Court Advantage**: +0.18 coefficient
3. **Opponent Net Rating (L10)**: -0.38 coefficient
4. **Rest Advantage**: +0.09 coefficient
5. **Win Streak (EMA)**: +0.06 coefficient

**Finding**: Team performance metrics (net rating, win %, offensive/defensive efficiency) far outweigh schedule factors (rest, travel, back-to-backs).

### Calibration Analysis
- **Expected Calibration Error (ECE)**: 0.0423
- **Status**: Well-calibrated (predictions match actual outcomes)
- When model predicts 70% win probability, teams win ~68-72% of the time

## Project Structure

```
nba-predictor/
├── NBA_Predictor_Development.ipynb    # Model development & validation
├── NBA_Prediction_Summary.ipynb       # Final results & visualizations
└── README.md                          # This file
```

## Notebook Sections

### NBA_Predictor_Development.ipynb

**Part 1: Data Acquisition & Processing**
- Load 9,798 NBA games from 2015-2024 seasons
- Compute rolling statistics with proper time-series handling:
  - L5 and L10 averages (points, rebounds, assists, efficiency)
  - Exponential moving averages (α=0.95 decay factor)
  - Win streaks and recent form metrics
- Create schedule-based features:
  - Days of rest (capped at 7)
  - Back-to-back games (home and away)
  - Games in last 7 days
- Season normalization to prevent era effects

**Part 2: Exploratory Analysis**
- Home court advantage analysis (56.3% win rate)
- Rest impact visualization (back-to-backs reduce win % by 5.2%)
- Correlation analysis between features and outcomes
- Distribution analysis of key predictors

**Part 3: Model Development**
- Time-based train/test split:
  - Training: 2015-16 to 2022-23 (8,568 games)
  - Testing: 2023-24 season (1,230 games)
- Logistic regression baseline (L2 regularization, C=0.1)
- XGBoost model (200 estimators, max_depth=5, learning_rate=0.05)
- Ensemble optimization (50/50 weight split)

**Part 4: Validation & Testing**
- Rolling window validation on 2021-22, 2022-23, 2023-24
- Calibration curve analysis
- Confusion matrix and classification metrics
- Feature importance extraction
- Threshold sensitivity testing

### NBA_Prediction_Summary.ipynb

**Part 1: Model Performance Dashboard**
- Accuracy, log loss, and Brier score by season
- Calibration curves with reliability diagrams
- Prediction confidence distributions

**Part 2: Feature Analysis**
- Top 10 feature importance rankings
- Partial dependence plots for key features
- SHAP value analysis (if computed)

**Part 3: Error Analysis**
- Largest prediction errors (upsets missed)
- Home vs. away prediction accuracy
- Strong team vs. weak team prediction accuracy
- Rest advantage impact on prediction errors

**Part 4: Business Insights**
- Betting market comparison (model vs. Vegas)
- ROI simulation on hypothetical $100 bets
- Confidence threshold recommendations

## Methodology

### 1. Data Sources
- **Games**: 9,798 NBA regular season games (2015-2024)
- **Features**: Team box scores aggregated to game level
- **Source**: Basketball-Reference.com data (scraped or CSV export)
- **Note**: No player-level data; all predictions are team-based

### 2. Time-Series Design (Critical for Valid Results)
**Train/Test Split:**
- Training: Seasons 2015-16 through 2022-23
- Testing: Season 2023-24 (never seen during training)
- **No random shuffling**: Future games never leak into past

**Rolling Features:**
```python
# Correct: Only use past games
df['L10_pts'] = df.groupby('team')['pts'].shift(1).rolling(10).mean()

# Wrong: Would leak future into past
df['L10_pts'] = df.groupby('team')['pts'].rolling(10).mean()
```

**Season Boundaries:**
- Features reset at the start of each season
- No cross-season rolling windows (October games don't use June stats)

### 3. Feature Engineering

**Team Performance Metrics (Rolling Windows):**
- Points, rebounds, assists, field goal %, 3-point %, free throw %
- Offensive rating, defensive rating, net rating
- Pace (possessions per 48 minutes)
- Win percentage and win streaks

**Schedule Context:**
- Days of rest (0 = back-to-back, 1 = one day off, etc.)
- Back-to-back indicator (home and away separately)
- Games played in last 7 days (fatigue proxy)
- Rest advantage (home_rest - away_rest)

**Exponential Moving Averages:**
- Recent games weighted higher: `EMA_t = α × value_t + (1-α) × EMA_{t-1}`
- Optimal α = 0.95 (determined via validation)

**Season Normalization:**
```python
# Prevents 2015 stats from being "lower" than 2024 stats
df['net_rating_norm'] = df.groupby('season')['net_rating'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### 4. Models

**Logistic Regression (Baseline):**
- Linear model with L2 regularization (C=0.1)
- Interpretable coefficients
- Fast inference

**XGBoost (Advanced):**
- Gradient boosting with 200 trees
- Max depth = 5 (prevents overfitting)
- Learning rate = 0.05 (gradual learning)
- Captures non-linear interactions

**Ensemble (Final Model):**
- 50% Logistic Regression + 50% XGBoost
- Weights optimized on validation set (2022-23 season)
- Combines interpretability and flexibility

### 5. Evaluation Metrics

**Accuracy:**
- Percentage of correct predictions
- Baseline: 56.3% (home team always wins)
- Target: > 65%

**Log Loss:**
- Measures probability calibration
- Penalizes confident wrong predictions heavily
- Lower is better

**Brier Score:**
- Mean squared error of predicted probabilities
- Range: 0 (perfect) to 1 (worst)
- Lower is better

**Calibration:**
- Expected Calibration Error (ECE)
- Reliability diagrams (predicted vs. actual win rates)
- Target: Predictions match reality

### 6. Validation Strategy

**Holdout Testing:**
- Final test set: 2023-24 season (1,230 games)
- Never used during development

**Rolling Window Validation:**
- Train on 2015-2021, validate on 2021-22
- Train on 2015-2022, validate on 2022-23
- Train on 2015-2023, test on 2023-24
- Ensures robustness across different seasons

**Cross-Validation NOT Used:**
- Time-series data violates i.i.d. assumption
- K-fold would leak future into past

## Technologies Used

- **Python 3.x**
- **pandas, numpy**: Data manipulation and feature engineering
- **scikit-learn**: Logistic regression, preprocessing, metrics
- **XGBoost**: Gradient boosting implementation
- **matplotlib, seaborn**: Visualization
- **Google Colab**: Development environment

## How to Run

1. **Open `NBA_Predictor_Development.ipynb` in Google Colab**
2. **Upload your NBA games dataset** (CSV with columns: date, home_team, away_team, home_pts, away_pts, etc.)
3. **Run cells sequentially** (Runtime → Run all)
   - Data loading: ~2 minutes
   - Feature engineering: ~5 minutes
   - Model training: ~3 minutes
   - Evaluation: ~2 minutes
4. **Open `NBA_Prediction_Summary.ipynb`** for final visualizations
5. **Total runtime**: ~15-20 minutes

## Results Summary

### Statistical Tests
- **McNemar's Test** (LR vs. XGBoost): p < 0.05 (models significantly different)
- **Hosmer-Lemeshow Test** (calibration): p = 0.24 (well-calibrated, p > 0.05 desired)

### Threshold Sensitivity
Tested at 50%, 55%, 60%, 65%, 70% confidence thresholds:
- **Accuracy increases** as confidence threshold rises (70% threshold → 76% accuracy)
- **Coverage decreases** (70% threshold → only 40% of games predicted)
- **Optimal threshold**: 55% (balances accuracy and coverage)

### Feature Ablation Study
Removing feature groups and measuring impact:

| Feature Group Removed | Accuracy Drop |
|-----------------------|---------------|
| Net Rating (L10) | -4.2% |
| Home Court Advantage | -2.1% |
| Rest Features | -0.8% |
| Win Streaks | -0.5% |

**Conclusion**: Net rating is by far the most important feature.

## Interpretation

✅ **Model achieves 68.5% accuracy**, beating baseline by 12 percentage points  
✅ **Ensemble outperforms individual models** (log loss improvement: 2.7%)  
✅ **Well-calibrated predictions**: 70% confidence → ~70% actual win rate  
✅ **Robust across seasons**: Performance stable in 2021-22, 2022-23, 2023-24  
✅ **Net rating dominates**: Single best predictor (42% relative importance)  

⚠️ **Schedule factors matter but are weak**: Rest adds only 0.8% accuracy  
⚠️ **Home court advantage real but small**: ~56% win rate, not 60%+  
⚠️ **Model struggles with upsets**: Underdogs predicted correctly only 42% of the time  

## Limitations

### Data Limitations
- **No player-level data**: Cannot account for injuries, rotations, or lineup changes
- **No travel distance**: Missing timezone/distance traveled features
- **Aggregated box scores only**: No play-by-play or possession-level data
- **Historical data only**: No real-time or in-game adjustments

### Methodological Limitations
- **No causal claims**: Correlation between features and outcomes, not causation
- **Static features**: No dynamic adjustment during games
- **Binary outcomes only**: Win/loss prediction, not point spreads
- **No betting market calibration**: Model not optimized for betting ROI

### Generalizability
Results may not transfer to:
- Playoff games (different intensity and strategy)
- International basketball leagues (FIBA, EuroLeague)
- Different sports (NFL, MLB, NHL)
- Different eras (pre-2015 data may have different patterns)

## What This Project Does NOT Claim

❌ **Predicts individual player performance**  
❌ **Accounts for injuries or roster changes**  
❌ **Provides betting advice or guarantees**  
❌ **Claims perfect predictions or "sure bets"**  
❌ **Replaces human scouting or basketball expertise**  

## Key Takeaways

### Technical Learnings
1. **Time-series splits are non-negotiable**: Random splits destroy validity
2. **L10 > L5**: Longer rolling windows more stable and predictive
3. **Exponential weighting works**: Recent form matters (α=0.95 optimal)
4. **Season normalization critical**: Prevents era effects from biasing predictions
5. **Ensemble modeling pays off**: 50/50 LR/XGBoost split optimal

### Basketball Insights
1. **Net rating is king**: Single best predictor of game outcomes
2. **Home court advantage overrated**: 56% win rate, not 60%+
3. **Back-to-backs hurt performance**: ~5% win rate decrease
4. **Rest matters more for road teams**: Away B2B more predictive than home B2B
5. **Win streaks mean less than underlying metrics**: Streaks explain only 0.5% accuracy

### Project Management
1. **Clear scope prevents scope creep**: This is prediction, not betting strategy
2. **Validation strategy matters more than model choice**: XGBoost only 0.9% better than LR
3. **Limitations section is critical**: Transparency builds credibility
4. **Visualizations are essential**: Calibration curves communicate trust

## Future Extensions

### Immediate Next Steps
- **Player impact modeling**: Incorporate individual player stats and injuries
- **Travel distance features**: Add timezone changes and miles traveled
- **Playoff model**: Separate model for postseason games (different dynamics)
- **Real-time updates**: API integration for live predictions

### Advanced Extensions
- **Point spread prediction**: Predict margin of victory, not just winner
- **Expected value betting**: Optimize for ROI against Vegas lines
- **Contextual embeddings**: Use game narratives/news for "momentum" features
- **Opponent-specific modeling**: Different models for matchup styles (pace, defense)
- **Causal inference**: Estimate true effect of rest using natural experiments

### Research Directions
- Compare different feature engineering approaches (manual vs. automated)
- Test neural network architectures (LSTMs for sequential modeling)
- Validate against betting market efficiency
- Examine prediction errors for coaching strategy insights

## Academic Context

This project demonstrates techniques from:

- **Machine Learning**: Ensemble methods, gradient boosting, calibration
- **Time-Series Analysis**: Rolling windows, exponential smoothing, temporal validation
- **Sports Analytics**: Pythagorean expectation, net rating, pace adjustment
- **Statistical Modeling**: Logistic regression, probability calibration, hypothesis testing

## Contact

Questions or collaboration? Feel free to reach out!

**Email**: abdoulayeaseydi@gmail.com

---

**Note**: This project was built as a learning exercise in sports analytics and time-series machine learning. All limitations are clearly documented, and no claims are made about betting profitability or guaranteed predictions. The methodology is rigorous, the scope is intentionally narrow, and the findings are interpreted with appropriate caution.

## License

MIT License - Feel free to use this methodology for educational purposes.

---

**Built with**: Python, XGBoost, scikit-learn, Google Colab  
**Project Type**: Sports Analytics, Machine Learning, Time-Series Prediction  
**Status**: Complete ✓
