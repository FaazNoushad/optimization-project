import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import cvxpy as cp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load and preprocess data more robustly
def load_and_preprocess_data():
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv("https://raw.githubusercontent.com/FaazNoushad/optimization-project/15af3dca1589c13ac9bceafd7e08906b3cdadade/Insurance%20claims%20data%202.csv")
    
    logger.info("Handling missing values...")
    for col in ['CREDIT_SCORE', 'ANNUAL_MILEAGE']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"Filled missing values in {col} with median: {df[col].median():.2f}")
    
    logger.info("Encoding categorical variables...")
    experience_mapping = {'0-9y': 5, '10-19y': 15, '20-29y': 25, '30y+': 35}
    df['DRIVING_EXPERIENCE'] = df['DRIVING_EXPERIENCE'].map(experience_mapping)
    df['DRIVING_EXPERIENCE'] = df['DRIVING_EXPERIENCE'].fillna(df['DRIVING_EXPERIENCE'].median())
    logger.info(f"DRIVING_EXPERIENCE mapped with median value: {df['DRIVING_EXPERIENCE'].median():.2f}")
    
    if 'OUTCOME' not in df.columns:
        logger.error("OUTCOME column missing from dataset")
        raise ValueError("OUTCOME column missing from dataset")
    
    logger.info("Calculating base premiums...")
    df['BASE_PREMIUM'] = 500 + (1 - df['CREDIT_SCORE']) * 1000 + df['PAST_ACCIDENTS'] * 200
    logger.info(f"Sample premium calculation: {df['BASE_PREMIUM'].iloc[0]:.2f}")
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def optimize_insurance_policy(df):
    logger.info("Starting insurance policy optimization...")
    
    # Data preprocessing
    logger.info("Filtering data based on business constraints...")
    filtered_df = df[
        (df['CREDIT_SCORE'] >= 0.3) &
        (df['PAST_ACCIDENTS'] <= 5) &
        (df['DUIS'] <= 1) &
        (df['SPEEDING_VIOLATIONS'] <= 5) &
        (df['DRIVING_EXPERIENCE'] > 1) &
        (df['ANNUAL_MILEAGE'] < 20000)
    ].copy()
    
    if filtered_df.empty:
        logger.error("No customers match the filtering criteria")
        raise ValueError("No customers match the filtering criteria")
    
    logger.info(f"Filtered dataset shape: {filtered_df.shape}")
    
    features = ['CREDIT_SCORE', 'PAST_ACCIDENTS', 'DUIS', 
                'SPEEDING_VIOLATIONS', 'DRIVING_EXPERIENCE', 'ANNUAL_MILEAGE']
    X = filtered_df[features]
    y = filtered_df['OUTCOME']
    
    logger.info("Building logistic regression model...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, solver='lbfgs', penalty='l2', C=1.0)
    )
    model.fit(X, y)
    
    # Get coefficients
    a = model.named_steps['logisticregression'].coef_[0]
    b = model.named_steps['logisticregression'].intercept_[0]
    
    logger.info("Model coefficients:")
    for name, coef in zip(features, a):
        logger.info(f"{name}: {coef:.4f}")
    logger.info(f"Intercept: {b:.4f}")
    
    # Optimization setup
    logger.info("Setting up optimization problem...")
    x = cp.Variable(len(features))
    
    norm_factors = np.array([1.0, 5.0, 1.0, 5.0, 35.0, 20000.0])
    logger.info("Normalization factors:")
    for name, factor in zip(features, norm_factors):
        logger.info(f"{name}: {factor:.2f}")
    
    x_norm = x / norm_factors
    
    c = np.array([-300, 250, 500, 200, -15, 0.01]) / norm_factors
    d = 1000
    profit = c.T @ x + d
    risk_score = a.T @ x_norm + b
    
    logger.info("Profit coefficients (c):")
    for name, coef in zip(features, c):
        logger.info(f"{name}: {coef:.4f}")
    logger.info(f"Base profit (d): {d:.2f}")
    
    objective = cp.Maximize(profit - 1.0 * risk_score)
    constraints = [
        x[0] >= 0.3, x[1] <= 5, x[2] <= 1,
        x[3] <= 5, x[4] >= 1, x[5] <= 20000,
        x >= 0
    ]
    
    logger.info("Constraints:")
    for i, name in enumerate(features):
        if i == 0 or i == 4:
            logger.info(f"{name} ≥ {constraints[i].value}")
        else:
            logger.info(f"{name} ≤ {constraints[i].value}")
    
    # Solve the problem
    logger.info("Solving optimization problem...")
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=True, max_iters=1000, abstol=1e-6, reltol=1e-5)
        logger.info(f"Solver status: {problem.status}")
    except Exception as e:
        logger.error(f"Solver error: {str(e)}")
        return None, None
    
    if problem.status == cp.OPTIMAL:
        optimal_vars = x.value
        z = a.T @ (optimal_vars / norm_factors) + b
        claim_prob = 1/(1 + np.exp(-z))
        premium = c.T @ optimal_vars + d
        expected_profit = premium - (claim_prob * premium * 0.8)
        
        logger.info("\nOptimal Customer Profile:")
        for name, value in zip(features, optimal_vars):
            logger.info(f"{name}: {value:.2f}")
        
        logger.info("\nBusiness Results:")
        logger.info(f"Premium: ${premium:.2f}")
        logger.info(f"Claim Probability: {claim_prob:.4f}")
        logger.info(f"Expected Profit: ${expected_profit:.2f}")
        
        return optimal_vars, expected_profit
    else:
        logger.error(f"Optimization failed with status: {problem.status}")
        if problem.status == cp.INFEASIBLE:
            logger.error("Problem is infeasible - constraints may be too restrictive")
        elif problem.status == cp.UNBOUNDED:
            logger.error("Problem is unbounded - objective may be poorly formulated")
        elif problem.status == cp.INFEASIBLE_INACCURATE:
            logger.error("Problem is likely infeasible (numerical inaccuracy)")
        return None, None

# Main execution
if __name__ == "__main__":
    try:
        logger.info("Starting insurance optimization program")
        df = load_and_preprocess_data()
        optimal_profile, profit = optimize_insurance_policy(df)
        if optimal_profile is not None:
            logger.info("Optimization completed successfully")
        else:
            logger.warning("Optimization did not produce a result")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}", exc_info=True)
