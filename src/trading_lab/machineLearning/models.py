import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import itertools
import numpy as np

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params):
    """
    Docstring pour train_xgboost classifier
    
    :param X_train: Description
    :type X_train: pd.DataFrame
    :param y_train: Description
    :type y_train: pd.Series
    :param params: Description
    """
    # Map target : target Series need to be non-negative
    y_train_map = y_train.map({-1: 0, 0: 1, 1: 2})

    model = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=3,
                              random_state=42,
                              **params)
    
    model.fit(X_train, y_train_map)

    return model

def create_param_grid() -> dict:
    """
    Docstring pour create_param_grid
    
    :return: Description
    :rtype: dict
    """
    param_grid = {
    'max_depth': [2, 3, 4],              # ← Shallower trees
    'learning_rate': [0.01, 0.05],       # ← Slower learning
    'n_estimators': [50, 100, 150],      # ← Fewer trees
    'min_child_weight': [5, 10, 15],     # ← More samples per leaf
    'gamma': [0.1, 0.3, 0.5],            # ← Higher split threshold
    'reg_alpha': [1, 5, 10],             # ← Strong L1
    'reg_lambda': [5, 10, 15],           # ← Strong L2
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8]
}
    return param_grid

def predict(model, X):
    """
    Predict and map back to original labels
    
    :param model: Trained XGBoost model
    :param X: Features
    :return: Predictions as -1, 0, 1
    """
    preds_mapped = model.predict(X)
    
    # Map back: 0→-1, 1→0, 2→1
    reverse_map = {0: -1, 1: 0, 2: 1}
    preds = pd.Series([reverse_map[p] for p in preds_mapped], index=X.index)
    
    return preds


def evaluate(model, X, y, set_name='Validation'):
    """
    Evaluate model performance
    
    :param model: Trained model
    :param X: Features
    :param y: True labels
    :param set_name: Name for printing
    """
    preds = predict(model, X)
    acc = accuracy_score(y, preds)
    
    print(f"\n{'='*50}")
    print(f"{set_name} Set - Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y, preds, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))

def grid_search(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                param_grid: dict, max_combinations: int=50) -> tuple:
    """
    Docstring pour grid_search
    
    :param X_train: Description
    :type X_train: pd.DataFrame
    :param y_train: Description
    :type y_train: pd.Series
    :param X_val: Description
    :type X_val: pd.DataFrame
    :param y_val: Description
    :type y_val: pd.Series
    :param param_grid: Description
    :type param_grid: dict
    :param max_combinations: Description
    :type max_combinations: int
    :return: Description
    :rtype: tuple
    """
    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH")
    print("=" * 60)

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    all_combos = list(itertools.product(*values))

    # Limit if too many
    if len(all_combos) > max_combinations:
        print(f"Total combinations: {len(all_combos)}")
        print(f"Sampling {max_combinations} random combinations...")
        import numpy as np
        np.random.seed(42)
        indices = np.random.choice(len(all_combos), max_combinations, replace=False)
        combos = [all_combos[i] for i in indices]
    else:
        combos = all_combos
    
    print(f"Testing {len(combos)} combinations...\n")

    results = []
    best_acc = 0
    best_params = None
    best_model = None

    for idx, combo in enumerate(combos, 1):
        # Create param dict
        params = dict(zip(keys, combo))
        # Train model
        model = train_xgboost(X_train=X_train,
                              y_train=y_train,
                              params=params)
        
        # Evaluate on validation set
        preds = predict(model, X_val)
        acc = accuracy_score(y_val, preds)

        # Store results
        result = params.copy()
        result['val_accuracy'] = acc
        results.append(result)

        # Update best
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model
        
        # Progress
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(combos)} | Current: {acc:.4f} | Best: {best_acc:.4f}")
    
    # Results DataFrame
    results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)
    
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param:>20}: {value}")
    print("="*60 + "\n")
    
    return best_params, best_model, results_df
