# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import logging
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from .config import MORGAN_GENERATOR, KEAP1_MODEL_PATH, EGFR_MODEL_PATH, IKKB_MODEL_PATH
from .property_calculator import get_features

logger = logging.getLogger(__name__)

def train_and_select_best_model(df: pd.DataFrame, model_filename: str):
    """Обучает несколько моделей, выбирает лучшую и сохраняет ее."""
    if df is None or df.empty or len(df) < 50:
        logger.warning(f"Недостаточно данных для обучения модели {model_filename} ({len(df) if df is not None else 0} строк).")
        return None

    logger.info(f"--- Конкурс моделей для [{model_filename}] ---")

    # Подготовка признаков и целевой переменной
    mols = [Chem.MolFromSmiles(smi) for smi in df['canonical_smiles']]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]

    mols_valid = [mols[i] for i in valid_indices]
    df_valid = df.iloc[valid_indices]

    if len(df_valid) < 50:
        logger.warning(f"После валидации SMILES осталось мало данных ({len(df_valid)}), обучение для {model_filename} пропущено.")
        return None

    y = df_valid['pIC50'].values
    try:
        X = np.vstack([get_features(m) for m in mols_valid])
    except Exception as e:
        logger.error(f"Ошибка генерации признаков для {model_filename}: {e}")
        return None

    # Удаление строк с NaN/inf в признаках
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if len(X) < 50:
        logger.warning(f"После очистки признаков осталось мало данных ({len(X)}), обучение для {model_filename} пропущено.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Определение моделей
    estimators = [
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1))
    ]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=-1)

    models = {
        'LightGBM': estimators[2][1],
        'RandomForest': estimators[0][1],
        'XGBoost': estimators[1][1],
        'StackingEnsemble': stacking_model
    }

    best_model, best_score, best_name = None, -np.inf, ""

    for name, model in models.items():
        try:
            logger.info(f"  [*] Обучение {name}...")
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            logger.info(f"    -> R2-score на тесте: {score:.3f}")
            if score > best_score:
                best_score, best_model, best_name = score, model, name
        except Exception as e:
            logger.error(f"    -> Ошибка обучения {name}: {e}")
            continue

    if best_model:
        logger.info(f"\n  [+] Победитель для [{model_filename}]: {best_name} с R2-score {best_score:.3f}")
        joblib.dump(best_model, model_filename)
        logger.info(f"  [+] Лучшая модель сохранена в: {model_filename}")
        return best_model
    else:
        logger.error(f"[!] Ни одна модель не была успешно обучена для {model_filename}.")
        return None

def prepare_models(keap1_clean, egfr_clean, ikkb_clean):
    """Главная функция для подготовки всех моделей-предикторов."""
    logger.info("\n[*] Этап 3: Конкурс моделей-предикторов...")
    
    keap1_model = train_and_select_best_model(keap1_clean, KEAP1_MODEL_PATH)
    egfr_model = train_and_select_best_model(egfr_clean, EGFR_MODEL_PATH)
    ikkb_model = train_and_select_best_model(ikkb_clean, IKKB_MODEL_PATH)
    
    logger.info("\n✅ Все лучшие модели-предикторы готовы.")
    return keap1_model, egfr_model, ikkb_model
