from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.feature_names]
        except KeyError as e:
            logger.error(f"Feature selection error: {str(e)}")
            raise

class MLUtils:
    @staticmethod
    def create_pipeline(model, features: List[str]) -> Pipeline:
        """Create an optimized ML pipeline"""
        try:
            return Pipeline([
                ('selector', FeatureSelector(features)),
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        except Exception as e:
            logger.error(f"Pipeline creation error: {str(e)}")
            raise

    @staticmethod
    def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """Model evaluation with cross-validation"""
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv)
            return {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
        except Exception as e:
            logger.error(f"Model evaluation error: {str(e)}")
            raise

    @staticmethod
    def get_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                return dict(zip(feature_names, 
                              pipeline.named_steps['model'].feature_importances_))
            return {}
        except Exception as e:
            logger.error(f"Feature importance error: {str(e)}")
            return {}