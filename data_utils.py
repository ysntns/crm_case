import logging
from functools import lru_cache
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import streamlit as st

logger = logging.getLogger(__name__)

class DataUtils:
    def __init__(self):
        self._key = Fernet.generate_key()
        self._cipher = Fernet(self._key)
    
    @lru_cache(maxsize=128)
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Cached data loading with error handling"""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            st.error("Veri yükleme hatası oluştu!")
            return pd.DataFrame()

    def encrypt_sensitive_data(self, data: pd.DataFrame, sensitive_cols: list) -> pd.DataFrame:
        """Sensitive data encryption"""
        try:
            df = data.copy()
            for col in sensitive_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(
                        lambda x: self._cipher.encrypt(x.encode()).decode()
                    )
            return df
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            return data

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning with advanced error handling"""
        try:
            df = data.copy()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            
            # Remove outliers using IQR
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            
            return df
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return data