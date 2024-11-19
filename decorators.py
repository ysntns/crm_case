import functools
import traceback
import streamlit as st
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def handle_exceptions(func: Callable) -> Callable:
    """Decorator for consistent error handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            # Log detailed error
            logger.error(f"Error in {func.__name__}: {error_msg}\n{stack_trace}")
            
            # User-friendly error message
            error_messages = {
                'ValueError': 'Geçersiz değer hatası oluştu.',
                'KeyError': 'Gerekli veri alanı bulunamadı.',
                'FileNotFoundError': 'Dosya bulunamadı.',
                'MemoryError': 'Yetersiz bellek.',
                'RuntimeError': 'İşlem sırasında hata oluştu.'
            }
            
            user_msg = error_messages.get(
                e.__class__.__name__, 
                'Beklenmeyen bir hata oluştu.'
            )
            
            st.error(f"🚨 {user_msg} Lütfen sistem yöneticisi ile iletişime geçin.")
            
            # Optional: Show technical details in expander
            with st.expander("Teknik Detaylar"):
                st.code(stack_trace)
            
            return None
    return wrapper

def cache_data(ttl_seconds: int = 3600):
    """Decorator for caching data with TTL"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Check if data exists in cache
            if cache_key in st.session_state:
                return st.session_state[cache_key]
            
            # If not in cache, compute and store
            result = func(*args, **kwargs)
            st.session_state[cache_key] = result
            return result
            
        return wrapper
    return decorator