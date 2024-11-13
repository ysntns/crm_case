# Streamlit ve veri gösterimi için gereken kütüphaneler
from imblearn.over_sampling import SMOTE
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Veri işleme ve analizi için gereken kütüphaneler
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from mlxtend.evaluate import accuracy_score
from pandas import ExcelWriter
from io import BytesIO

# Makine öğrenmesi ve istatistiksel analiz için gereken kütüphaneler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Diğer yardımcı kütüphaneler
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import logging
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="iGaming CRM Analytics",
    page_icon="🎲",
    layout="wide"
)

# Loading bildirimi
with st.spinner("Dashboard yükleniyor... "):
    time.sleep(1)
st.success("Dashboard hazır!")

# Ana stil ve üst menü
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .top-menu {
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .menu-item {
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        background: rgba(255,255,255,0.1);
        margin: 5px;
        text-align: center;
        min-width: 200px;
        transition: transform 0.3s ease;
    }
    .menu-item:hover {
        transform: translateY(-3px);
        background: rgba(255,255,255,0.2);
    }
    .menu-icon {
        font-size: 1.2em;
        margin-right: 8px;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        width: 100%;
    }
    </style>

    <h1 class='main-header'>🎲 iGaming için CRM Analizi ve Raporlama Panosu🎲</h1>

    <div class="top-menu">
        <div class="menu-item">
            <span class="menu-icon">🎯</span>
            Gerçek Zamanlı Analiz
        </div>
        <div class="menu-item">
            <span class="menu-icon">🤖</span>
            AI Tahminleme
        </div>
        <div class="menu-item">
            <span class="menu-icon">📊</span>
            Detaylı Raporlama
        </div>
        <div class="menu-item">
            <span class="menu-icon">⚡</span>
            Performans İzleme
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        color: white;
        padding: 20px 35px;
        text-align: center;
        border-top: 1px solid #4CAF50;
        z-index: 999;
        font-size: 1em;
    }
    .footer-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
        justify-content: center;
    }
    .footer-title {
        color: #4CAF50;
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .footer-subtitle {
        color: #9e9e9e;
        font-size: 1em;
        margin-bottom: 10px;
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .social-link {
        color: white;
        text-decoration: none;
        font-size: 1.1em;
        transition: color 0.3s ease;
    }
    .social-link:hover {
        color: #4CAF50;
    }
    .developer-info {
        color: #4CAF50;
        font-size: 0.9em;
    }
    .disclaimer {
        font-size: 0.75em;
        color: #757575;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .footer {
            padding: 15px 20px;
            font-size: 0.9em;  /* Optimize font size */
        }
        .footer-title {
            font-size: 1em;
            margin-bottom: 5px;
        }
        .footer-subtitle {
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        .social-links {
            gap: 10px;
            justify-content: center;
        }
        .social-link {
            font-size: 1em;
        }
        .developer-info {
            font-size: 0.85em;
        }
        .disclaimer {
            font-size: 0.7em;
        }
    }
    </style>

    <div class="footer">
        <div class="footer-content">
            <div class="footer-title">🎮 iGaming Analytics Suite</div>
            <div class="footer-subtitle">Gelişmiş CRM Analiz ve Tahminleme Sistemi</div>
            <div class="social-links">
                <a href="https://github.com/ysntns" target="_blank" class="social-link">
                    <i class="fab fa-github"></i> Github
                </a>
                <a href="https://www.linkedin.com/in/ysntns" target="_blank" class="social-link">
                    <i class="fab fa-linkedin"></i> Linkedin
                </a>
                <a href="https://twitter.com/ysntnss" target="_blank" class="social-link">
                    <i class="fab fa-twitter"></i> Twitter
                </a>
                <a href="mailto:ysn.tnss@gmail.com" class="social-link">
                    <i class="fas fa-envelope"></i> Mail
                </a>
            </div>
            <div class="developer-info">Geliştirici: Yasin Tanış | v2.0</div>
            <div class="disclaimer">Bu sistem demo amaçlı geliştirilmiş olup, gerçek verileri simüle etmektedir.</div>
        </div>
    </div>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)


# Cache süresi
CACHE_TTL = 3600  # 1 saat


@st.cache_data(ttl=CACHE_TTL)
def generate_betting_data(n_players: int = 1000) -> pd.DataFrame:
    """Örnek bahis verisi oluşturur."""
    try:
        np.random.seed(42)
        # Temel veriler
        player_ids = range(1, n_players + 1)
        registration_dates = [datetime.now() - timedelta(days=np.random.randint(1, 365))
                              for _ in range(n_players)]

        # Finansal veriler
        deposits = np.random.normal(1000, 300, n_players)
        withdrawals = deposits * np.random.uniform(0.5, 0.9, n_players)
        bets_placed = np.random.poisson(50, n_players)
        ggr = deposits - withdrawals
        bonus_usage = np.random.uniform(0, 1, n_players) * 200

        # Kategorik veriler
        bet_types = np.random.choice(['Sport', 'Casino', 'Poker', 'Virtual'], n_players,
                                     p=[0.4, 0.3, 0.2, 0.1])
        locations = np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Bursa'],
                                     n_players)

        # Risk ve aktivite verileri
        risk_scores = np.random.beta(2, 5, n_players) * 100
        login_frequency = np.random.poisson(8, n_players)
        average_stake = np.random.normal(100, 30, n_players)

        return pd.DataFrame({
            'OyuncuID': player_ids,
            'KayitTarihi': registration_dates,
            'ToplamDepozit': deposits,
            'ToplamCekim': withdrawals,
            'BahisSayisi': bets_placed,
            'GGR': ggr,
            'BonusKullanimi': bonus_usage,
            'OyunTipi': bet_types,
            'Sehir': locations,
            'RiskSkoru': risk_scores,
            'GirisSikligi': login_frequency,
            'OrtBahis': average_stake,
            'SonAktivite': np.random.randint(1, 30, n_players)
        })
    except Exception as e:
        st.error(f"Veri oluşturma hatası: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def prepare_filtered_data(data: pd.DataFrame,
                          selected_games: List[str],
                          selected_cities: List[str],
                          min_ggr: float) -> pd.DataFrame:
    try:
        return data[
            (data['OyunTipi'].isin(selected_games)) &
            (data['Sehir'].isin(selected_cities)) &
            (data['GGR'] >= min_ggr)
            ]
    except Exception as e:
        st.error(f"Veri filtreleme hatası: {str(e)}")
        return data



# Ana veri yükleme
with st.spinner('Veriler hazırlanıyor...'):
    try:
        data = generate_betting_data()
        if data.empty:
            st.error("Veri yüklenemedi!")
            st.stop()
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        st.stop()

def configure_sidebar():
    with st.sidebar:
        st.markdown("### Ana Menü")

        analysis_type = st.selectbox(
            "Analiz Türü Seçin",
            options=[
                "Genel Bakış",
                "Oyuncu Segmentasyonu",
                "GGR Analizi",
                "Risk Analizi",
                "Bonus Performansı",
                "Oyuncu Davranışı",
                "Model Bazlı Tahminler",
                "Cohort Analizi",
                "A/B Test Analizi",
                "ANOVA Analizi",
                "ROI Analizi",
                "Trend Analizi"
            ]
        )

        # Filtreleme seçenekleri
        st.markdown("### Filtreler")
        with st.expander("Filtreleme Seçenekleri", expanded=True):
            selected_games = st.multiselect(
                "Oyun Tipi",
                options=data['OyunTipi'].unique(),
                default=data['OyunTipi'].unique()
            )

            selected_cities = st.multiselect(
                "Şehir",
                options=data['Sehir'].unique(),
                default=data['Sehir'].unique()
            )

            min_ggr = st.number_input(
                "Minimum GGR",
                min_value=float(data['GGR'].min()),
                max_value=float(data['GGR'].max()),
                value=float(data['GGR'].min())
            )

        return analysis_type, selected_games, selected_cities, min_ggr


def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Temel metrikleri hesaplar."""
    try:
        return {
            'total_ggr': data['GGR'].sum(),
            'avg_deposit': data['ToplamDepozit'].mean(),
            'bonus_ratio': (data['BonusKullanimi'].sum() / data['ToplamDepozit'].sum() * 100),
            'avg_risk': data['RiskSkoru'].mean(),
            'active_players': (data['SonAktivite'] < 7).sum(),
            'conversion_rate': (data['GGR'] > 0).mean() * 100
        }
    except Exception as e:
        st.error(f"Metrik hesaplama hatası: {str(e)}")
        return {}


def show_overview(data: pd.DataFrame):
    """Genel bakış dashboardunu gösterir."""
    try:
        metrics = calculate_metrics(data)

        # KPI'lar
        cols = st.columns(4)
        with cols[0]:
            st.metric("Toplam GGR", f"₺{metrics['total_ggr']:,.2f}", "↑ 15%")
        with cols[1]:
            st.metric("Ortalama Depozit", f"₺{metrics['avg_deposit']:,.2f}", "↑ 8%")
        with cols[2]:
            st.metric("Bonus/Depozit", f"{metrics['bonus_ratio']:.1f}%", "↓ 2%")
        with cols[3]:
            st.metric("Risk Skoru", f"{metrics['avg_risk']:.1f}", "↓ 0.5")

        # Grafikler
        col1, col2 = st.columns(2)

        with col1:
            # Oyun tipi dağılımı
            fig = px.pie(data, names='OyunTipi', values='GGR', title='GGR Dağılımı')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Günlük trend
            daily_data = data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date)['GGR'].sum().reset_index()
            fig = px.line(daily_data, x='KayitTarihi', y='GGR', title='GGR Trendi')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Genel bakış hatası: {str(e)}")


def show_player_segmentation(data: pd.DataFrame):
    st.subheader("Oyuncu Segmentasyonu Analizi 👥")

    with st.expander("ℹ️ Segmentasyon Hakkında", expanded=False):
        st.info("""
        ### Segmentasyon Kriterleri
        - 💎 **VIP**: Yüksek GGR ve sık aktivite
        - 🌟 **Aktif**: Düzenli aktivite gösteren
        - 🌱 **Yeni**: Son 30 gün içinde kayıt olan
        - ⚠️ **Risk**: Yüksek kayıp veya riskli davranış
        - 💤 **Uyuyan**: 30+ gün aktivite göstermeyen
        """)

    try:
        # RFM Analizi için metrikler
        data['Recency'] = (datetime.now() - pd.to_datetime(data['KayitTarihi'])).dt.days
        data['Frequency'] = data['BahisSayisi']
        data['Monetary'] = data['GGR']

        # Veri hazırlama
        features = ['Recency', 'Frequency', 'Monetary']
        X = data[features].copy()

        # Eksik değerleri doldur
        X = X.fillna(X.mean())

        # Standardizasyon
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Örnek sayısı kontrolü ve kümeleme
        n_samples = len(X_scaled)
        n_clusters = min(n_samples, 5)  # Maximum 5 küme

        if n_samples >= 2:  # En az 2 örnek olmalı
            # K-means kümeleme
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data['Segment'] = kmeans.fit_predict(X_scaled)

            # Segment isimlendirme
            # Ortalama GGR'a göre segmentleri sırala ve isimlendir
            segment_means = data.groupby('Segment')['GGR'].mean().sort_values(ascending=False)
            segment_mapping = {}
            segment_names = ['VIP 💎', 'Aktif 🌟', 'Yeni 🌱', 'Riskli ⚠️', 'Uyuyan 💤']

            for i, (segment, _) in enumerate(segment_means.items()):
                if i < len(segment_names):
                    segment_mapping[segment] = segment_names[i]

            data['SegmentName'] = data['Segment'].map(segment_mapping)

            # Segment Metrikleri
            segments_counts = data['SegmentName'].value_counts()

            cols = st.columns(len(segments_counts))

            for i, (segment, count) in enumerate(segments_counts.items()):
                with cols[i]:
                    st.metric(
                        segment,
                        f"{count:,}",
                        f"{count / len(data) * 100:.1f}%"
                    )

            # Segment Dağılımı
            fig_distribution = px.pie(
                data,
                names='SegmentName',
                values='GGR',
                title='Segment Bazlı GGR Dağılımı',
                template='plotly_dark'
            )
            st.plotly_chart(fig_distribution, use_container_width=True)

            # Segment Karakteristikleri
            st.subheader("Segment Karakteristikleri")

            segment_stats = data.groupby('SegmentName').agg({
                'GGR': ['mean', 'sum'],
                'BahisSayisi': 'mean',
                'BonusKullanimi': 'mean',
                'OyuncuID': 'count'
            }).round(2)

            st.dataframe(
                segment_stats.style.background_gradient(cmap='YlOrRd'),
                use_container_width=True
            )

            # Segment Karşılaştırma
            st.subheader("Segment Karşılaştırma")

            comparison_metric = st.selectbox(
                "Karşılaştırma Metriği",
                ['GGR', 'BahisSayisi', 'BonusKullanimi', 'OrtBahis']
            )

            fig_comparison = go.Figure()

            for segment in data['SegmentName'].unique():
                segment_data = data[data['SegmentName'] == segment][comparison_metric]

                fig_comparison.add_trace(go.Box(
                    y=segment_data,
                    name=segment,
                    boxpoints='outliers'
                ))

            fig_comparison.update_layout(
                title=f'Segment Bazlı {comparison_metric} Dağılımı',
                template='plotly_dark'
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

            # Segment Önerileri
            st.subheader("Segment Bazlı Öneriler")

            for segment in segment_mapping.values():
                segment_data = data[data['SegmentName'] == segment]

                with st.expander(f"{segment} Segment Önerileri"):
                    if 'VIP' in segment:
                        st.success(f"""
                        - 🎁 Özel VIP bonusları ve promosyonlar
                        - 👤 Kişisel hesap yöneticisi atama
                        - 🎯 Özelleştirilmiş kampanyalar
                        - ⚡ Yüksek bahis limitleri
                        Oyuncu Sayısı: {len(segment_data):,}
                        Ortalama GGR: ₺{segment_data['GGR'].mean():,.2f}
                        """)
                    elif 'Aktif' in segment:
                        st.info(f"""
                        - 🎮 Oyun çeşitliliği sunma
                        - 🎁 Düzenli bonus teklifleri
                        - 📊 Aktivite bazlı ödüller
                        - 🎯 Cross-selling fırsatları
                        Oyuncu Sayısı: {len(segment_data):,}
                        Ortalama GGR: ₺{segment_data['GGR'].mean():,.2f}
                        """)
                    elif 'Yeni' in segment:
                        st.info(f"""
                        - 🎁 Hoşgeldin bonusları
                        - 📚 Platform kullanım rehberi
                        - 🎮 Düşük riskli oyunlar önerme
                        - 📞 Destek hattı önceliklendirme
                        Oyuncu Sayısı: {len(segment_data):,}
                        Ortalama GGR: ₺{segment_data['GGR'].mean():,.2f}
                        """)
                    elif 'Riskli' in segment:
                        st.warning(f"""
                        - ⚠️ Risk limitlerini düzenleme
                        - 📞 Proaktif iletişim
                        - 🛡️ Sorumlu oyun araçları
                        - 📊 Aktivite monitöring
                        Oyuncu Sayısı: {len(segment_data):,}
                        Ortalama GGR: ₺{segment_data['GGR'].mean():,.2f}
                        """)
                    else:  # Uyuyan
                        st.error(f"""
                        - 🎁 Geri dönüş kampanyaları
                        - 💌 Re-aktivasyon e-postaları
                        - 🎯 Kişiselleştirilmiş teklifler
                        - 📞 Win-back aramaları
                        Oyuncu Sayısı: {len(segment_data):,}
                        Ortalama GGR: ₺{segment_data['GGR'].mean():,.2f}
                        """)
        else:
            st.warning("Segmentasyon için yeterli veri bulunmuyor. En az 2 oyuncu verisi gerekli.")

    except Exception as e:
        st.error(f"Segmentasyon analizi sırasında bir hata oluştu: {str(e)}")
        logger.error(f"Segmentation error: {str(e)}")


def show_risk_analysis(data: pd.DataFrame):
    """Risk analizi bölümünü gösterir."""
    st.subheader("Risk Analizi ve İzleme 🎯")

    with st.expander("ℹ️ Risk Analizi Hakkında", expanded=False):
        st.info("""
        ### Risk Değerlendirme Kriterleri
        - 🔴 **Yüksek Risk (70-100)**: Acil aksiyon gerektirir
        - 🟡 **Orta Risk (30-70)**: Yakın takip gerektirir
        - 🟢 **Düşük Risk (0-30)**: Normal takip

        Risk skoru şu faktörlere göre hesaplanır:
        - Oyun davranışı
        - Yatırım/Çekim oranı
        - Aktivite sıklığı
        - Bonus kullanımı
        """)

    # Risk Metrikleri
    col1, col2, col3, col4 = st.columns(4)

    high_risk = len(data[data['RiskSkoru'] > 70])
    avg_risk = data['RiskSkoru'].mean()
    risk_change = data['RiskSkoru'].mean() - 50  # Baseline'dan fark

    with col1:
        st.metric(
            "Yüksek Riskli Oyuncular 🚨",
            f"{high_risk:,}",
            f"{high_risk / len(data) * 100:.1f}% Toplam"
        )

    with col2:
        st.metric(
            "Ortalama Risk Skoru 📊",
            f"{avg_risk:.1f}",
            f"{risk_change:+.1f} Değişim"
        )

    with col3:
        st.metric(
            "Risk Trendi 📈",
            "Yükseliş" if risk_change > 0 else "Düşüş",
            f"{abs(risk_change):.1f} puan"
        )

    with col4:
        st.metric(
            "Risk Altındaki GGR 💰",
            f"₺{data[data['RiskSkoru'] > 70]['GGR'].sum():,.2f}",
            f"{(data[data['RiskSkoru'] > 70]['GGR'].sum() / data['GGR'].sum() * 100):.1f}%"
        )

    # Risk Analizi Seçenekleri
    analysis_tab = st.radio(
        "📊 Risk Analiz Türü",
        ["Dağılım Analizi", "Segment Bazlı", "Trend Analizi", "Aksiyon Önerileri"],
        horizontal=True
    )

    if analysis_tab == "Dağılım Analizi":
        # Risk Dağılımı
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=data['RiskSkoru'],
            name='Risk Dağılımı',
            nbinsx=30,
            marker_color='rgba(255, 99, 71, 0.7)'
        ))

        # Risk eşikleri
        fig.add_vline(x=30, line_dash="dash", line_color="green",
                     annotation_text="Düşük Risk Eşiği")
        fig.add_vline(x=70, line_dash="dash", line_color="red",
                     annotation_text="Yüksek Risk Eşiği")

        fig.update_layout(
            title='Risk Skoru Dağılımı',
            xaxis_title='Risk Skoru',
            yaxis_title='Oyuncu Sayısı',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk-GGR İlişkisi
        fig_scatter = px.scatter(
            data,
            x='RiskSkoru',
            y='GGR',
            color='OyunTipi',
            size='BahisSayisi',
            title='Risk-GGR İlişkisi',
            template='plotly_dark'
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    elif analysis_tab == "Segment Bazlı":
        # Risk segmentleri
        data['RiskSegment'] = pd.qcut(
            data['RiskSkoru'],
            q=4,
            labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek']
        )

        # Segment analizi
        segment_metrics = data.groupby('RiskSegment').agg({
            'OyuncuID': 'count',
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': 'mean',
            'BahisSayisi': 'mean'
        }).round(2)

        # Segment görselleştirmesi
        fig = px.sunburst(
            data,
            path=['RiskSegment', 'OyunTipi'],
            values='GGR',
            title='Risk Segment Dağılımı'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Segment detayları
        st.dataframe(
            segment_metrics.style.background_gradient(cmap='RdYlGn_r'),
            use_container_width=True
        )

    elif analysis_tab == "Trend Analizi":
        # Günlük risk trendi
        daily_risk = data.groupby(
            pd.to_datetime(data['KayitTarihi']).dt.date
        ).agg({
            'RiskSkoru': ['mean', 'max', 'count'],
            'GGR': 'sum'
        }).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_risk['KayitTarihi'],
            y=daily_risk['RiskSkoru']['mean'],
            name='Ortalama Risk',
            line=dict(color='orange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=daily_risk['KayitTarihi'],
            y=daily_risk['RiskSkoru']['max'],
            name='Maksimum Risk',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Risk Skoru Trend Analizi',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    else:  # Aksiyon Önerileri
        st.subheader("Risk Azaltma Önerileri")

        # Yüksek riskli oyuncular
        high_risk_players = data[data['RiskSkoru'] > 70].sort_values(
            'RiskSkoru', ascending=False
        )

        for _, player in high_risk_players.head().iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(
                        "Risk Skoru",
                        f"{player['RiskSkoru']:.1f}",
                        "Yüksek Risk"
                    )

                with col2:
                    st.write("**Önerilen Aksiyonlar:**")
                    if player['RiskSkoru'] > 90:
                        st.error("""
                        1. 🚫 Hesap limitleri aktif edilmeli
                        2. 📞 Acil müşteri temsilcisi araması
                        3. 📊 Günlük aktivite raporu
                        """)
                    elif player['RiskSkoru'] > 80:
                        st.warning("""
                        1. ⚠️ Uyarı mesajları gönderilmeli
                        2. 📋 Haftalık aktivite özeti
                        3. ⏰ Mola hatırlatıcıları
                        """)
                    else:
                        st.info("""
                        1. 📝 Düzenli takip
                        2. 📈 Aktivite monitöring
                        3. ℹ️ Bilgilendirme mesajları
                        """)

    # Risk Raporu İndirme
    with st.sidebar:
        st.markdown("### 📥 Risk Raporu")
        report_format = st.selectbox(
            "Format Seçin",
            ["Excel", "PDF", "CSV"]
        )

        if st.button("Risk Raporu İndir"):
            with st.spinner('Rapor hazırlanıyor...'):
                # Excel raporu
                if report_format == "Excel":
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer) as writer:
                        data.to_excel(writer, sheet_name='Risk Data')
                        daily_risk.to_excel(writer, sheet_name='Risk Trends')
                        segment_metrics.to_excel(writer, sheet_name='Risk Segments')
                        high_risk_players.to_excel(writer, sheet_name='High Risk Players')

                    st.download_button(
                        label="📥 Excel İndir",
                        data=buffer,
                        file_name="risk_analysis.xlsx",
                        mime="application/vnd.ms-excel"
                    )


def show_bonus_performance(data: pd.DataFrame):
    st.subheader("Bonus Performans Analizi 🎁")

    with st.expander("ℹ️ Bonus Performansı Hakkında", expanded=False):
        st.info("""
        ### Bonus Performans Metrikleri
        - 🎯 **Dönüşüm Oranı**: Bonusların GGR'a dönüşme oranı
        - 💰 **ROI**: Bonus yatırımının geri dönüş oranı
        - 📊 **Etkinlik**: Bonus kullanım etkinliği
        - 🎮 **Oyun Bazlı**: Oyun tipine göre bonus performansı
        """)

    # Bonus Metrikleri
    total_bonus = data['BonusKullanimi'].sum()
    total_ggr = data['GGR'].sum()
    bonus_users = len(data[data['BonusKullanimi'] > 0])
    conversion_rate = (data[data['BonusKullanimi'] > 0]['GGR'] > 0).mean() * 100

    # KPI'lar
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Toplam Bonus 🎁",
            f"₺{total_bonus:,.2f}",
            f"{(total_bonus / data['ToplamDepozit'].sum() * 100):.1f}% Oran"
        )

    with col2:
        st.metric(
            "Bonus ROI 📈",
            f"{((total_ggr - total_bonus) / total_bonus * 100):.1f}%",
            "Yatırım Getirisi"
        )

    with col3:
        st.metric(
            "Bonus Kullanıcıları 👥",
            f"{bonus_users:,}",
            f"{(bonus_users / len(data) * 100):.1f}% Penetrasyon"
        )

    with col4:
        st.metric(
            "Dönüşüm Oranı 🎯",
            f"{conversion_rate:.1f}%",
            "GGR Pozitif Oran"
        )

    # Bonus Analiz Türü
    analysis_type = st.radio(
        "📊 Analiz Türü",
        ["Zaman Bazlı", "Oyun Tipi Bazlı", "Segment Bazlı", "Etki Analizi"],
        horizontal=True
    )

    if analysis_type == "Zaman Bazlı":
        # Günlük bonus ve GGR trendi
        daily_data = data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date).agg({
            'BonusKullanimi': ['sum', 'count'],
            'GGR': 'sum'
        }).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_data['KayitTarihi'],
            y=daily_data['BonusKullanimi']['sum'],
            name='Toplam Bonus',
            line=dict(color='#2ecc71')
        ))

        fig.add_trace(go.Scatter(
            x=daily_data['KayitTarihi'],
            y=daily_data['GGR']['sum'],
            name='GGR',
            line=dict(color='#e74c3c')
        ))

        fig.update_layout(
            title='Bonus ve GGR Trendi',
            xaxis_title='Tarih',
            yaxis_title='Tutar (₺)',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Oyun Tipi Bazlı":
        game_bonus = data.groupby('OyunTipi').agg({
            'BonusKullanimi': ['sum', 'mean', 'count'],
            'GGR': 'sum',
            'OyuncuID': 'nunique'
        }).round(2)

        # Bonus dağılımı
        fig_dist = px.pie(
            data,
            values='BonusKullanimi',
            names='OyunTipi',
            title='Oyun Tipi Bazlı Bonus Dağılımı'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Oyun tipi detayları
        st.dataframe(
            game_bonus.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )

    elif analysis_type == "Segment Bazlı":
        # RFM bazlı segmentasyon
        data['BonusSegment'] = pd.qcut(
            data['BonusKullanimi'].clip(lower=0),
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )

        segment_analysis = data.groupby('BonusSegment').agg({
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': ['sum', 'mean'],
            'OyuncuID': 'count'
        }).round(2)

        # Segment gösterimi
        fig_segment = px.sunburst(
            data,
            path=['BonusSegment', 'OyunTipi'],
            values='BonusKullanimi',
            title='Bonus Segment Dağılımı'
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        # Segment detayları
        st.dataframe(
            segment_analysis.style.background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )

    else:  # Etki Analizi
        st.subheader("Bonus Etki Analizi")

        # Bonus-GGR ilişkisi
        fig_scatter = px.scatter(
            data[data['BonusKullanimi'] > 0],
            x='BonusKullanimi',
            y='GGR',
            color='OyunTipi',
            trendline="ols",
            title='Bonus-GGR İlişkisi'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Bonus etki metrikleri
        bonus_impact = {
            'with_bonus': data[data['BonusKullanimi'] > 0]['GGR'].mean(),
            'without_bonus': data[data['BonusKullanimi'] == 0]['GGR'].mean()
        }

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Bonuslu Oyuncular Ort. GGR",
                f"₺{bonus_impact['with_bonus']:.2f}",
                f"{((bonus_impact['with_bonus'] / bonus_impact['without_bonus'] - 1) * 100):.1f}% Fark"
            )

        with col2:
            st.metric(
                "Bonussuz Oyuncular Ort. GGR",
                f"₺{bonus_impact['without_bonus']:.2f}",
                "Baz Değer"
            )

    # Bonus Önerileri
    st.subheader("Bonus Optimizasyon Önerileri")

    if (total_ggr - total_bonus) / total_bonus > 0.5:
        st.success("""
        ✅ Bonus stratejisi etkili:
        - 🎯 Mevcut bonus yapısını koruyun
        - 📈 Yüksek performanslı segmentlere odaklanın
        - 🎮 Oyun bazlı bonus çeşitlendirmesi yapın
        """)
    else:
        st.warning("""
        ⚠️ Bonus optimizasyonu gerekli:
        - 📊 Bonus/GGR oranlarını gözden geçirin
        - 🎯 Hedef kitleyi daraltın
        - ⚡ Bonus koşullarını optimize edin
        """)


def show_ggr_analysis(data: pd.DataFrame):
    """GGR analizi bölümünü gösterir."""
    st.subheader("GGR Performans Analizi 📈")

    with st.expander("ℹ️ GGR Analizi Hakkında", expanded=False):
        st.info("""
        - **GGR (Gross Gaming Revenue)**: Toplam oyuncu kaybı
        - **Net GGR**: Bonuslar çıkarıldıktan sonraki net kazanç
        - **GGR/Depozit Oranı**: Yatırımların ne kadar efektif kullanıldığı
        """)

    # KPI'lar - Animasyonlu metrikler
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        total_ggr = data['GGR'].sum()
        avg_ggr = data['GGR'].mean()
        max_ggr = data['GGR'].max()
        profit_rate = (data['GGR'].sum() / data['ToplamDepozit'].sum() * 100)

        with col1:
            st.metric(
                label="Toplam GGR 💰",
                value=f"₺{total_ggr:,.2f}",
                delta=f"{((total_ggr / data['ToplamDepozit'].sum()) * 100):.1f}% Oran"
            )

        with col2:
            st.metric(
                label="Ortalama GGR/Oyuncu 👤",
                value=f"₺{avg_ggr:,.2f}",
                delta=f"{((avg_ggr / data['GGR'].mean().mean()) - 1) * 100:.1f}%"
            )

        with col3:
            st.metric(
                label="En Yüksek GGR 🏆",
                value=f"₺{max_ggr:,.2f}",
                delta="Top %1"
            )

        with col4:
            st.metric(
                label="Karlılık Oranı 📊",
                value=f"{profit_rate:.1f}%",
                delta="Hedefin üzerinde" if profit_rate > 20 else "Hedefin altında"
            )

    # Detaylı Analiz Seçenekleri
    analysis_type = st.radio(
        "📊 Analiz Türü Seçin",
        ["Zaman Serisi", "Segment Analizi", "Oyun Tipi Analizi", "Korelasyon Analizi"],
        horizontal=True
    )

    if analysis_type == "Zaman Serisi":
        st.subheader("GGR Zaman Serisi Analizi 📅")

        # Tarih aralığı seçimi
        date_range = st.date_input(
            "Tarih Aralığı Seçin",
            [data['KayitTarihi'].min(), data['KayitTarihi'].max()]
        )

        # Trend analizi
        fig = go.Figure()

        # Günlük GGR
        daily_ggr = data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date)['GGR'].sum()

        fig.add_trace(go.Scatter(
            x=daily_ggr.index,
            y=daily_ggr.values,
            name='Günlük GGR',
            line=dict(color='#2ecc71', width=2)
        ))

        # 7 günlük hareketli ortalama
        fig.add_trace(go.Scatter(
            x=daily_ggr.index,
            y=daily_ggr.rolling(7).mean(),
            name='7 Günlük Ortalama',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))

        fig.update_layout(
            title='GGR Trend Analizi',
            xaxis_title='Tarih',
            yaxis_title='GGR (₺)',
            hovermode='x unified',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Segment Analizi":
        st.subheader("GGR Segment Analizi 👥")

        # GGR segmentleri
        ggr_segments = pd.qcut(data['GGR'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        segment_analysis = data.groupby(ggr_segments).agg({
            'OyuncuID': 'count',
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': 'mean',
            'RiskSkoru': 'mean'
        }).round(2)

        # Segment gösterimi
        fig = px.treemap(
            data_frame=pd.DataFrame({
                'Segment': ggr_segments,
                'GGR': data['GGR']
            }),
            path=[px.Constant("GGR Segmentleri"), 'Segment'],
            values='GGR',
            title='GGR Segment Dağılımı'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Segment detayları
        st.dataframe(
            segment_analysis.style.background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )

    elif analysis_type == "Oyun Tipi Analizi":
        st.subheader("Oyun Tipi Bazlı GGR Analizi 🎮")

        # Oyun tipi analizi
        game_analysis = data.groupby('OyunTipi').agg({
            'GGR': ['sum', 'mean', 'count'],
            'BonusKullanimi': 'sum',
            'RiskSkoru': 'mean'
        }).round(2)

        # Sunburst chart
        fig = px.sunburst(
            data,
            path=['OyunTipi'],
            values='GGR',
            title='Oyun Tipi GGR Dağılımı'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Oyun tipi detayları
        st.dataframe(
            game_analysis.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

    else:  # Korelasyon Analizi
        st.subheader("GGR Korelasyon Analizi 🔍")

        # Korelasyon matrisi
        corr_cols = ['GGR', 'BonusKullanimi', 'ToplamDepozit', 'BahisSayisi', 'RiskSkoru']
        corr_matrix = data[corr_cols].corr()

        fig = px.imshow(
            corr_matrix,
            title='GGR Korelasyon Matrisi',
            color_continuous_scale='RdBu'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detaylı ilişki analizi
        st.subheader("Metrikler Arası İlişki Detayı")
        selected_metric = st.selectbox(
            "İlişki analizi için metrik seçin",
            corr_cols
        )

        # Scatter plot
        fig_scatter = px.scatter(
            data,
            x=selected_metric,
            y='GGR',
            color='OyunTipi',
            trendline="ols",
            title=f'GGR vs {selected_metric}'
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    # İndirilebilir Rapor
    st.sidebar.markdown("### 📥 Rapor İndir")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    data.to_excel(writer, sheet_name='Raw Data')
                    if 'daily_ggr' in locals():  # Kontrol ekleyelim
                        daily_ggr.to_frame().to_excel(writer, sheet_name='Daily GGR')
                    if 'segment_analysis' in locals():  # Kontrol ekleyelim
                        segment_analysis.to_excel(writer, sheet_name='Segment Analysis')
                    if 'game_analysis' in locals():  # Kontrol ekleyelim
                        game_analysis.to_excel(writer, sheet_name='Game Analysis')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="ggr_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_bonus_analysis(data: pd.DataFrame):
    """Bonus analizi bölümünü gösterir."""
    st.subheader("Bonus Performans Analizi 🎁")

    with st.expander("ℹ️ Bonus Analizi Hakkında", expanded=False):
        st.info("""
        ### Bonus Performans Metrikleri
        - 💰 **Bonus/Depozit Oranı**: Verilen bonusların depozitlere oranı
        - 🎯 **Bonus Dönüşüm Oranı**: Bonusların GGR'a dönüşüm oranı
        - 📊 **Bonus ROI**: Bonus yatırımının geri dönüş oranı
        - 🔄 **Bonus Kullanım Oranı**: Verilen bonusların kullanılma oranı
        """)

    # Temel Bonus Metrikleri
    total_bonus = data['BonusKullanimi'].sum()
    total_deposit = data['ToplamDepozit'].sum()
    bonus_users = len(data[data['BonusKullanimi'] > 0])
    bonus_conversion = (data[data['BonusKullanimi'] > 0]['GGR'] > 0).mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Toplam Bonus 🎁",
            f"₺{total_bonus:,.2f}",
            f"{(total_bonus / total_deposit * 100):.1f}% Depozit Oranı"
        )

    with col2:
        st.metric(
            "Bonus Kullanıcıları 👥",
            f"{bonus_users:,}",
            f"{(bonus_users / len(data) * 100):.1f}% Penetrasyon"
        )

    with col3:
        st.metric(
            "Bonus ROI 📈",
            f"{((data['GGR'].sum() - total_bonus) / total_bonus * 100):.1f}%",
            "Yatırım Getirisi"
        )

    with col4:
        st.metric(
            "Dönüşüm Oranı 🎯",
            f"{bonus_conversion:.1f}%",
            "GGR Pozitif Oran"
        )

    # Analiz Türü Seçimi
    analysis_type = st.radio(
        "📊 Analiz Türü Seçin",
        ["Bonus Etkinliği", "Segment Analizi", "Trend Analizi", "Oyun Tipi Analizi"],
        horizontal=True
    )

    if analysis_type == "Bonus Etkinliği":
        st.subheader("Bonus Etkinlik Analizi 📊")

        # Bonus-GGR İlişkisi
        fig_scatter = px.scatter(
            data[data['BonusKullanimi'] > 0],
            x='BonusKullanimi',
            y='GGR',
            color='OyunTipi',
            size='ToplamDepozit',
            trendline="ols",
            title='Bonus-GGR İlişkisi'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Bonus Kullanım Dağılımı
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=data[data['BonusKullanimi'] > 0]['BonusKullanimi'],
            nbinsx=50,
            name='Bonus Dağılımı'
        ))
        fig_dist.update_layout(
            title='Bonus Kullanım Dağılımı',
            xaxis_title='Bonus Miktarı (₺)',
            yaxis_title='Kullanıcı Sayısı',
            template='plotly_dark'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    elif analysis_type == "Segment Analizi":
        st.subheader("Bonus Segment Analizi 👥")

        # Bonus segmentleri
        data['BonusSegment'] = pd.qcut(
            data['BonusKullanimi'].clip(lower=0),
            q=4,
            labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek']
        )

        segment_analysis = data.groupby('BonusSegment').agg({
            'OyuncuID': 'count',
            'BonusKullanimi': ['sum', 'mean'],
            'GGR': ['sum', 'mean'],
            'ToplamDepozit': 'mean'
        }).round(2)

        # Segment görselleştirmesi
        fig_segment = px.sunburst(
            data,
            path=['BonusSegment', 'OyunTipi'],
            values='BonusKullanimi',
            title='Bonus Segment Dağılımı'
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        # Segment metrikleri
        st.dataframe(
            segment_analysis.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )

    elif analysis_type == "Trend Analizi":
        st.subheader("Bonus Trend Analizi 📈")

        # Günlük bonus trendi
        daily_bonus = data.groupby(
            pd.to_datetime(data['KayitTarihi']).dt.date
        ).agg({
            'BonusKullanimi': ['sum', 'mean', 'count'],
            'GGR': 'sum'
        }).reset_index()

        fig_trend = go.Figure()

        # Toplam bonus
        fig_trend.add_trace(go.Scatter(
            x=daily_bonus['KayitTarihi'],
            y=daily_bonus['BonusKullanimi']['sum'],
            name='Toplam Bonus',
            line=dict(color='#2ecc71')
        ))

        # 7 günlük ortalama
        fig_trend.add_trace(go.Scatter(
            x=daily_bonus['KayitTarihi'],
            y=daily_bonus['BonusKullanimi']['sum'].rolling(7).mean(),
            name='7 Günlük Ortalama',
            line=dict(color='#e74c3c', dash='dash')
        ))

        fig_trend.update_layout(
            title='Bonus Kullanım Trendi',
            xaxis_title='Tarih',
            yaxis_title='Bonus Miktarı (₺)',
            template='plotly_dark'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Bonus kullanıcı sayısı trendi
        fig_users = go.Figure()
        fig_users.add_trace(go.Scatter(
            x=daily_bonus['KayitTarihi'],
            y=daily_bonus['BonusKullanimi']['count'],
            name='Bonus Kullanan Oyuncular',
            fill='tozeroy'
        ))
        fig_users.update_layout(
            title='Bonus Kullanan Oyuncu Sayısı Trendi',
            xaxis_title='Tarih',
            yaxis_title='Oyuncu Sayısı',
            template='plotly_dark'
        )
        st.plotly_chart(fig_users, use_container_width=True)

    else:  # Oyun Tipi Analizi
        st.subheader("Oyun Tipi Bazlı Bonus Analizi 🎮")

        # Oyun tipi bazlı analiz
        game_bonus = data.groupby('OyunTipi').agg({
            'BonusKullanimi': ['sum', 'mean', 'count'],
            'GGR': 'sum',
            'OyuncuID': 'count'
        }).round(2)

        game_bonus['Bonus_ROI'] = ((game_bonus['GGR']['sum'] - game_bonus['BonusKullanimi']['sum']) /
                                   game_bonus['BonusKullanimi']['sum'] * 100)
        

        # Oyun tipi dağılımı
        fig_game = px.bar(
            game_bonus.reset_index(),
            x='OyunTipi',
            y=('BonusKullanimi', 'sum'),
            title='Oyun Tipi Bazlı Bonus Dağılımı'
        )
        st.plotly_chart(fig_game, use_container_width=True)

        # ROI Karşılaştırması
        fig_roi = px.bar(
            x=game_bonus.index,
            y=game_bonus['Bonus_ROI'],
            title='Oyun Tipi Bazlı Bonus ROI',
            labels={'x': 'Oyun Tipi', 'y': 'ROI (%)'}
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Detaylı metrikler
        st.dataframe(
            game_bonus.style.background_gradient(subset=[('BonusKullanimi', 'sum')], cmap='Greens'),
            use_container_width=True
        )

    # Bonus Önerileri
    st.subheader("Bonus Optimizasyon Önerileri 💡")

    # En iyi ve en kötü performans gösteren bonus segmentleri
    best_roi = game_bonus['Bonus_ROI'].idxmax()
    worst_roi = game_bonus['Bonus_ROI'].idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        ✅ En İyi Performans: {best_roi}
        - ROI: {game_bonus.loc[best_roi, 'Bonus_ROI']:.1f}%
        - Toplam Bonus: ₺{game_bonus.loc[best_roi, ('BonusKullanimi', 'sum')]:,.2f}

        Öneri: Bonus bütçesini artırın
        """)

    with col2:
        st.warning(f"""
        ⚠️ En Düşük Performans: {worst_roi}
        - ROI: {game_bonus.loc[worst_roi, 'Bonus_ROI']:.1f}%
        - Toplam Bonus: ₺{game_bonus.loc[worst_roi, ('BonusKullanimi', 'sum')]:,.2f}

        Öneri: Bonus stratejisini gözden geçirin
        """)

    # Rapor İndirme
    st.sidebar.markdown("### 📥 Bonus Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    data.to_excel(writer, sheet_name='Bonus Data')
                    daily_bonus.to_excel(writer, sheet_name='Daily Trends')
                    game_bonus.to_excel(writer, sheet_name='Game Analysis')
                    segment_analysis.to_excel(writer, sheet_name='Segment Analysis')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="bonus_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_trend_analysis(data: pd.DataFrame):
    """Trend analizi bölümünü gösterir."""
    st.subheader("Trend Analizi 📈")

    # Info expander
    with st.expander("ℹ️ Trend Analizi Hakkında", expanded=False):
        st.info("""
        ### Trend Analizi Özellikleri
        - 📊 Zaman bazlı performans analizi
        - 🔄 Dönemsel değişimlerin tespiti
        - 📈 Büyüme ve düşüş trendleri
        - 🎯 Mevsimsellik analizi
        - 💡 Tahminleme
        """)

    # Metrik seçimi
    col1, col2 = st.columns(2)
    with col1:
        selected_metric = st.selectbox(
            "Analiz Metriği",
            ["GGR", "BahisSayisi", "BonusKullanimi", "GirisSikligi", "RiskSkoru"]
        )

    with col2:
        period = st.selectbox(
            "Analiz Periyodu",
            ["Günlük", "Haftalık", "Aylık"]
        )

    # Trend tipi seçimi
    trend_type = st.radio(
        "Trend Analiz Türü",
        ["Zaman Serisi", "Büyüme Analizi", "Mevsimsellik", "Karşılaştırmalı Analiz"],
        horizontal=True
    )

    if trend_type == "Zaman Serisi":
        st.subheader(f"{selected_metric} Zaman Serisi Analizi")

        # Zaman serisi verisi hazırlama
        if period == "Günlük":
            grouper = pd.Grouper(key='KayitTarihi', freq='D')
        elif period == "Haftalık":
            grouper = pd.Grouper(key='KayitTarihi', freq='W')
        else:
            grouper = pd.Grouper(key='KayitTarihi', freq='M')

        time_series = (
            data.set_index('KayitTarihi')
            .groupby(grouper)[selected_metric]
            .agg(['sum', 'mean', 'count'])
            .reset_index()
        )

        # Ana trend grafiği
        fig = go.Figure()

        # Gerçek değerler
        fig.add_trace(go.Scatter(
            x=time_series['KayitTarihi'],
            y=time_series['sum'],
            name=f'{period} {selected_metric}',
            line=dict(color='blue', width=2)
        ))

        # Hareketli ortalama
        window = 7 if period == "Günlük" else 4
        fig.add_trace(go.Scatter(
            x=time_series['KayitTarihi'],
            y=time_series['sum'].rolling(window=window).mean(),
            name=f'{window} {period} Ortalama',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'{selected_metric} Trend Analizi',
            xaxis_title='Tarih',
            yaxis_title=selected_metric,
            template='plotly_dark',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    elif trend_type == "Büyüme Analizi":
        st.subheader(f"{selected_metric} Büyüme Analizi")

        # Büyüme hesaplama
        growth_df = (
            data.groupby(pd.Grouper(key='KayitTarihi', freq='M'))
            [selected_metric].sum()
            .pct_change()
            .mul(100)
            .reset_index()
        )

        # Büyüme grafiği
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=growth_df['KayitTarihi'],
            y=growth_df[selected_metric],
            name='Aylık Büyüme (%)',
            marker_color=np.where(growth_df[selected_metric] > 0, 'green', 'red')
        ))

        fig.update_layout(
            title=f'{selected_metric} Aylık Büyüme Oranları',
            xaxis_title='Ay',
            yaxis_title='Büyüme Oranı (%)',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Kümülatif büyüme
        total_growth = ((data.groupby(pd.Grouper(key='KayitTarihi', freq='M'))[selected_metric].sum().iloc[-1] /
                         data.groupby(pd.Grouper(key='KayitTarihi', freq='M'))[selected_metric].sum().iloc[
                             0] - 1) * 100)

        st.metric(
            "Toplam Büyüme",
            f"{total_growth:.1f}%",
            "Başlangıçtan bugüne"
        )

    elif trend_type == "Mevsimsellik":
        st.subheader(f"{selected_metric} Mevsimsellik Analizi")

        # Günlük pattern
        daily_pattern = (
            data.groupby(data['KayitTarihi'].dt.day_name())[selected_metric]
            .mean()
            .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        )

        # Haftalık pattern grafiği
        fig_daily = go.Figure(data=go.Bar(
            x=daily_pattern.index,
            y=daily_pattern.values,
            marker_color='lightblue'
        ))

        fig_daily.update_layout(
            title='Günlük Pattern',
            xaxis_title='Gün',
            yaxis_title=f'Ortalama {selected_metric}',
            template='plotly_dark'
        )

        st.plotly_chart(fig_daily, use_container_width=True)

        # Aylık pattern
        monthly_pattern = (
            data.groupby(data['KayitTarihi'].dt.month_name())[selected_metric]
            .mean()
            .reindex(['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December'])
        )

        # Aylık pattern grafiği
        fig_monthly = go.Figure(data=go.Bar(
            x=monthly_pattern.index,
            y=monthly_pattern.values,
            marker_color='lightgreen'
        ))

        fig_monthly.update_layout(
            title='Aylık Pattern',
            xaxis_title='Ay',
            yaxis_title=f'Ortalama {selected_metric}',
            template='plotly_dark'
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

    else:  # Karşılaştırmalı Analiz
        st.subheader("Karşılaştırmalı Trend Analizi")

        # Metrik seçimi
        compare_metric = st.selectbox(
            "Karşılaştırma Metriği",
            [m for m in ["GGR", "BahisSayisi", "BonusKullanimi", "GirisSikligi", "RiskSkoru"]
             if m != selected_metric]
        )

        # Karşılaştırmalı trend
        fig = go.Figure()

        # İlk metrik
        fig.add_trace(go.Scatter(
            x=data['KayitTarihi'],
            y=data[selected_metric],
            name=selected_metric,
            yaxis='y'
        ))

        # İkinci metrik
        fig.add_trace(go.Scatter(
            x=data['KayitTarihi'],
            y=data[compare_metric],
            name=compare_metric,
            yaxis='y2'
        ))

        fig.update_layout(
            title=f'{selected_metric} vs {compare_metric}',
            xaxis_title='Tarih',
            yaxis_title=selected_metric,
            yaxis2=dict(
                title=compare_metric,
                overlaying='y',
                side='right'
            ),
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Korelasyon analizi
        corr = data[[selected_metric, compare_metric]].corr().iloc[0, 1]
        st.metric(
            "Korelasyon Katsayısı",
            f"{corr:.2f}",
            "Güçlü İlişki" if abs(corr) > 0.7 else "Zayıf İlişki"
        )

    # Trend Özeti
    st.subheader("Trend Özeti 📋")

    # Son dönem performansı
    last_period = data.groupby(pd.Grouper(key='KayitTarihi', freq='M'))[selected_metric].sum().iloc[-1]
    prev_period = data.groupby(pd.Grouper(key='KayitTarihi', freq='M'))[selected_metric].sum().iloc[-2]
    change = ((last_period / prev_period) - 1) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Son Dönem Performansı",
            f"{last_period:,.0f}",
            f"{change:+.1f}%"
        )

    with col2:
        st.metric(
            "Trend Yönü",
            "Yükseliş" if change > 0 else "Düşüş",
            f"{abs(change):.1f}% Değişim"
        )

    with col3:
        st.metric(
            "Volatilite",
            f"{data[selected_metric].std() / data[selected_metric].mean() * 100:.1f}%",
            "Standart Sapma / Ortalama"
        )

    # Rapor İndirme
    st.sidebar.markdown("### 📥 Trend Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    time_series.to_excel(writer, sheet_name='Time Series')
                    growth_df.to_excel(writer, sheet_name='Growth Analysis')
                    daily_pattern.to_frame().to_excel(writer, sheet_name='Daily Pattern')
                    monthly_pattern.to_frame().to_excel(writer, sheet_name='Monthly Pattern')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="trend_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_cohort_analysis(data: pd.DataFrame):
    """Cohort analizi bölümünü gösterir."""
    st.subheader("Cohort Analizi 📊")

    with st.expander("ℹ️ Cohort Analizi Hakkında", expanded=False):
        st.info("""
        ### Cohort Analizi Nedir?
        - 👥 Belirli bir zaman diliminde sisteme katılan kullanıcı gruplarının analizi
        - 📈 Zaman içindeki davranış değişikliklerinin takibi
        - 🎯 Kullanıcı yaşam döngüsü analizi
        - 💰 Gelir ve retention metriklerinin cohort bazlı incelenmesi

        Bu analiz, farklı zamanlarda katılan kullanıcı gruplarının karşılaştırmalı performansını gösterir.
        """)

    # Cohort tipi seçimi
    cohort_type = st.radio(
        "Cohort Analiz Türü",
        ["Retention", "GGR", "Aktivite", "LTV"],
        horizontal=True
    )

    # Zaman periyodu seçimi
    months_to_analyze = st.slider(
        "Analiz Periyodu (Ay)",
        min_value=1,
        max_value=12,
        value=6
    )

    # Cohort verisi hazırlama
    data['CohortMonth'] = pd.to_datetime(data['KayitTarihi']).dt.strftime('%Y-%m')
    data['CohortIndex'] = (pd.to_datetime(data['KayitTarihi']).dt.to_period('M') -
                           data['CohortMonth']).apply(lambda x: x.n)

    if cohort_type == "Retention":
        # Retention matrisi oluşturma
        cohort_data = data.groupby(['CohortMonth', 'CohortIndex'])['OyuncuID'].nunique().reset_index()
        cohort_matrix = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='OyuncuID')
        retention_matrix = cohort_matrix.div(cohort_matrix[0], axis=0) * 100

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=retention_matrix.values,
            x=['Ay ' + str(i) for i in retention_matrix.columns],
            y=[str(i) for i in retention_matrix.index],
            colorscale='RdYlBu_r',
            text=np.round(retention_matrix.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Aylık Retention Oranları (%)',
            xaxis_title='Ay',
            yaxis_title='Cohort',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Retention metrikleri
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'retention_matrix' in locals() and 1 in retention_matrix.columns:
             st.metric(
                 "1. Ay Retention",
                 f"{retention_matrix[1].mean():.1f}%",
                 "Ortalama"
             )
        with col2:
            st.metric(
                "3. Ay Retention",
                f"{retention_matrix[3].mean():.1f}%",
                "Ortalama"
            )
        with col3:
            st.metric(
                "En İyi Cohort",
                f"{retention_matrix.index[retention_matrix[1].argmax()]}",
                f"{retention_matrix[1].max():.1f}% (1. Ay)"
            )

    elif cohort_type == "GGR":
        # GGR cohort analizi
        ggr_data = data.groupby(['CohortMonth', 'CohortIndex'])['GGR'].mean().reset_index()
        ggr_matrix = ggr_data.pivot(index='CohortMonth', columns='CohortIndex', values='GGR')

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=ggr_matrix.values,
            x=['Ay ' + str(i) for i in ggr_matrix.columns],
            y=[str(i) for i in ggr_matrix.index],
            colorscale='Viridis',
            text=np.round(ggr_matrix.values, 0),
            texttemplate='₺%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Cohort Bazlı Ortalama GGR',
            xaxis_title='Ay',
            yaxis_title='Cohort',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # GGR trend analizi
        st.subheader("Cohort GGR Trendi")
        fig_trend = go.Figure()

        for cohort in ggr_matrix.index[-6:]:  # Son 6 cohort
            fig_trend.add_trace(go.Scatter(
                x=ggr_matrix.columns,
                y=ggr_matrix.loc[cohort],
                name=str(cohort),
                mode='lines+markers'
            ))

        fig_trend.update_layout(
            title='Son 6 Cohort GGR Trendi',
            xaxis_title='Ay',
            yaxis_title='Ortalama GGR (₺)',
            template='plotly_dark'
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    elif cohort_type == "Aktivite":
        # Aktivite cohort analizi
        activity_data = data.groupby(['CohortMonth', 'CohortIndex'])['BahisSayisi'].mean().reset_index()
        activity_matrix = activity_data.pivot(index='CohortMonth', columns='CohortIndex', values='BahisSayisi')

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activity_matrix.values,
            x=['Ay ' + str(i) for i in activity_matrix.columns],
            y=[str(i) for i in activity_matrix.index],
            colorscale='Viridis',
            text=np.round(activity_matrix.values, 1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Cohort Bazlı Ortalama Aktivite',
            xaxis_title='Ay',
            yaxis_title='Cohort',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Aktivite metrikleri
        col1, col2 = st.columns(2)
        with col1:
            # Aktivite trend grafiği
            fig_trend = px.line(
                activity_matrix.mean(),
                title='Ortalama Aktivite Trendi',
                template='plotly_dark'
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col2:
            # En aktif cohortlar
            fig_bar = px.bar(
                activity_matrix.mean(axis=1).sort_values(ascending=False),
                title='En Aktif Cohortlar',
                template='plotly_dark'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:  # LTV Analizi
        # LTV cohort analizi
        ltv_data = data.groupby(['CohortMonth', 'CohortIndex'])['GGR'].sum().reset_index()
        ltv_matrix = ltv_data.pivot(index='CohortMonth', columns='CohortIndex', values='GGR').cumsum(axis=1)

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=ltv_matrix.values,
            x=['Ay ' + str(i) for i in ltv_matrix.columns],
            y=[str(i) for i in ltv_matrix.index],
            colorscale='Viridis',
            text=np.round(ltv_matrix.values, 0),
            texttemplate='₺%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Cohort Bazlı Kümülatif LTV',
            xaxis_title='Ay',
            yaxis_title='Cohort',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # LTV projeksiyon analizi
        st.subheader("LTV Projeksiyon Analizi")
        avg_ltv_curve = ltv_matrix.mean()

        fig_proj = go.Figure()

        fig_proj.add_trace(go.Scatter(
            x=list(range(len(avg_ltv_curve))),
            y=avg_ltv_curve.values,
            name='Gerçek LTV',
            mode='lines+markers'
        ))

        # Basit projeksiyon (son 3 ay ortalaması)
        if len(avg_ltv_curve) >= 3:
            last_3_growth = avg_ltv_curve.pct_change().tail(3).mean()
            projected_values = [avg_ltv_curve.iloc[-1]]
            for _ in range(3):
                projected_values.append(projected_values[-1] * (1 + last_3_growth))

            fig_proj.add_trace(go.Scatter(
                x=list(range(len(avg_ltv_curve) - 1, len(avg_ltv_curve) + 3)),
                y=projected_values,
                name='Projeksiyon',
                line=dict(dash='dash')
            ))

        fig_proj.update_layout(
            title='LTV Projeksiyon Analizi',
            xaxis_title='Ay',
            yaxis_title='Kümülatif LTV (₺)',
            template='plotly_dark'
        )

        st.plotly_chart(fig_proj, use_container_width=True)

    # Cohort performans özeti
    st.subheader("Cohort Performans Özeti")

    if cohort_type == "Retention":
        best_cohort = retention_matrix.index[retention_matrix[1].argmax()]
        best_retention = retention_matrix[1].max()

        st.success(f"""
        ✅ En İyi Performans Gösteren Cohort:
        - Cohort: {best_cohort}
        - 1. Ay Retention: {best_retention:.1f}%
        - Önerilen Aksiyon: Bu cohorttaki kullanıcı deneyimini inceleyin ve diğer cohortlara uygulayın
        """)

    elif cohort_type in ["GGR", "LTV"]:
        best_cohort = ltv_matrix.index[ltv_matrix.iloc[:, -1].argmax()]
        best_ltv = ltv_matrix.iloc[:, -1].max()

        st.success(f"""
        ✅ En Yüksek Değer Yaratan Cohort:
        - Cohort: {best_cohort}
        - Kümülatif Değer: ₺{best_ltv:,.2f}
        - Önerilen Aksiyon: Bu cohorttaki kullanıcı segmentlerini analiz edin
        """)

    # Rapor İndirme
    st.sidebar.markdown("### 📥 Cohort Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    if cohort_type == "Retention":
                        retention_matrix.to_excel(writer, sheet_name='Retention Matrix')
                    elif cohort_type == "GGR":
                        ggr_matrix.to_excel(writer, sheet_name='GGR Matrix')
                    elif cohort_type == "Aktivite":
                        activity_matrix.to_excel(writer, sheet_name='Activity Matrix')
                    else:
                        ltv_matrix.to_excel(writer, sheet_name='LTV Matrix')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="cohort_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_ab_test_analysis(data: pd.DataFrame):
    """A/B test analizi bölümünü gösterir."""
    st.subheader("A/B Test Analizi 🔬")

    with st.expander("ℹ️ A/B Test Analizi Hakkında", expanded=False):
        st.info("""
        ### A/B Test Analizi Nedir?
        - 🔍 İki farklı grup arasındaki performans farklarını ölçer
        - 📊 İstatistiksel anlamlılık testleri kullanır
        - 🎯 Hangi yaklaşımın daha etkili olduğunu belirler
        - 📈 Veri odaklı karar vermeyi sağlar

        Test Metrikleri:
        - GGR (Gross Gaming Revenue)
        - Bahis Sayısı
        - Bonus Kullanımı
        - Aktivite Oranı
        """)

    # Test konfigürasyonu
    col1, col2 = st.columns(2)

    with col1:
        test_metric = st.selectbox(
            "Test Metriği",
            ["GGR", "BahisSayisi", "BonusKullanimi", "GirisSikligi"]
        )

        confidence_level = st.selectbox(
            "Güven Düzeyi",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"%{int(x * 100)}"
        )

    with col2:
        split_ratio = st.slider(
            "Test/Kontrol Grup Oranı",
            min_value=0.1,
            max_value=0.5,
            value=0.5,
            step=0.1
        )

        min_sample_size = st.number_input(
            "Minimum Örnek Boyutu",
            min_value=100,
            value=1000,
            step=100
        )

    # Test gruplarını oluştur
    np.random.seed(42)
    data['group'] = np.random.choice(
        ['Control', 'Test'],
        size=len(data),
        p=[1 - split_ratio, split_ratio]
    )

    # Test sonuçlarını hesapla
    control_data = data[data['group'] == 'Control'][test_metric]
    test_data = data[data['group'] == 'Test'][test_metric]

    # T-test uygula
    t_stat, p_value = stats.ttest_ind(control_data, test_data)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(control_data) + np.var(test_data)) / 2)
    cohens_d = (np.mean(test_data) - np.mean(control_data)) / pooled_std

    # Güven aralıkları
    control_ci = stats.t.interval(
        confidence_level,
        len(control_data) - 1,
        loc=np.mean(control_data),
        scale=stats.sem(control_data)
    )

    test_ci = stats.t.interval(
        confidence_level,
        len(test_data) - 1,
        loc=np.mean(test_data),
        scale=stats.sem(test_data)
    )

    # Sonuçları göster
    st.subheader("Test Sonuçları")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "İstatistiksel Anlamlılık",
            "Var" if p_value < (1 - confidence_level) else "Yok",
            f"p-value: {p_value:.4f}"
        )

    with col2:
        st.metric(
            "Etki Büyüklüğü",
            f"{abs(cohens_d):.2f}",
            "Cohen's d"
        )

    with col3:
        uplift = ((test_data.mean() / control_data.mean()) - 1) * 100
        st.metric(
            "Değişim Oranı",
            f"{uplift:.1f}%",
            "Test vs Control"
        )

    # Dağılım karşılaştırması
    st.subheader("Grup Dağılımları")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=control_data,
        name='Kontrol Grubu',
        nbinsx=30,
        opacity=0.7
    ))

    fig.add_trace(go.Histogram(
        x=test_data,
        name='Test Grubu',
        nbinsx=30,
        opacity=0.7
    ))

    fig.add_vline(
        x=control_data.mean(),
        line_dash="dash",
        annotation_text="Kontrol Ort."
    )

    fig.add_vline(
        x=test_data.mean(),
        line_dash="dash",
        annotation_text="Test Ort."
    )

    fig.update_layout(
        title=f'{test_metric} Dağılımı: Test vs Kontrol',
        xaxis_title=test_metric,
        yaxis_title='Frekans',
        barmode='overlay',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Güven aralıkları gösterimi
    st.subheader("Güven Aralıkları")

    fig_ci = go.Figure()

    fig_ci.add_trace(go.Box(
        y=control_data,
        name='Kontrol Grubu',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))

    fig_ci.add_trace(go.Box(
        y=test_data,
        name='Test Grubu',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))

    fig_ci.update_layout(
        title=f'Güven Aralıkları ve Dağılım ({int(confidence_level * 100)}%)',
        yaxis_title=test_metric,
        template='plotly_dark'
    )

    st.plotly_chart(fig_ci, use_container_width=True)

    # Detaylı istatistikler
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Kontrol Grubu İstatistikleri")
        st.write(f"Örneklem Boyutu: {len(control_data)}")
        st.write(f"Ortalama: {control_data.mean():.2f}")
        st.write(f"Standart Sapma: {control_data.std():.2f}")
        st.write(f"Güven Aralığı: [{control_ci[0]:.2f}, {control_ci[1]:.2f}]")

    with col2:
        st.markdown("### Test Grubu İstatistikleri")
        st.write(f"Örneklem Boyutu: {len(test_data)}")
        st.write(f"Ortalama: {test_data.mean():.2f}")
        st.write(f"Standart Sapma: {test_data.std():.2f}")
        st.write(f"Güven Aralığı: [{test_ci[0]:.2f}, {test_ci[1]:.2f}]")

    # Test önerileri
    st.subheader("Test Sonuç Değerlendirmesi")

    if p_value < (1 - confidence_level):
        if uplift > 0:
            st.success(f"""
            ✅ Test grubu istatistiksel olarak anlamlı bir şekilde daha iyi performans gösterdi:
            - {uplift:.1f}% performans artışı
            - {confidence_level * 100}% güven düzeyi
            - {cohens_d:.2f} etki büyüklüğü

            **Önerilen Aksiyon:** Test grubundaki değişiklikleri uygulamaya geçirin.
            """)
        else:
            st.error(f"""
            ❌ Test grubu istatistiksel olarak anlamlı bir şekilde daha kötü performans gösterdi:
            - {abs(uplift):.1f}% performans düşüşü
            - {confidence_level * 100}% güven düzeyi
            - {cohens_d:.2f} etki büyüklüğü

            **Önerilen Aksiyon:** Mevcut sistemi koruyun ve yeni test senaryoları geliştirin.
            """)
    else:
        st.warning(f"""
        ⚠️ Test ve kontrol grupları arasında istatistiksel olarak anlamlı bir fark bulunamadı:
        - {uplift:.1f}% fark
        - p-value: {p_value:.4f}
        - Minimum gerekli örneklem boyutu: {min_sample_size:,}

        **Önerilen Aksiyon:** 
        1. Testi daha büyük bir örneklem ile tekrarlayın
        2. Test süresini uzatın
        3. Farklı değişkenler test edin
        """)

    # Rapor İndirme
    st.sidebar.markdown("### 📥 A/B Test Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()

                # Test sonuçlarını DataFrame'e dönüştür
                results_df = pd.DataFrame({
                    'Metric': ['Sample Size', 'Mean', 'Std Dev', 'CI Lower', 'CI Upper'],
                    'Control': [
                        len(control_data),
                        control_data.mean(),
                        control_data.std(),
                        control_ci[0],
                        control_ci[1]
                    ],
                    'Test': [
                        len(test_data),
                        test_data.mean(),
                        test_data.std(),
                        test_ci[0],
                        test_ci[1]
                    ]
                })

                with pd.ExcelWriter(buffer) as writer:
                    # Test sonuçları
                    results_df.to_excel(writer, sheet_name='Test Results')

                    # Ham veriler
                    pd.DataFrame({
                        'Group': ['Control'] * len(control_data) + ['Test'] * len(test_data),
                        'Value': pd.concat([control_data, test_data])
                    }).to_excel(writer, sheet_name='Raw Data')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="ab_test_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_anova_analysis(data: pd.DataFrame, groups=None):
    """ANOVA analizi bölümünü gösterir."""
    st.subheader("ANOVA (Varyans) Analizi 📊")
    from scipy import stats
    f_stat, p_value = stats.f_oneway(*groups)

    with st.expander("ℹ️ ANOVA Analizi Hakkında", expanded=False):
        st.info("""
        ### ANOVA (Varyans Analizi) Nedir?
        - 📈 İkiden fazla grup arasındaki farklılıkları analiz eder
        - 🔍 Gruplar arası ve grup içi varyansları karşılaştırır
        - 📊 İstatistiksel anlamlılık testleri uygular
        - 🎯 Hangi grupların birbirinden farklı olduğunu belirler

        Analiz Türleri:
        - Oyun Tipi Analizi
        - Şehir Bazlı Analiz
        - Risk Segmenti Analizi
        """)

    # Analiz konfigürasyonu
    col1, col2 = st.columns(2)

    with col1:
        analysis_metric = st.selectbox(
            "Analiz Metriği",
            ["GGR", "BahisSayisi", "BonusKullanimi", "GirisSikligi", "RiskSkoru"]
        )

        group_variable = st.selectbox(
            "Gruplama Değişkeni",
            ["OyunTipi", "Sehir", "RiskSegment"]
        )

    with col2:
        significance_level = st.selectbox(
            "Anlamlılık Düzeyi",
            [0.01, 0.05, 0.10],
            index=1,
            format_func=lambda x: f"% {x * 100}"
        )

        min_group_size = st.number_input(
            "Minimum Grup Boyutu",
            min_value=10,
            value=30,
            step=10
        )

    # Risk segmentlerini oluştur (eğer seçildiyse)
    if group_variable == 'RiskSegment':
        data['RiskSegment'] = pd.qcut(
            data['RiskSkoru'],
            q=4,
            labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek']
        )

    # Grup bazlı veriler
    grouped_data = data.groupby(group_variable)[analysis_metric].agg(['count', 'mean', 'std'])
    valid_groups = grouped_data[grouped_data['count'] >= min_group_size].index

    # Yeterli veri olan grupları filtrele
    filtered_data = data[data[group_variable].isin(valid_groups)]

    # ANOVA testi için grupları hazırla
    groups = [group[analysis_metric].values for name, group in filtered_data.groupby(group_variable)]

    # ANOVA testi uygula
    f_stat, p_value = stats.f_oneway(*groups)

    # Etki büyüklüğü (Eta-squared) hesapla
    df_between = len(groups) - 1
    df_within = sum(len(group) - 1 for group in groups)
    ss_between = sum(len(group) * (group.mean() - filtered_data[analysis_metric].mean()) ** 2 for group in groups)
    ss_total = sum((filtered_data[analysis_metric] - filtered_data[analysis_metric].mean()) ** 2)
    eta_squared = ss_between / ss_total

    # Temel metrikleri göster
    st.subheader("ANOVA Test Sonuçları")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "F-İstatistiği",
            f"{f_stat:.2f}",
            f"p-value: {p_value:.4f}"
        )

    with col2:
        st.metric(
            "Etki Büyüklüğü",
            f"{eta_squared:.3f}",
            "Eta-kare"
        )

    with col3:
        st.metric(
            "Anlamlı Fark",
            "Var" if p_value < significance_level else "Yok",
            f"α = {significance_level}"
        )

    # Grup istatistiklerini göster
    st.subheader("Grup İstatistikleri")

    group_stats = filtered_data.groupby(group_variable)[analysis_metric].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)

    st.dataframe(
        group_stats.style.background_gradient(subset=['mean'], cmap='YlOrRd'),
        use_container_width=True
    )

    # Box plot ile grup karşılaştırması
    fig_box = go.Figure()

    for group_name in valid_groups:
        group_data = filtered_data[filtered_data[group_variable] == group_name][analysis_metric]

        fig_box.add_trace(go.Box(
            y=group_data,
            name=str(group_name),
            boxpoints='outliers'
        ))

    fig_box.update_layout(
        title=f'{analysis_metric} Dağılımı - Grup Bazlı',
        yaxis_title=analysis_metric,
        template='plotly_dark'
    )

    st.plotly_chart(fig_box, use_container_width=True)

    # Tukey's HSD testi
    if p_value < significance_level:
        st.subheader("Post-hoc Analiz (Tukey HSD)")

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey = pairwise_tukeyhsd(
            filtered_data[analysis_metric],
            filtered_data[group_variable],
            alpha=significance_level
        )

        # Anlamlı farklılıkları göster
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

        sig_diff = tukey_df[tukey_df['p-adj'] < significance_level]

        if not sig_diff.empty:
            st.write("**Anlamlı Farklılıklar:**")
            for _, row in sig_diff.iterrows():
                st.write(f"• {row['group1']} vs {row['group2']}: " +
                         f"Fark = {row['meandiff']:.2f}, " +
                         f"p = {row['p-adj']:.4f}")
        else:
            st.info("Gruplar arasında istatistiksel olarak anlamlı bir fark bulunamadı.")

        # Means plot with confidence intervals
        fig_means = go.Figure()

        group_means = filtered_data.groupby(group_variable)[analysis_metric].agg(['mean', 'std', 'count'])

        for idx, (group, stats) in enumerate(group_means.iterrows()):
            ci = stats['std'] / np.sqrt(stats['count']) * stats.t.ppf(0.975, stats['count'] - 1)

            fig_means.add_trace(go.Scatter(
                x=[idx],
                y=[stats['mean']],
                error_y=dict(
                    type='data',
                    array=[ci],
                    visible=True
                ),
                name=group,
                mode='markers',
                marker=dict(size=12)
            ))

        fig_means.update_layout(
            title='Grup Ortalamaları ve Güven Aralıkları',
            xaxis=dict(
                ticktext=list(group_means.index),
                tickvals=list(range(len(group_means))),
                title=group_variable
            ),
            yaxis_title=analysis_metric,
            template='plotly_dark'
        )

        st.plotly_chart(fig_means, use_container_width=True)

    # Analiz özeti
    st.subheader("Analiz Özeti")

    if p_value < significance_level:
        st.success(f"""
        ✅ ANOVA analizi sonuçlarına göre, {group_variable} grupları arasında 
        {analysis_metric} açısından istatistiksel olarak anlamlı bir fark bulunmuştur 
        (F = {f_stat:.2f}, p = {p_value:.4f}).

        Etki büyüklüğü (η² = {eta_squared:.3f}) 
        {' düşük' if eta_squared < 0.06 else ' orta' if eta_squared < 0.14 else ' yüksek'}
        düzeydedir.

        **Öneriler:**
        1. En yüksek performans gösteren grupların özelliklerini inceleyin
        2. Düşük performans gösteren gruplar için özel stratejiler geliştirin
        3. Grup farklılıklarının nedenlerini araştırın
        """)
    else:
        st.warning(f"""
        ⚠️ ANOVA analizi sonuçlarına göre, {group_variable} grupları arasında 
        {analysis_metric} açısından istatistiksel olarak anlamlı bir fark bulunamamıştır 
        (F = {f_stat:.2f}, p = {p_value:.4f}).

        **Öneriler:**
        1. Daha büyük örneklem boyutu ile testi tekrarlayın
        2. Farklı gruplama değişkenleri deneyin
        3. Alternatif metrikler üzerinde analiz yapın
        """)

    # Rapor İndirme
    st.sidebar.markdown("### 📥 ANOVA Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    # Grup istatistikleri
                    group_stats.to_excel(writer, sheet_name='Group Statistics')

                    # ANOVA sonuçları
                    pd.DataFrame({
                        'Metric': ['F-statistic', 'p-value', 'eta-squared'],
                        'Value': [f_stat, p_value, eta_squared]
                    }).to_excel(writer, sheet_name='ANOVA Results')

                    if p_value < significance_level:
                        # Tukey sonuçları
                        tukey_df.to_excel(writer, sheet_name='Tukey Results')

                    # Ham veriler
                    filtered_data[[group_variable, analysis_metric]].to_excel(
                        writer, sheet_name='Raw Data'
                    )

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="anova_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_roi_analysis(data: pd.DataFrame):
    """ROI (Return on Investment) analizi bölümünü gösterir."""
    st.subheader("ROI (Yatırım Getirisi) Analizi 💰")

    with st.expander("ℹ️ ROI Analizi Hakkında", expanded=False):
        st.info("""
        ### ROI (Return on Investment) Nedir?
        - 💰 Yatırımların geri dönüş oranını ölçer
        - 📊 Farklı yatırım tiplerinin performansını karşılaştırır
        - 📈 Trend analizi ve tahminleme yapar
        - 🎯 Kaynak optimizasyonu için kullanılır

        ROI Hesaplama:
        ```
        ROI = ((Gelir - Maliyet) / Maliyet) × 100
        ```
        """)

    # ROI analiz türü seçimi
    analysis_type = st.radio(
        "ROI Analiz Türü",
        ["Genel ROI", "Bonus ROI", "Oyun Tipi ROI", "Segment ROI"],
        horizontal=True
    )

    if analysis_type == "Genel ROI":
        show_general_roi(data)
    elif analysis_type == "Bonus ROI":
        show_bonus_roi(data)
    elif analysis_type == "Oyun Tipi ROI":
        show_game_type_roi(data)
    else:
        show_segment_roi(data)


def show_general_roi(data: pd.DataFrame):
    """Genel ROI analizini gösterir."""
    # Genel ROI metrikleri
    total_revenue = data['GGR'].sum()
    total_cost = data['BonusKullanimi'].sum()
    total_roi = ((total_revenue - total_cost) / total_cost * 100)
    profit_margin = ((total_revenue - total_cost) / total_revenue * 100)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Toplam ROI",
            f"{total_roi:.1f}%",
            "Yatırım Getirisi"
        )

    with col2:
        st.metric(
            "Kar Marjı",
            f"{profit_margin:.1f}%",
            f"₺{total_revenue - total_cost:,.2f}"
        )

    with col3:
        st.metric(
            "Toplam Gelir",
            f"₺{total_revenue:,.2f}",
            "GGR"
        )

    with col4:
        st.metric(
            "Toplam Maliyet",
            f"₺{total_cost:,.2f}",
            "Bonuslar"
        )

    # Zaman bazlı ROI trendi
    st.subheader("ROI Trend Analizi")

    daily_roi = (
        data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date)
        .agg({
            'GGR': 'sum',
            'BonusKullanimi': 'sum'
        })
        .assign(ROI=lambda x: ((x['GGR'] - x['BonusKullanimi']) / x['BonusKullanimi'] * 100))
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_roi.index,
        y=daily_roi['ROI'],
        name='Günlük ROI',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=daily_roi.index,
        y=daily_roi['ROI'].rolling(7).mean(),
        name='7 Günlük Ortalama',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='ROI Trendi',
        xaxis_title='Tarih',
        yaxis_title='ROI (%)',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)


def show_bonus_roi(data: pd.DataFrame):
    """Bonus bazlı ROI analizini gösterir."""
    st.subheader("Bonus ROI Analizi")

    # Bonus segmentleri
    data['BonusSegment'] = pd.qcut(
        data['BonusKullanimi'].clip(lower=0),
        q=4,
        labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek']
    )

    # Segment analizi
    segment_analysis = (
        data.groupby('BonusSegment')
        .agg({
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': ['sum', 'mean'],
            'OyuncuID': 'count'
        })
    )

    segment_analysis['ROI'] = (
            (segment_analysis['GGR']['sum'] - segment_analysis['BonusKullanimi']['sum']) /
            segment_analysis['BonusKullanimi']['sum'] * 100
    )

    # Bonus ROI görselleştirmesi
    fig = px.bar(
        segment_analysis.reset_index(),
        x='BonusSegment',
        y='ROI',
        title='Segment Bazlı Bonus ROI',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detaylı metrikler
    st.dataframe(
        segment_analysis.style.background_gradient(subset=[('ROI', '')], cmap='RdYlGn'),
        use_container_width=True
    )


def show_game_type_roi(data: pd.DataFrame):
    """Oyun tipi bazlı ROI analizini gösterir."""
    st.subheader("Oyun Tipi ROI Analizi")

    # Oyun tipi analizi
    game_analysis = (
        data.groupby('OyunTipi')
        .agg({
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': ['sum', 'mean'],
            'OyuncuID': 'count'
        })
    )

    game_analysis['ROI'] = (
            (game_analysis['GGR']['sum'] - game_analysis['BonusKullanimi']['sum']) /
            game_analysis['BonusKullanimi']['sum'] * 100
    )

    # ROI karşılaştırma grafiği
    fig = px.bar(
        game_analysis.reset_index(),
        x='OyunTipi',
        y='ROI',
        title='Oyun Tipi ROI Karşılaştırması',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detaylı metrikler
    st.dataframe(
        game_analysis.style.background_gradient(subset=[('ROI', '')], cmap='RdYlGn'),
        use_container_width=True
    )

    # En iyi ve en kötü performans
    best_game = game_analysis['ROI'].idxmax()
    worst_game = game_analysis['ROI'].idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        ✅ En Yüksek ROI: {best_game}
        - ROI: {game_analysis.loc[best_game, 'ROI']:.1f}%
        - Oyuncu Sayısı: {game_analysis.loc[best_game, ('OyuncuID', 'count')]:,}

        Öneri: Bu oyun tipine yatırımı artırın
        """)

    with col2:
        st.warning(f"""
        ⚠️ En Düşük ROI: {worst_game}
        - ROI: {game_analysis.loc[worst_game, 'ROI']:.1f}%
        - Oyuncu Sayısı: {game_analysis.loc[worst_game, ('OyuncuID', 'count')]:,}

        Öneri: Yatırım stratejisini gözden geçirin
        """)


def show_segment_roi(data: pd.DataFrame):
    """Segment bazlı ROI analizini gösterir."""
    st.subheader("Segment ROI Analizi")

    # Risk segmentleri
    data['RiskSegment'] = pd.qcut(
        data['RiskSkoru'],
        q=4,
        labels=['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek']
    )

    # Segment analizi
    segment_analysis = (
        data.groupby(['RiskSegment', 'OyunTipi'])
        .agg({
            'GGR': ['sum', 'mean'],
            'BonusKullanimi': ['sum', 'mean'],
            'OyuncuID': 'count'
        })
    )

    segment_analysis['ROI'] = (
            (segment_analysis['GGR']['sum'] - segment_analysis['BonusKullanimi']['sum']) /
            segment_analysis['BonusKullanimi']['sum'] * 100
    )

    # Segment ROI görselleştirmesi
    fig = px.sunburst(
        data,
        path=['RiskSegment', 'OyunTipi'],
        values='GGR',
        title='Segment ve Oyun Tipi ROI Dağılımı'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Segment bazlı karşılaştırma
    fig_compare = px.treemap(
        segment_analysis.reset_index(),
        path=[px.Constant("Tüm Segmentler"), 'RiskSegment', 'OyunTipi'],
        values=('GGR', 'sum'),
        title='Segment Bazlı GGR Dağılımı'
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Optimizasyon önerileri
    st.subheader("Segment Optimizasyon Önerileri")

    best_segments = (
        segment_analysis.sort_values(('ROI', ''), ascending=False)
        .head(3)
    )

    worst_segments = (
        segment_analysis.sort_values(('ROI', ''), ascending=True)
        .head(3)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ En Yüksek ROI Segmentleri")
        for idx in best_segments.index:
            st.write(f"""
            **{idx[0]} - {idx[1]}:**
            - ROI: {best_segments.loc[idx, 'ROI']:.1f}%
            - Oyuncu: {best_segments.loc[idx, ('OyuncuID', 'count')]:,}
            """)

    with col2:
        st.warning("⚠️ En Düşük ROI Segmentleri")
        for idx in worst_segments.index:
            st.write(f"""
            **{idx[0]} - {idx[1]}:**
            - ROI: {worst_segments.loc[idx, 'ROI']:.1f}%
            - Oyuncu: {worst_segments.loc[idx, ('OyuncuID', 'count')]:,}
            """)

    # Rapor İndirme
    st.sidebar.markdown("### 📥 ROI Raporu")
    report_type = st.sidebar.selectbox(
        "Rapor Formatı",
        ["Excel", "PDF", "CSV"]
    )

    if st.sidebar.button("Rapor Oluştur"):
        with st.spinner('Rapor hazırlanıyor...'):
            if report_type == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    # Genel ROI verileri
                    daily_roi.to_excel(writer, sheet_name='Daily ROI')

                    # Segment analizleri
                    segment_analysis.to_excel(writer, sheet_name='Segment Analysis')

                    # Oyun tipi analizleri
                    game_analysis.to_excel(writer, sheet_name='Game Analysis')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="roi_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )


def show_player_behavior(data: pd.DataFrame):
    st.subheader("Oyuncu Davranış Analizi 🎮")

    with st.expander("ℹ️ Davranış Analizi Hakkında", expanded=False):
        st.info("""
        ### Davranış Analiz Metrikleri
        - 🎯 **Aktivite Paterni**: Oyun sıklığı ve zamanlaması
        - 💰 **Bahis Davranışı**: Ortalama bahis ve risk profili
        - 🎮 **Oyun Tercihi**: Favori oyun tipleri ve geçişler
        - ⏰ **Zaman Analizi**: Günlük/haftalık aktivite dağılımı
        """)

    # Temel Metrikler
    active_players = len(data[data['SonAktivite'] < 7])
    avg_bets = data['BahisSayisi'].mean()
    avg_stake = data['OrtBahis'].mean()
    activity_rate = (data['GirisSikligi'] > 5).mean() * 100

    # KPI'lar
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Aktif Oyuncular 👥",
            f"{active_players:,}",
            f"{(active_players / len(data) * 100):.1f}% Oran"
        )

    with col2:
        st.metric(
            "Ortalama Bahis/Oyuncu 🎲",
            f"{avg_bets:.1f}",
            "Bahis Sayısı"
        )

    with col3:
        st.metric(
            "Ortalama Bahis Tutarı 💰",
            f"₺{avg_stake:.2f}",
            "Bahis Başına"
        )

    with col4:
        st.metric(
            "Yüksek Aktivite Oranı ⚡",
            f"{activity_rate:.1f}%",
            "5+ Giriş/Hafta"
        )

    # Analiz Türü Seçimi
    analysis_type = st.radio(
        "📊 Analiz Türü",
        ["Aktivite Analizi", "Oyun Tercihleri", "Davranış Paterni", "Geçiş Analizi"],
        horizontal=True
    )

    if analysis_type == "Aktivite Analizi":
        # Günlük aktivite heat map'i
        st.subheader("Günlük Aktivite Dağılımı")

        # Örnek veri oluştur (gerçek veride timestamp kullanılmalı)
        hour_data = pd.DataFrame(
            np.random.randint(0, 100, size=(24, 7)),
            columns=['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'],
            index=range(24)
        )

        fig = px.imshow(
            hour_data,
            title='Saatlik Aktivite Yoğunluğu',
            labels=dict(x="Gün", y="Saat", color="Aktivite"),
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Aktivite trendi
        daily_activity = data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date)['BahisSayisi'].sum().reset_index()

        fig_trend = px.line(
            daily_activity,
            x='KayitTarihi',
            y='BahisSayisi',
            title='Günlük Aktivite Trendi'
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    elif analysis_type == "Oyun Tercihleri":
        # Oyun tipi dağılımı
        game_prefs = data.groupby('OyunTipi').agg({
            'BahisSayisi': 'sum',
            'GGR': 'sum',
            'OyuncuID': 'nunique'
        }).reset_index()

        # Oyun tipi pasta grafiği
        fig_pie = px.pie(
            game_prefs,
            values='BahisSayisi',
            names='OyunTipi',
            title='Oyun Tipi Dağılımı (Bahis Sayısı)'
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # Oyun tipi detayları
        st.dataframe(
            game_prefs.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )

        # Oyun başına metrikler
        fig_metrics = go.Figure(data=[
            go.Bar(name='Bahis Sayısı', x=game_prefs['OyunTipi'], y=game_prefs['BahisSayisi']),
            go.Bar(name='Oyuncu Sayısı', x=game_prefs['OyunTipi'], y=game_prefs['OyuncuID'])
        ])

        fig_metrics.update_layout(barmode='group', title='Oyun Tipi Metrikleri')
        st.plotly_chart(fig_metrics, use_container_width=True)

    elif analysis_type == "Davranış Paterni":
        # Bahis tutarı dağılımı
        fig_dist = px.histogram(
            data,
            x='OrtBahis',
            title='Ortalama Bahis Tutarı Dağılımı',
            nbins=50
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        # Risk-Aktivite İlişkisi
        fig_scatter = px.scatter(
            data,
            x='RiskSkoru',
            y='GirisSikligi',
            color='OyunTipi',
            title='Risk-Aktivite İlişkisi',
            trendline="ols"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Davranış Segmentleri
        st.subheader("Davranış Segmentleri")

        data['AktiviteSegment'] = pd.qcut(
            data['GirisSikligi'],
            q=3,
            labels=['Düşük', 'Orta', 'Yüksek']
        )

        segment_analysis = data.groupby('AktiviteSegment').agg({
            'OrtBahis': 'mean',
            'GGR': 'mean',
            'RiskSkoru': 'mean',
            'OyuncuID': 'count'
        }).round(2)

        st.dataframe(
            segment_analysis.style.background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )

    else:  # Geçiş Analizi
        st.subheader("Oyun Tipi Geçiş Analizi")

        # Örnek geçiş matrisi (gerçek veriden hesaplanmalı)
        transition_matrix = pd.DataFrame(
            np.random.rand(4, 4),
            columns=['Sport', 'Casino', 'Poker', 'Virtual'],
            index=['Sport', 'Casino', 'Poker', 'Virtual']
        )

        fig_transition = px.imshow(
            transition_matrix,
            title='Oyun Tipleri Arası Geçiş Matrisi',
            labels=dict(x="Hedef Oyun", y="Kaynak Oyun", color="Geçiş Oranı")
        )

        st.plotly_chart(fig_transition, use_container_width=True)

    # Davranış Önerileri
    st.subheader("Davranış Bazlı Öneriler")

    # Aktivite bazlı öneriler
    high_activity = data['GirisSikligi'] > data['GirisSikligi'].quantile(0.75)
    high_risk = data['RiskSkoru'] > data['RiskSkoru'].quantile(0.75)

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        ✅ Yüksek Aktiviteli Oyuncular ({len(data[high_activity])} oyuncu):
        - 🎁 VIP programı değerlendirmesi
        - 🎯 Özel turnuva davetleri
        - 💎 Kişiselleştirilmiş bonuslar
        - ⭐ Sadakat programı
        """)

    with col2:
        st.warning(f"""
        ⚠️ Yüksek Riskli Oyuncular ({len(data[high_risk])} oyuncu):
        - 🛡️ Limit kontrolleri
        - 📊 Aktivite monitöring
        - ⏰ Mola hatırlatıcıları
        - 🎮 Düşük riskli alternatifler
        """)

    # İstatistiksel Özetler
    with st.expander("📊 Detaylı İstatistikler"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Aktivite İstatistikleri**")
            st.write(f"Ortalama Giriş: {data['GirisSikligi'].mean():.1f}")
            st.write(f"Medyan Giriş: {data['GirisSikligi'].median():.1f}")
            st.write(f"Std Sapma: {data['GirisSikligi'].std():.1f}")

        with col2:
            st.write("**Bahis İstatistikleri**")
            st.write(f"Ortalama Bahis: ₺{data['OrtBahis'].mean():.2f}")
            st.write(f"Medyan Bahis: ₺{data['OrtBahis'].median():.2f}")
            st.write(f"Std Sapma: ₺{data['OrtBahis'].std():.2f}")


def calculate_reliability_score(data: pd.DataFrame, model_performance: float = None, prediction_type: str = None) -> \
Tuple[float, Dict]:
    """Tahmin güvenilirlik skorunu hesaplar."""
    try:
        # Veri kalitesi skorları
        quality_scores = {
            'sample_size_score': min(len(data) / 1000, 1),  # Örnek boyutu skoru
            'data_completeness': max(0, 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))),
            # Veri tamlığı
            'feature_correlation': abs(data[['GGR', 'BahisSayisi', 'GirisSikligi']].corr().mean().mean()),
            # Feature korelasyonu
            'model_accuracy': max(0, min(1, model_performance if model_performance is not None else 0.5)),
            # Model doğruluğu
            'class_balance_score': 0.5  # Varsayılan değer
        }

        # Tahmin türüne göre özel metrikler
        if prediction_type == "Churn Tahmini" and 'Churn' in data.columns:
            class_counts = data['Churn'].value_counts()
            if len(class_counts) >= 2:
                quality_scores['class_balance_score'] = min(class_counts.min() / class_counts.max(), 1)

        # Ağırlıklar
        weights = {
            'sample_size_score': 0.2,
            'class_balance_score': 0.2,
            'data_completeness': 0.2,
            'feature_correlation': 0.2,
            'model_accuracy': 0.2
        }

        # Ağırlıklı ortalama hesaplama
        reliability_score = sum(score * weights[metric] for metric, score in quality_scores.items())

        # Skor sınırlandırma
        reliability_score = max(0, min(1, reliability_score))

        return reliability_score * 100, quality_scores

    except Exception as e:
        logger.error(f"Güvenilirlik skoru hesaplama hatası: {str(e)}")
        return 0, {
            'sample_size_score': 0,
            'class_balance_score': 0,
            'data_completeness': 0,
            'feature_correlation': 0,
            'model_accuracy': 0
        }


def show_reliability_analysis(reliability_score: float, quality_scores: Dict):
    """Güvenilirlik analizi gösterimini yapar."""
    st.subheader("Tahmin Güvenilirlik Analizi")

    # Güvenilirlik bileşenleri
    st.write("### Güvenilirlik Bileşenleri")
    col1, col2 = st.columns(2)

    with col1:
        st.write("📊 Veri Kalitesi")
        st.progress(quality_scores['data_completeness'])
        st.caption(f"{quality_scores['data_completeness'] * 100:.1f}%")

        st.write("🎯 Model Performansı")
        st.progress(quality_scores['model_accuracy'])
        st.caption(f"{quality_scores['model_accuracy'] * 100:.1f}%")

    with col2:
        st.write("⚖️ Sınıf Dengesi")
        st.progress(quality_scores['class_balance_score'])
        st.caption(f"{quality_scores['class_balance_score'] * 100:.1f}%")

        st.write("📈 Feature İlişkileri")
        st.progress(quality_scores['feature_correlation'])
        st.caption(f"{quality_scores['feature_correlation'] * 100:.1f}%")

    # Güvenilirlik göstergesi
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reliability_score,
        title={'text': "Tahmin Güvenilirliği"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': reliability_score
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)


def show_prediction_analysis(data: pd.DataFrame):
    st.subheader("Tahminleme ve AI Analizi 🤖")

    with st.expander("ℹ️ Tahminleme Analizi Hakkında", expanded=False):
        st.info("""
        ### Tahminleme Modelleri
        - 📈 **GGR Tahmini**: Gelecek dönem GGR projeksiyonu
        - 🔄 **Churn Tahmini**: Oyuncu kaybı riski analizi
        - 💰 **LTV Tahmini**: Yaşam boyu değer projeksiyonu
        - 🎯 **Davranış Tahmini**: Aktivite ve tercih tahminleri
        """)

    # Tahmin türü seçimi
    prediction_type = st.radio(
        "🎯 Tahmin Türü",
        ["GGR Tahmini", "Churn Tahmini", "LTV Tahmini", "Davranış Tahmini"],
        horizontal=True
    )

    # Minimum örnek sayısı kontrolü
    if len(data) < 10:
        st.warning("Tahminleme için yeterli veri bulunmuyor. En az 10 kayıt gerekli.")
        return

    try:
        if prediction_type == "GGR Tahmini":
            st.subheader("GGR Tahmin Analizi 📈")

            # Tahmin periyodu seçimi
            forecast_days = st.slider(
                "Tahmin Periyodu (Gün)",
                min_value=7,
                max_value=90,
                value=30
            )

            # Tarihsel GGR verisi hazırlama
            daily_ggr = data.groupby(pd.to_datetime(data['KayitTarihi']).dt.date)['GGR'].sum().reset_index()

            # En az 2 gün verisi kontrolü
            if len(daily_ggr) < 2:
                st.warning("GGR tahmini için yeterli günlük veri bulunmuyor.")
                return

            daily_ggr.set_index('KayitTarihi', inplace=True)

            # Feature hazırlama
            X = np.arange(len(daily_ggr)).reshape(-1, 1)
            y = daily_ggr['GGR'].values

            # Model eğitimi
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)

            # Model performansı hesaplama
            predictions_train = model.predict(X)
            model_performance = max(0, r2_score(y, predictions_train))
            reliability_score, quality_scores = calculate_reliability_score(data, model_performance, "GGR Tahmini")

            # Gelecek tahminleri
            future_dates = pd.date_range(daily_ggr.index[-1], periods=forecast_days + 1)[1:]
            future_X = np.arange(len(daily_ggr), len(daily_ggr) + forecast_days).reshape(-1, 1)
            predictions = model.predict(future_X)

            # Görselleştirme
            fig = go.Figure()

            # Gerçek veriler
            fig.add_trace(go.Scatter(
                x=daily_ggr.index,
                y=daily_ggr['GGR'],
                name='Gerçek GGR',
                line=dict(color='blue')
            ))

            # Tahminler
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name='Tahmin',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title='GGR Tahmin Analizi',
                xaxis_title='Tarih',
                yaxis_title='GGR (₺)',
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tahmin metrikleri
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Tahmini Toplam GGR",
                    f"₺{predictions.sum():,.2f}",
                    f"{(predictions.mean() / daily_ggr['GGR'].mean() - 1) * 100:.1f}% Değişim"
                )

            with col2:
                st.metric(
                    "Günlük Ortalama",
                    f"₺{predictions.mean():,.2f}",
                    "Tahmin"
                )

            with col3:
                st.metric(
                    "Trend",
                    "Yükseliş" if predictions[-1] > predictions[0] else "Düşüş",
                    f"{abs(predictions[-1] / predictions[0] - 1) * 100:.1f}%"
                )

            # Güvenilirlik analizi gösterimi
            show_reliability_analysis(reliability_score, quality_scores)

        elif prediction_type == "Churn Tahmini":

            st.subheader("Churn (Kayıp) Tahmin Analizi 🔄")

            # Churn tanımı

            churn_days = st.slider(

                "Churn Tanımı (İnaktif Gün)",

                min_value=7,

                max_value=90,

                value=28  # Varsayılan değer 1 ay

            )

            try:

                # Daha kapsamlı churn tanımı

                data['Churn'] = np.where(

                    (data['SonAktivite'] > churn_days) |  # İnaktiflik

                    (data['GGR'] < data['GGR'].quantile(0.1)) |  # Düşük GGR

                    (data['GirisSikligi'] < data['GirisSikligi'].quantile(0.1)),  # Düşük aktivite

                    1, 0

                )

                # Sınıf dağılımını kontrol et

                class_distribution = data['Churn'].value_counts()

                col1, col2 = st.columns(2)

                with col1:

                    st.write("**Sınıf Dağılımı:**")

                    st.write(f"✅ Aktif Oyuncular: {class_distribution.get(0, 0):,}")

                with col2:

                    st.write("**Churn Oranı:**")

                    st.write(
                        f"⚠️ Churn Oyuncular: {class_distribution.get(1, 0):,} ({class_distribution.get(1, 0) / len(data) * 100:.1f}%)")

                # Sınıf dengesi kontrolü

                if len(class_distribution) < 2 or min(class_distribution) < 2:
                    st.warning("""

                            Churn tahmini için yeterli veri dağılımı bulunmuyor.

                            Her iki sınıfta da (aktif ve churn) en az 2 örnek olmalı.


                            Churn tanımı şu kriterlere göre yapılmaktadır:

                            1. İnaktif gün sayısı belirlenen limitin üzerinde olanlar

                            2. GGR değeri en düşük %10'luk dilimde olanlar

                            3. Giriş sıklığı en düşük %10'luk dilimde olanlar


                            Lütfen churn tanımını veya veri filtreleme kriterlerini ayarlayın.

                            """)

                    return

                # Feature hazırlama

                features = ['GGR', 'BahisSayisi', 'BonusKullanimi', 'GirisSikligi', 'RiskSkoru']

                X = data[features].fillna(0)

                y = data['Churn']

                # Train-test split

                X_train, X_test, y_train, y_test = train_test_split(

                    X, y, test_size=0.2, random_state=42, stratify=y

                )

                # SMOTE uygula

                smote = SMOTE(random_state=42)

                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

                # Dengelenmiş veri dağılımı

                balanced_distribution = pd.Series(y_train_balanced).value_counts()

                st.write("\n**Dengelenmiş Eğitim Verisi:**")

                st.write(
                    f"✅ Aktif: {balanced_distribution.get(0, 0):,} | ⚠️ Churn: {balanced_distribution.get(1, 0):,}")

                # Model eğitimi

                model = GradientBoostingClassifier(

                    n_estimators=100,

                    learning_rate=0.1,

                    max_depth=3,

                    random_state=42

                )

                model.fit(X_train_balanced, y_train_balanced)

                # Tahminler

                y_pred = model.predict(X_test)

                y_prob = model.predict_proba(X_test)[:, 1]

                # Model metrikleri

                accuracy = accuracy_score(y_test, y_pred)

                conf_matrix = confusion_matrix(y_test, y_pred)

                class_report = classification_report(y_test, y_pred)

                # Güvenilirlik hesaplama

                reliability_score, quality_scores = calculate_reliability_score(data, accuracy, "Churn Tahmini")

                # Sonuçları göster

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(

                        "Model Doğruluğu",

                        f"{accuracy:.1%}",

                        "Tahmin Başarısı"

                    )

                with col2:

                    st.metric(

                        "Tahmini Churn Oranı",

                        f"{y_pred.mean():.1%}",

                        f"{(y_pred.mean() / y.mean() - 1) * 100:.1f}% Fark"

                    )

                # Confusion Matrix

                st.subheader("Confusion Matrix")

                conf_matrix_df = pd.DataFrame(

                    conf_matrix,

                    columns=['Tahmin: Aktif', 'Tahmin: Churn'],

                    index=['Gerçek: Aktif', 'Gerçek: Churn']

                )

                st.dataframe(conf_matrix_df.style.background_gradient(cmap='RdYlGn_r'))

                # Feature importance

                importance_df = pd.DataFrame({

                    'feature': features,

                    'importance': model.feature_importances_

                }).sort_values('importance', ascending=True)  # Değişiklik: ascending=True yapıldı

                fig_imp = px.bar(

                    importance_df,

                    x='importance',

                    y='feature',

                    orientation='h',

                    title='Churn Tahmin Faktörleri'

                )

                fig_imp.update_layout(template='plotly_dark')

                st.plotly_chart(fig_imp, use_container_width=True)

                # Risk Grupları

                st.subheader("Churn Risk Grupları")

                # Tüm veri için churn olasılıkları

                all_probs = model.predict_proba(X)[:, 1]

                data['ChurnProbability'] = all_probs

                # Risk grupları

                data['ChurnRiskGroup'] = pd.qcut(

                    data['ChurnProbability'],

                    q=4,

                    labels=['Düşük Risk', 'Orta-Düşük Risk', 'Orta-Yüksek Risk', 'Yüksek Risk']

                )

                risk_analysis = data.groupby('ChurnRiskGroup').agg({

                    'OyuncuID': 'count',

                    'GGR': 'mean',

                    'ChurnProbability': 'mean',

                    'BonusKullanimi': 'mean'

                }).round(2)

                st.dataframe(

                    risk_analysis.style.background_gradient(cmap='RdYlGn_r'),

                    use_container_width=True

                )

                # Model performansı hesaplama
                accuracy = accuracy_score(y_test, y_pred)
                reliability_score, quality_scores = calculate_reliability_score(data, accuracy, "Churn Tahmini")


                # Risk Önerileri

                st.subheader("Risk Grubu Önerileri")

                for risk_group in ['Yüksek Risk', 'Orta-Yüksek Risk', 'Orta-Düşük Risk', 'Düşük Risk']:

                    group_data = data[data['ChurnRiskGroup'] == risk_group]

                    with st.expander(f"{risk_group} Grubu Önerileri"):

                        if risk_group == 'Yüksek Risk':

                            st.error(f"""

                                    ⚠️ Acil Aksiyon Gerekli:

                                    - 📞 Proaktif iletişim başlat

                                    - 🎁 Özel win-back kampanyaları

                                    - 💰 Kişiselleştirilmiş bonus teklifleri

                                    - 📊 Günlük aktivite takibi


                                    Oyuncu Sayısı: {len(group_data):,}

                                    Ortalama Churn Olasılığı: {group_data['ChurnProbability'].mean():.1%}

                                    """)

                        elif risk_group == 'Orta-Yüksek Risk':

                            st.warning(f"""

                                    ⚠️ Yakın Takip:

                                    - 📱 Düzenli iletişim

                                    - 🎮 Yeni özellik ve oyun önerileri

                                    - 🎁 Hedefli promosyonlar

                                    - 📈 Haftalık aktivite takibi


                                    Oyuncu Sayısı: {len(group_data):,}

                                    Ortalama Churn Olasılığı: {group_data['ChurnProbability'].mean():.1%}

                                    """)

                        elif risk_group == 'Orta-Düşük Risk':

                            st.info(f"""

                                    ✅ İzleme:

                                    - 📊 Düzenli aktivite analizi

                                    - 🎮 Çapraz satış fırsatları

                                    - 🌟 Sadakat programı teklifleri

                                    - 📈 Aylık performans değerlendirmesi


                                    Oyuncu Sayısı: {len(group_data):,}

                                    Ortalama Churn Olasılığı: {group_data['ChurnProbability'].mean():.1%}

                                    """)

                        else:  # Düşük Risk

                            st.success(f"""

                                    ✅ Sürdürülebilirlik:

                                    - 🌟 VIP programı değerlendirmesi

                                    - 🎮 Yeni özellik önceliği

                                    - 🎁 Sadakat ödülleri

                                    - 📊 Rutin takip


                                    Oyuncu Sayısı: {len(group_data):,}

                                    Ortalama Churn Olasılığı: {group_data['ChurnProbability'].mean():.1%}

                                    """)

                # Güvenilirlik analizi gösterimi

                show_reliability_analysis(reliability_score, quality_scores)


            except Exception as e:

                st.error(f"Churn tahmini sırasında bir hata oluştu: {str(e)}")

                logger.error(f"Churn prediction error: {str(e)}")


        elif prediction_type == "LTV Tahmini":

            st.subheader("LTV (Life Time Value) Tahmin Analizi 💰")

            # Minimum veri kontrolü

            if len(data) < 50:
                st.warning("LTV tahmini için yeterli veri bulunmuyor. En az 50 kayıt gerekli.")

                return

            try:

                # Feature hazırlama

                data['AccountAge'] = (pd.to_datetime('now') - pd.to_datetime(data['KayitTarihi'])).dt.days

                # Temel ve türetilmiş özellikler

                features = [

                    'AccountAge', 'GirisSikligi', 'OrtBahis',

                    'BonusKullanimi', 'RiskSkoru', 'BahisSayisi'

                ]

                X = data[features].fillna(0)

                y = data['GGR'].fillna(0)

                # Train-test split

                X_train, X_test, y_train, y_test = train_test_split(

                    X, y, test_size=0.2, random_state=42

                )

                # Model eğitimi

                model = GradientBoostingRegressor(

                    n_estimators=100,

                    learning_rate=0.1,

                    max_depth=3,

                    random_state=42

                )

                model.fit(X_train, y_train)

                # Model performansı

                y_pred_test = model.predict(X_test)

                r2 = r2_score(y_test, y_pred_test)

                # Güvenilirlik hesaplama

                reliability_score, quality_scores = calculate_reliability_score(

                    data, r2, "LTV Tahmini"

                )

                # Tüm veri seti için LTV tahminleri

                predictions = model.predict(X)

                data['PredictedLTV'] = predictions

                # LTV segmentleri

                data['LTVSegment'] = pd.qcut(

                    data['PredictedLTV'],

                    q=4,

                    labels=['Bronze 🥉', 'Silver 🥈', 'Gold 🥇', 'Platinum 💎']

                )

                # Segment analizi

                segment_analysis = data.groupby('LTVSegment').agg({

                    'OyuncuID': 'count',

                    'PredictedLTV': ['mean', 'std'],

                    'GGR': 'mean',

                    'AccountAge': 'mean',

                    'BonusKullanimi': 'mean'

                }).round(2)

                # Performans Metrikleri

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(

                        "Model Performansı",

                        f"{r2:.1%}",

                        "R² Skoru"

                    )

                with col2:

                    st.metric(

                        "Ortalama LTV",

                        f"₺{predictions.mean():,.2f}",

                        f"{(predictions.mean() - y.mean()) / y.mean() * 100:+.1f}% Fark"

                    )

                with col3:

                    st.metric(

                        "Potansiyel Gelir",

                        f"₺{predictions.sum():,.2f}",

                        "Toplam LTV"

                    )

                # LTV Dağılımı

                st.subheader("LTV Segment Analizi")

                # Segment Dağılımı Grafiği

                fig_dist = px.pie(

                    data,

                    names='LTVSegment',

                    values='PredictedLTV',

                    title='LTV Segment Dağılımı',

                    color='LTVSegment',

                    color_discrete_map={

                        'Bronze 🥉': '#CD7F32',

                        'Silver 🥈': '#C0C0C0',

                        'Gold 🥇': '#FFD700',

                        'Platinum 💎': '#E5E4E2'

                    }

                )

                fig_dist.update_traces(textposition='inside', textinfo='percent+label')

                fig_dist.update_layout(template='plotly_dark')

                st.plotly_chart(fig_dist, use_container_width=True)

                # Segment Detayları

                st.markdown("### Segment Detayları")

                st.dataframe(

                    segment_analysis.style.background_gradient(cmap='YlOrRd'),

                    use_container_width=True

                )

                # Feature Importance

                importance_df = pd.DataFrame({

                    'feature': features,

                    'importance': model.feature_importances_

                }).sort_values('importance', ascending=True)

                fig_imp = px.bar(

                    importance_df,

                    x='importance',

                    y='feature',

                    orientation='h',

                    title='LTV Tahmin Faktörleri'

                )

                fig_imp.update_layout(template='plotly_dark')

                st.plotly_chart(fig_imp, use_container_width=True)

                # Model performansı hesaplama
                model_performance = max(0, r2_score(y_test, y_pred_test))
                reliability_score, quality_scores = calculate_reliability_score(data, model_performance, "LTV Tahmini")

                # Segment Önerileri

                st.subheader("Segment Bazlı Öneriler")

                for segment in ['Platinum 💎', 'Gold 🥇', 'Silver 🥈', 'Bronze 🥉']:

                    segment_data = data[data['LTVSegment'] == segment]

                    with st.expander(f"{segment} Segment Önerileri"):

                        if segment == 'Platinum 💎':

                            st.success(f"""

                                    💎 VIP Segment:

                                    - 🌟 Özel VIP yöneticisi atama

                                    - 🎁 Kişiselleştirilmiş bonuslar

                                    - 🎯 Yüksek limitli özel etkinlikler

                                    - 🏆 VIP turnuvalar ve etkinlikler


                                    Oyuncu Sayısı: {len(segment_data):,}

                                    Ortalama LTV: ₺{segment_data['PredictedLTV'].mean():,.2f}

                                    """)

                        elif segment == 'Gold 🥇':

                            st.info(f"""

                                    🥇 Yüksek Potansiyel:

                                    - 📈 VIP programına yükseltme fırsatı

                                    - 🎮 Özel oyun önerileri

                                    - 💰 Yatırım bonusları

                                    - 🎁 Sadakat ödülleri


                                    Oyuncu Sayısı: {len(segment_data):,}

                                    Ortalama LTV: ₺{segment_data['PredictedLTV'].mean():,.2f}

                                    """)

                        elif segment == 'Silver 🥈':

                            st.info(f"""

                                    🥈 Gelişim Potansiyeli:

                                    - 🎯 Aktivite bazlı bonuslar

                                    - 🎮 Yeni oyun önerileri

                                    - 📊 Haftalık aktivite raporu

                                    - 💫 Özel promosyonlar


                                    Oyuncu Sayısı: {len(segment_data):,}

                                    Ortalama LTV: ₺{segment_data['PredictedLTV'].mean():,.2f}

                                    """)

                        else:  # Bronze

                            st.warning(f"""

                                    🥉 Gelişim Alanı:

                                    - 🎁 Hoş geldin kampanyaları

                                    - 📱 Platform kullanım rehberi

                                    - 🎮 Düşük riskli oyun önerileri

                                    - 📊 Basit hedefler ve ödüller


                                    Oyuncu Sayısı: {len(segment_data):,}

                                    Ortalama LTV: ₺{segment_data['PredictedLTV'].mean():,.2f}

                                    """)

                # Güvenilirlik analizi gösterimi

                show_reliability_analysis(reliability_score, quality_scores)


            except Exception as e:

                st.error(f"LTV tahmini sırasında bir hata oluştu: {str(e)}")

                logger.error(f"LTV prediction error: {str(e)}")


        else:  # Davranış Tahmini

            st.subheader("Oyuncu Davranış Tahmini 🎯")

            try:

                if len(data) < 30:
                    st.warning("Davranış tahmini için yeterli veri bulunmuyor. En az 30 kayıt gerekli.")

                    return

                # Analiz türü seçimi

                behavior_type = st.selectbox(

                    "Davranış Analiz Türü",

                    ["Oyun Tercihi", "Aktivite Tahmini", "Yatırım Davranışı"]

                )

                if behavior_type == "Oyun Tercihi":

                    # Oyun tercihi analizi

                    game_pred = data.groupby('OyunTipi').agg({

                        'GGR': 'sum',

                        'BahisSayisi': 'count',

                        'OyuncuID': 'nunique'

                    }).reset_index()

                    # Metrikler

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        most_popular = game_pred.loc[game_pred['OyuncuID'].idxmax()]

                        st.metric(

                            "En Popüler Oyun",

                            most_popular['OyunTipi'],

                            f"{most_popular['OyuncuID']} Oyuncu"

                        )

                    with col2:

                        most_profitable = game_pred.loc[game_pred['GGR'].idxmax()]

                        st.metric(

                            "En Karlı Oyun",

                            most_profitable['OyunTipi'],

                            f"₺{most_profitable['GGR']:,.2f} GGR"

                        )

                    with col3:

                        most_active = game_pred.loc[game_pred['BahisSayisi'].idxmax()]

                        st.metric(

                            "En Aktif Oyun",

                            most_active['OyunTipi'],

                            f"{most_active['BahisSayisi']:,} Bahis"

                        )

                    # Oyun tercihi dağılımı

                    fig_games = go.Figure()

                    fig_games.add_trace(go.Bar(

                        name='GGR',

                        x=game_pred['OyunTipi'],

                        y=game_pred['GGR'],

                        marker_color='blue'

                    ))

                    fig_games.add_trace(go.Bar(

                        name='Bahis Sayısı',

                        x=game_pred['OyunTipi'],

                        y=game_pred['BahisSayisi'],

                        yaxis='y2',

                        marker_color='red'

                    ))

                    fig_games.update_layout(

                        title='Oyun Tercihi Analizi',

                        yaxis=dict(title='GGR (₺)', side='left'),

                        yaxis2=dict(title='Bahis Sayısı', side='right', overlaying='y'),

                        template='plotly_dark'

                    )

                    st.plotly_chart(fig_games, use_container_width=True)


                elif behavior_type == "Aktivite Tahmini":

                    # Günlük aktivite verisi

                    activity_data = data.groupby(

                        pd.to_datetime(data['KayitTarihi']).dt.date

                    ).agg({

                        'BahisSayisi': 'mean',

                        'OyuncuID': 'nunique',

                        'GGR': 'sum'

                    }).reset_index()

                    if len(activity_data) < 2:
                        st.warning("Aktivite tahmini için yeterli günlük veri bulunmuyor.")

                        return

                    # Aktivite tahmini modeli

                    X = np.arange(len(activity_data)).reshape(-1, 1)

                    y = activity_data['BahisSayisi'].values

                    model = GradientBoostingRegressor(

                        n_estimators=100,

                        learning_rate=0.1,

                        max_depth=3,

                        random_state=42

                    )

                    model.fit(X, y)

                    # Gelecek tahminleri

                    future_days = 7

                    future_X = np.arange(len(activity_data), len(activity_data) + future_days).reshape(-1, 1)

                    future_activity = model.predict(future_X)

                    # Model performansı

                    r2 = r2_score(y, model.predict(X))

                    reliability_score, quality_scores = calculate_reliability_score(

                        data, r2, "Aktivite Tahmini"

                    )

                    # Metrikler

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(

                            "Günlük Ortalama Aktivite",

                            f"{activity_data['BahisSayisi'].mean():.1f}",

                            "Bahis/Gün"

                        )

                    with col2:

                        st.metric(

                            "Tahmini Aktivite",

                            f"{future_activity.mean():.1f}",

                            f"{(future_activity.mean() / activity_data['BahisSayisi'].mean() - 1) * 100:+.1f}% Değişim"

                        )

                    with col3:

                        st.metric(

                            "Aktif Oyuncu",

                            f"{activity_data['OyuncuID'].mean():.0f}",

                            "Günlük Ortalama"

                        )

                    # Aktivite trendi ve tahmin grafiği

                    fig_activity = go.Figure()

                    fig_activity.add_trace(go.Scatter(

                        x=activity_data['KayitTarihi'],

                        y=activity_data['BahisSayisi'],

                        name='Gerçek Aktivite',

                        mode='lines+markers'

                    ))

                    future_dates = pd.date_range(

                        activity_data['KayitTarihi'].iloc[-1],

                        periods=future_days + 1

                    )[1:]

                    fig_activity.add_trace(go.Scatter(

                        x=future_dates,

                        y=future_activity,

                        name='Tahmin',

                        mode='lines',

                        line=dict(dash='dash')

                    ))

                    fig_activity.update_layout(

                        title='Aktivite Trendi ve Tahmin',

                        xaxis_title='Tarih',

                        yaxis_title='Ortalama Bahis Sayısı',

                        template='plotly_dark'

                    )

                    st.plotly_chart(fig_activity, use_container_width=True)


                else:  # Yatırım Davranışı

                    # Yatırım analizi

                    deposit_data = data.groupby(

                        pd.to_datetime(data['KayitTarihi']).dt.date

                    ).agg({

                        'ToplamDepozit': 'sum',

                        'ToplamCekim': 'sum',

                        'OyuncuID': 'nunique'

                    }).reset_index()

                    # Metrikler

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(

                            "Ortalama Yatırım",

                            f"₺{data['ToplamDepozit'].mean():,.2f}",

                            "Kişi Başı"

                        )

                    with col2:

                        retention = data['ToplamCekim'].sum() / data['ToplamDepozit'].sum()

                        st.metric(

                            "Para Tutma Oranı",

                            f"{(1 - retention) * 100:.1f}%",

                            "Depozit/Çekim"

                        )

                    with col3:

                        st.metric(

                            "Aktif Yatırımcı",

                            f"{(data['ToplamDepozit'] > 0).sum()}",

                            f"{(data['ToplamDepozit'] > 0).mean() * 100:.1f}%"

                        )

                    if behavior_type == "Aktivite Tahmini":
                        # Model performansı hesaplama
                        predictions_train = model.predict(X)
                        model_performance = max(0, r2_score(y, predictions_train))
                        reliability_score, quality_scores = calculate_reliability_score(data, model_performance,
                                                                                        "Davranış Tahmini")

                    # Yatırım trendi grafiği

                    fig_deposit = go.Figure()

                    fig_deposit.add_trace(go.Bar(

                        name='Yatırım',

                        x=deposit_data['KayitTarihi'],

                        y=deposit_data['ToplamDepozit'],

                        marker_color='green'

                    ))

                    fig_deposit.add_trace(go.Bar(

                        name='Çekim',

                        x=deposit_data['KayitTarihi'],

                        y=deposit_data['ToplamCekim'],

                        marker_color='red'

                    ))

                    fig_deposit.update_layout(

                        title='Yatırım ve Çekim Trendi',

                        barmode='group',

                        xaxis_title='Tarih',

                        yaxis_title='Tutar (₺)',

                        template='plotly_dark'

                    )

                    st.plotly_chart(fig_deposit, use_container_width=True)

                # Güvenilirlik analizi gösterimi (sadece aktivite tahmini için)

                if behavior_type == "Aktivite Tahmini":
                    show_reliability_analysis(reliability_score, quality_scores)


            except Exception as e:

                st.error(f"Davranış tahmini sırasında bir hata oluştu: {str(e)}")

                logger.error(f"Behavior prediction error: {str(e)}")

        # Güvenilirlik skorunu hesapla
        reliability_score, quality_scores = calculate_reliability_score(data, accuracy)

        # Tahmin Güvenilirlik Analizi
        st.subheader("Tahmin Güvenilirlik Analizi")

        # Güvenilirlik bileşenleri
        st.write("### Güvenilirlik Bileşenleri")
        col1, col2 = st.columns(2)

        with col1:
            st.write("📊 Veri Kalitesi")
            st.progress(quality_scores['data_completeness'])
            st.caption(f"{quality_scores['data_completeness'] * 100:.1f}%")

            st.write("🎯 Model Performansı")
            st.progress(quality_scores['model_accuracy'])
            st.caption(f"{quality_scores['model_accuracy'] * 100:.1f}%")

        with col2:
            st.write("⚖️ Sınıf Dengesi")
            st.progress(quality_scores['class_balance_score'])
            st.caption(f"{quality_scores['class_balance_score'] * 100:.1f}%")

            st.write("📈 Feature İlişkileri")
            st.progress(quality_scores['feature_correlation'])
            st.caption(f"{quality_scores['feature_correlation'] * 100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reliability_score,
            title={'text': "Tahmin Güvenilirliği"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': reliability_score
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Tahminleme analizi sırasında bir hata oluştu: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")


def main():
    try:
        # Sidebar konfigürasyonu
        analysis_type, selected_games, selected_cities, min_ggr = configure_sidebar()

        # Veri filtreleme
        filtered_data = prepare_filtered_data(data, selected_games, selected_cities, min_ggr)

        # Seçilen analiz modülünü göster
        if analysis_type == "Genel Bakış":
            show_overview(filtered_data)
        elif analysis_type == "Oyuncu Segmentasyonu":
            show_player_segmentation(filtered_data)
        elif analysis_type == "GGR Analizi":
            show_ggr_analysis(filtered_data)
        elif analysis_type == "Risk Analizi":
            show_risk_analysis(filtered_data)
        elif analysis_type == "Bonus Performansı":
            show_bonus_performance(filtered_data)
        elif analysis_type == "Oyuncu Davranışı":
            show_player_behavior(filtered_data)
        elif analysis_type == "Model Bazlı Tahminler":
            show_prediction_analysis(filtered_data)
        elif analysis_type == "Trend Analizi":
            show_trend_analysis(filtered_data)
        elif analysis_type == "Cohort Analizi":
            show_cohort_analysis(filtered_data)
        elif analysis_type == "A/B Test Analizi":
            show_ab_test_analysis(filtered_data)
        elif analysis_type == "ANOVA Analizi":
            show_anova_analysis(filtered_data)
        elif analysis_type == "ROI Analizi":
            show_roi_analysis(filtered_data)
        else:
            st.info("Bu analiz modülü geliştirme aşamasındadır.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        logger.error(f"Main function error: {str(e)}")

if __name__ == "__main__":
    main()


