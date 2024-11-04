import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from io import BytesIO
import warnings
from typing import Tuple, Dict, List, Union, Any
import time

warnings.filterwarnings('ignore')

# Sayfa Yapılandırması
st.set_page_config(
    page_title="iGaming CRM Analytics",
    page_icon="🎲",
    layout="wide"
)

# Loading ve Demo Bildirimi
with st.spinner("Dashboard yükleniyor... 🚀"):
    time.sleep(1)

st.success("Dashboard hazır! 🎉")

# Ana başlık altına eklenecek:
import streamlit.components.v1 as components

# Özel CSS ve HTML ile popup
components.html(
    """
    <style>
    .welcome-popup {
        animation: slideIn 0.5s ease-out;
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    @keyframes slideIn {
        0% { transform: translateY(-100px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    .popup-title { color: #4CAF50; font-size: 24px; margin-bottom: 10px; }
    .popup-subtitle { color: #9e9e9e; font-size: 16px; }
    .popup-features {
        display: flex;
        justify-content: space-around;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    .feature-item {
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        min-width: 150px;
        text-align: center;
    }
    </style>
    <div class="welcome-popup">
        <div class="popup-title">🎮 iGaming Analytics Suite</div>
        <div class="popup-subtitle">Gelişmiş CRM Analiz ve Tahminleme Sistemi</div>
        <div class="popup-features">
            <div class="feature-item">🎯 Gerçek Zamanlı Analiz</div>
            <div class="feature-item">🤖 AI Tahminleme</div>
            <div class="feature-item">📊 Detaylı Raporlama</div>
            <div class="feature-item">⚡ Performans İzleme</div>
        </div>
        <p style="color: #4CAF50; margin-top: 15px;">Geliştirici: Yasin Tanış | v2.0</p>
        <p style="font-size: 12px; color: #757575;">Bu sistem demo amaçlı geliştirilmiş olup, gerçek verileri simüle etmektedir.</p>
    </div>
    """,
    height=300,
)
# Ana stil
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        margin-top: -2em;
    }
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 500;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    </style>
    <h1 class='main-header'>🎲 iGaming için Crm Analizi ve Raporlama 🎲</h1>
    """, unsafe_allow_html=True)

# Cache süresi ayarı
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
    """Veriyi filtreleme seçeneklerine göre hazırlar."""
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


# Sidebar konfigürasyonu
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
                "Tahminleme",
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


# Analiz modülleri için yardımcı fonksiyonlar
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


def show_risk_analysis(data: pd.DataFrame):
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


def show_ggr_analysis(data: pd.DataFrame):
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

        # Şık metrik kartları için CSS
        st.markdown("""
        <style>
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
        </style>
        """, unsafe_allow_html=True)

        with col1:
            st.metric(
                label="Toplam GGR 💰",
                value=f"₺{data['GGR'].sum():,.2f}",
                delta=f"{((data['GGR'].sum() / data['ToplamDepozit'].sum()) * 100):.1f}% Oran"
            )

        with col2:
            st.metric(
                label="Ortalama GGR/Oyuncu 👤",
                value=f"₺{data['GGR'].mean():,.2f}",
                delta=f"{((data['GGR'].mean() / data['GGR'].mean().mean()) - 1) * 100:.1f}%"
            )

        with col3:
            st.metric(
                label="En Yüksek GGR 🏆",
                value=f"₺{data['GGR'].max():,.2f}",
                delta="Top %1"
            )

        with col4:
            st.metric(
                label="Karlılık Oranı 📊",
                value=f"{(data['GGR'].sum() / data['ToplamDepozit'].sum() * 100):.1f}%",
                delta="Hedefin üzerinde" if data['GGR'].sum() / data['ToplamDepozit'].sum() > 0.2 else "Hedefin altında"
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
                    daily_ggr.to_excel(writer, sheet_name='Daily GGR')
                    segment_analysis.to_excel(writer, sheet_name='Segment Analysis')
                    game_analysis.to_excel(writer, sheet_name='Game Analysis')

                st.sidebar.download_button(
                    label="📥 Excel İndir",
                    data=buffer,
                    file_name="ggr_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )

def show_prediction_analysis(data: pd.DataFrame):
    """Tahminleme modülünü gösterir."""
    try:
        st.subheader("Tahminleme")

        pred_type = st.selectbox("Tahmin Türü", ["GGR", "Churn", "Risk"])

        if pred_type == "GGR":
            days = st.slider("Tahmin Günü", 7, 90, 30)

            # Basit tahmin modeli
            X = np.arange(len(data)).reshape(-1, 1)
            y = data['GGR']
            model = LinearRegression().fit(X, y)

            future_X = np.arange(len(data), len(data) + days).reshape(-1, 1)
            predictions = model.predict(future_X)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data['GGR'], name='Gerçek'))
            fig.add_trace(go.Scatter(x=np.arange(len(data), len(data) + days),
                                     y=predictions, name='Tahmin'))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Tahminleme hatası: {str(e)}")


def generate_pdf_report(data: pd.DataFrame, analysis_type: str):
    pdf_buffer = BytesIO()
    plt.figure(figsize=(10, 12))
    # ... Rapor içeriği
    return pdf_buffer


def download_report_button(data: pd.DataFrame, analysis_type: str):
    col1, col2, col3 = st.columns(3)
    with col2:
        report_format = st.selectbox(
            "Rapor Formatı",
            ["PDF", "Excel", "CSV", "PowerPoint"]
        )

        if st.button("📥 Rapor İndir", key=f"download_{analysis_type}"):
            with st.spinner("Rapor hazırlanıyor..."):
                if report_format == "PDF":
                    pdf_buffer = generate_pdf_report(data, analysis_type)
                    st.download_button(
                        "📑 PDF İndir",
                        data=pdf_buffer,
                        file_name=f"{analysis_type}_rapor.pdf",
                        mime="application/pdf"
                    )
                # ... Diğer formatlar

# Ana uygulama
def main():
    # Sidebar konfigürasyonu
    analysis_type, selected_games, selected_cities, min_ggr = configure_sidebar()

    # Veri filtreleme
    filtered_data = prepare_filtered_data(data, selected_games, selected_cities, min_ggr)

    # Seçilen analiz modülünü göster
    if analysis_type == "Genel Bakış":
        show_overview(filtered_data)
    elif analysis_type == "Risk Analizi":
        show_risk_analysis(filtered_data)
    elif analysis_type == "Tahminleme":
        show_prediction_analysis(filtered_data)
    # ... Diğer modüller




if __name__ == "__main__":
    main()

