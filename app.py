# Temel kütüphaneler
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Görselleştirme kütüphaneleri
import plotly.express as px
import plotly.graph_objects as go

# Sklearn kütüphaneleri
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

# Diğer ML kütüphaneleri
import xgboost as xgb
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules

# Export fonksiyonları için gerekli importlar
from fpdf import FPDF
import base64
from io import BytesIO
import tempfile
import os
import plotly.io as pio

# Uyarıları kapatma
import warnings
warnings.filterwarnings('ignore')


# Sayfa Yapılandırması
st.set_page_config(
    page_title="CRM Analitik Dashboard",
    page_icon="📊",
    layout="wide"
)

# Ana Başlık ve Stil
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
    </style>
    <h1 class='main-header'>CRM Analitik Dashboard</h1>
    """, unsafe_allow_html=True)

# Örnek Veri Oluşturma Fonksiyonu
@st.cache_data
def generate_sample_data(n_customers=1000):
    np.random.seed(42)
    
    customer_ids = range(1, n_customers + 1)
    registration_dates = [datetime.now() - timedelta(days=np.random.randint(1, 365)) 
                         for _ in range(n_customers)]
    
    purchase_amounts = np.random.normal(1000, 300, n_customers)
    purchase_frequency = np.random.poisson(5, n_customers)
    last_purchase_days = np.random.randint(1, 100, n_customers)
    
    age_groups = np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_customers)
    genders = np.random.choice(['Erkek', 'Kadın'], n_customers)
    locations = np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Bursa'], n_customers)
    
    website_visits = np.random.poisson(10, n_customers)
    email_opens = np.random.poisson(8, n_customers)
    cart_abandonment = np.random.random(n_customers)
    
    nps_scores = np.random.randint(0, 11, n_customers)
    satisfaction = np.random.choice(['Çok Memnun', 'Memnun', 'Nötr', 'Memnun Değil'], n_customers)
    
    return pd.DataFrame({
        'MusteriID': customer_ids,
        'KayitTarihi': registration_dates,
        'ToplamHarcama': purchase_amounts,
        'AlisverisFrekansi': purchase_frequency,
        'SonAlisverisGunu': last_purchase_days,
        'YasGrubu': age_groups,
        'Cinsiyet': genders,
        'Sehir': locations,
        'WebsiteZiyareti': website_visits,
        'EmailAcma': email_opens,
        'SepetTerkOrani': cart_abandonment,
        'NPSPuani': nps_scores,
        'MemnuniyetDurumu': satisfaction
    })

# Yardımcı Fonksiyonlar
def calculate_clv(customer_data):
    """Müşteri Yaşam Boyu Değeri hesaplama"""
    avg_purchase = customer_data['ToplamHarcama'].mean()
    purchase_frequency = customer_data['AlisverisFrekansi'].mean()
    customer_lifespan = (customer_data['KayitTarihi'].max() - 
                        customer_data['KayitTarihi'].min()).days / 365
    
    return avg_purchase * purchase_frequency * customer_lifespan

def predict_next_purchase(customer_data):
    """Gelecek alışveriş tahmini"""
    features = ['ToplamHarcama', 'AlisverisFrekansi', 'SonAlisverisGunu']
    X = customer_data[features]
    y = customer_data['SonAlisverisGunu']
    
    model = GradientBoostingRegressor()
    model.fit(X, y)
    
    return model

def perform_basket_analysis(transaction_data):
    """Basitleştirilmiş sepet analizi"""
    # Ürün kombinasyonlarını analiz etme
    product_combinations = pd.DataFrame({
        'combination': ['A+B', 'B+C', 'A+C', 'A+B+C'],
        'frequency': [45, 32, 28, 15],
        'lift': [1.5, 1.3, 1.2, 1.8]
    })
    return product_combinations
    
# Ana veri yükleme
data = generate_sample_data()

# Sidebar menüsü
st.sidebar.markdown("### Ana Menü")
analysis_type = st.sidebar.selectbox(
    "Analiz Türü Seçin",
    ["Genel Bakış", "Müşteri Segmentasyonu", "Satış Analizi", 
     "Müşteri Davranışı", "Tahminleme", "RFM Analizi",
     "A/B Test Analizi", "Cohort Analizi", "Churn Prediction",
     "Özel Rapor Oluşturma"]
)

# Tema seçimi
theme = st.sidebar.selectbox(
    "Tema Seçin",
    ["Light", "Dark", "Custom"]
)

if theme == "Custom":
    primary_color = st.sidebar.color_picker("Ana Renk", "#1f77b4")
    st.markdown(f"""
        <style>
        :root {{
            --primary-color: {primary_color};
        }}
        </style>
    """, unsafe_allow_html=True)

# Veri güncelleme aralığı
refresh_interval = st.sidebar.selectbox(
    "Veri Güncelleme Sıklığı",
    ["30 saniye", "1 dakika", "5 dakika", "15 dakika", "30 dakika"]
)

# Filtreleme seçenekleri
st.sidebar.markdown("### Filtreler")
selected_cities = st.sidebar.multiselect(
    "Şehir Seçin",
    options=data['Sehir'].unique(),
    default=data['Sehir'].unique()
)

selected_age_groups = st.sidebar.multiselect(
    "Yaş Grubu Seçin",
    options=data['YasGrubu'].unique(),
    default=data['YasGrubu'].unique()
)

min_purchase = st.sidebar.number_input(
    "Minimum Harcama",
    min_value=float(data['ToplamHarcama'].min()),
    max_value=float(data['ToplamHarcama'].max()),
    value=float(data['ToplamHarcama'].min())
)

# Ana içerik alanı
if analysis_type == "Genel Bakış":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Toplam Müşteri",
            f"{len(data):,}",
            "↑ 12% Geçen Aya Göre"
        )
    
    with col2:
        st.metric(
            "Ortalama Harcama",
            f"₺{data['ToplamHarcama'].mean():,.2f}",
            "↑ 8% Geçen Aya Göre"
        )
    
    with col3:
        st.metric(
            "Ortalama NPS",
            f"{data['NPSPuani'].mean():.1f}",
            "↑ 0.5 Geçen Aya Göre"
        )
    
    # Şehirlere Göre Dağılım
    st.subheader("Şehirlere Göre Müşteri Dağılımı")
    fig_city = px.pie(data, names='Sehir', title='Şehir Bazlı Müşteri Dağılımı')
    st.plotly_chart(fig_city, use_container_width=True)
    
    # Yaş Grubu ve Cinsiyet Dağılımı
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yaş Grubu Dağılımı")
        fig_age = px.bar(data['YasGrubu'].value_counts(), title='Yaş Grubu Dağılımı')
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.subheader("Cinsiyet Dağılımı")
        fig_gender = px.pie(data, names='Cinsiyet', title='Cinsiyet Dağılımı')
        st.plotly_chart(fig_gender, use_container_width=True)

# [Önceki kodlar aynı kalacak, sadece eksik analiz türlerini ekliyoruz]

elif analysis_type == "Satış Analizi":
    st.subheader("Satış Performans Analizi")
    
    # Satış trendleri
    sales_by_date = pd.DataFrame({
        'Tarih': pd.date_range(start='2023-01-01', end=datetime.now(), freq='D'),
        'Satış': np.random.normal(10000, 2000, 
                                (datetime.now() - pd.to_datetime('2023-01-01')).days + 1)
    })
    
    # Metrikler
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Toplam Satış",
            f"₺{sales_by_date['Satış'].sum():,.0f}",
            "↑ 15% Geçen Aya Göre"
        )
    with col2:
        st.metric(
            "Ortalama Günlük Satış",
            f"₺{sales_by_date['Satış'].mean():,.0f}",
            "↑ 8% Geçen Aya Göre"
        )
    with col3:
        st.metric(
            "En Yüksek Günlük Satış",
            f"₺{sales_by_date['Satış'].max():,.0f}",
            "Tüm Zamanların Rekoru"
        )
    
    # Satış trendi grafiği
    st.write("### Satış Trendi")
    fig_sales = px.line(sales_by_date, x='Tarih', y='Satış',
                       title='Günlük Satış Trendi')
    fig_sales.update_traces(line_color='#1f77b4')
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Dönemsel analiz
    col1, col2 = st.columns(2)
    
    with col1:
        # Aylık satışlar
        monthly_sales = sales_by_date.set_index('Tarih').resample('M')['Satış'].sum()
        fig_monthly = px.bar(monthly_sales, 
                           title='Aylık Toplam Satışlar',
                           labels={'value': 'Satış', 'index': 'Ay'})
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Haftalık satışlar
        weekly_sales = sales_by_date.set_index('Tarih').resample('W')['Satış'].sum()
        fig_weekly = px.bar(weekly_sales, 
                          title='Haftalık Toplam Satışlar',
                          labels={'value': 'Satış', 'index': 'Hafta'})
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Satış dağılımı analizi
    st.write("### Satış Dağılımı Analizi")
    fig_dist = px.histogram(sales_by_date, x='Satış', 
                          title='Satış Dağılımı',
                          marginal='box')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Tahmin modeli
    st.write("### Satış Tahmini")
    forecast_days = st.slider("Tahmin Günü Seçin", 7, 90, 30)
    
    # Basit bir tahmin modeli
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(sales_by_date)).reshape(-1, 1)
    y = sales_by_date['Satış']
    model = LinearRegression()
    model.fit(X, y)
    
    # Tahmin
    future_dates = pd.date_range(
        start=sales_by_date['Tarih'].max() + pd.Timedelta(days=1),
        periods=forecast_days
    )
    future_X = np.arange(len(sales_by_date), 
                        len(sales_by_date) + forecast_days).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Tahmin grafiği
    forecast_df = pd.DataFrame({
        'Tarih': pd.concat([sales_by_date['Tarih'], 
                           pd.Series(future_dates)]),
        'Satış': np.concatenate([sales_by_date['Satış'], future_y])
    })
    forecast_df['Tip'] = ['Gerçek'] * len(sales_by_date) + \
                        ['Tahmin'] * forecast_days
    
    fig_forecast = px.line(forecast_df, x='Tarih', y='Satış', 
                          color='Tip', title='Satış Tahmini')
    st.plotly_chart(fig_forecast, use_container_width=True)

elif analysis_type == "Müşteri Davranışı":
    st.subheader("Müşteri Davranış Analizi")
    
    # Aktivite metrikleri
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Ortalama Ziyaret Süresi",
            "4.5 dakika",
            "↑ 12% Geçen Aya Göre"
        )
    with col2:
        st.metric(
            "Sepet Terk Oranı",
            "24%",
            "↓ 3% Geçen Aya Göre"
        )
    with col3:
        st.metric(
            "Dönüşüm Oranı",
            "3.2%",
            "↑ 0.5% Geçen Aya Göre"
        )
    
    # Davranış segmentasyonu
    behavior_data = pd.DataFrame({
        'Müşteri': range(1000),
        'Ziyaret': np.random.poisson(10, 1000),
        'Süre': np.random.normal(5, 2, 1000),
        'Etkileşim': np.random.normal(3, 1, 1000)
    })
    
    # K-means kümeleme
    X = behavior_data[['Ziyaret', 'Süre', 'Etkileşim']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    behavior_data['Segment'] = kmeans.fit_predict(X_scaled)
    
    # Segment görselleştirmesi
    st.write("### Davranış Segmentleri")
    fig_segments = px.scatter_3d(behavior_data, 
                                x='Ziyaret', y='Süre', z='Etkileşim',
                                color='Segment',
                                title='Müşteri Davranış Segmentleri')
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Davranış trendleri
    st.write("### Davranış Trendleri")
    behavior_trends = pd.DataFrame({
        'Tarih': pd.date_range(start='2023-01-01', end=datetime.now(), freq='D'),
        'Ziyaret': np.random.normal(1000, 100, 
                                  (datetime.now() - pd.to_datetime('2023-01-01')).days + 1),
        'Etkileşim': np.random.normal(500, 50, 
                                    (datetime.now() - pd.to_datetime('2023-01-01')).days + 1)
    })
    
    metric_choice = st.selectbox("Metrik Seçin", ['Ziyaret', 'Etkileşim'])
    fig_trend = px.line(behavior_trends, x='Tarih', y=metric_choice,
                       title=f'{metric_choice} Trendi')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Heatmap
    st.write("### Günlük Aktivite Heatmap")
    hourly_data = pd.DataFrame(
        np.random.normal(100, 20, (24, 7)),
        index=range(24),
        columns=['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 
                 'Cuma', 'Cumartesi', 'Pazar']
    )
    fig_heat = px.imshow(hourly_data,
                        labels=dict(x="Gün", y="Saat", color="Aktivite"),
                        title="Günlük Aktivite Yoğunluğu")
    st.plotly_chart(fig_heat, use_container_width=True)

elif analysis_type == "Tahminleme":
    st.subheader("Gelişmiş Tahminleme Modülleri")
    
    prediction_type = st.selectbox(
        "Tahminleme Türü Seçin",
        ["Gelecek Dönem Satış Tahmini", "Müşteri Yaşam Boyu Değeri (CLV)",
         "Churn Riski", "Sonraki Alışveriş Tahmini"]
    )
    
    if prediction_type == "Gelecek Dönem Satış Tahmini":
        st.write("### Satış Tahmini")
        forecast_period = st.slider("Kaç günlük tahmin?", 7, 90, 30)
        
        # Örnek satış verisi
        historical_sales = pd.DataFrame({
            'Tarih': pd.date_range(start='2023-01-01', end=datetime.now(), freq='D'),
            'Satış': np.random.normal(10000, 2000, 
                                    (datetime.now() - pd.to_datetime('2023-01-01')).days + 1)
        })
        
        # Basit tahmin modeli
        model = LinearRegression()
        X = np.arange(len(historical_sales)).reshape(-1, 1)
        y = historical_sales['Satış']
        model.fit(X, y)
        
        # Tahminler
        future_X = np.arange(len(historical_sales), 
                           len(historical_sales) + forecast_period).reshape(-1, 1)
        future_y = model.predict(future_X)
        
        # Görselleştirme
        forecast_dates = pd.date_range(
            start=historical_sales['Tarih'].max() + pd.Timedelta(days=1),
            periods=forecast_period
        )
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=historical_sales['Tarih'],
            y=historical_sales['Satış'],
            name='Gerçek Satışlar',
            line=dict(color='blue')
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=future_y,
            name='Tahmin',
            line=dict(color='red', dash='dash')
        ))
        fig_forecast.update_layout(title='Satış Tahmini',
                                 xaxis_title='Tarih',
                                 yaxis_title='Satış')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    elif prediction_type == "Müşteri Yaşam Boyu Değeri (CLV)":
        st.write("### CLV Tahmini")
        
        # Örnek müşteri verisi
        clv_data = pd.DataFrame({
            'MusteriID': range(1000),
            'ToplamHarcama': np.random.normal(1000, 200, 1000),
            'AlisverisFrekansi': np.random.poisson(5, 1000),
            'MusteriYasi': np.random.randint(1, 1000, 1000)  # Gün cinsinden
        })
        
        # CLV hesaplama
        clv_data['CLV'] = (clv_data['ToplamHarcama'] * 
                          clv_data['AlisverisFrekansi'] * 
                          (1000 / clv_data['MusteriYasi']))
        
        # Görselleştirme
        fig_clv = px.histogram(clv_data, x='CLV',
                             title='CLV Dağılımı',
                             labels={'CLV': 'Müşteri Yaşam Boyu Değeri'})
        st.plotly_chart(fig_clv, use_container_width=True)
        
        # CLV segmentasyonu
        clv_data['CLV_Segment'] = pd.qcut(clv_data['CLV'], q=4, 
                                         labels=['Düşük', 'Orta', 'Yüksek', 'Premium'])
        
        fig_segments = px.pie(clv_data, names='CLV_Segment',
                            title='CLV Segmentleri')
        st.plotly_chart(fig_segments, use_container_width=True)
        
    elif prediction_type == "Churn Riski":
        st.write("### Churn Risk Tahmini")
        
        # Örnek özellikler için form
        with st.form("churn_prediction"):
            col1, col2 = st.columns(2)
            with col1:
                total_spend = st.number_input("Toplam Harcama", value=1000.0)
                frequency = st.number_input("Alışveriş Frekansı", value=5)
            with col2:
                last_purchase = st.number_input("Son Alışverişten Geçen Gün", value=30)
                satisfaction = st.slider("Memnuniyet Puanı", 0, 10, 7)
            
            predict_button = st.form_submit_button("Churn Riski Hesapla")
            
            if predict_button:
                # Basit bir risk skoru hesaplama
                risk_score = (
                    (last_purchase * 0.4) +
                    ((10 - satisfaction) * 0.3) +
                    ((10 - frequency) * 0.2) +
                    ((1000 - min(total_spend, 1000)) * 0.1)
                ) / 100
                
                risk_percentage = min(risk_score * 100, 100)
                
                # Risk göstergesi 
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percentage,
                    title={'text': "Churn Riski"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk değerlendirmesi
                if risk_percentage < 30:
                    st.success("Düşük Churn Riski: Müşteri sadık görünüyor.")
                elif risk_percentage < 70:
                    st.warning("Orta Churn Riski: İzlenmesi gerekiyor.")
                else:
                    st.error("Yüksek Churn Riski: Acil aksiyon gerekiyor!")
                
                # Öneriler
                st.write("### Öneriler")
                if risk_percentage >= 70:
                    st.write("""
                    1. Özel indirim kampanyası gönderin
                    2. Müşteri temsilcisi araması planlayın
                    3. Kişiselleştirilmiş teklifler sunun
                    """)
                elif risk_percentage >= 30:
                    st.write("""
                    1. Memnuniyet anketi gönderin
                    2. Yeni ürün önerilerinde bulunun
                    3. Email kampanyalarına dahil edin
                    """)
                else:
                    st.write("""
                    1. Sadakat programı teklifleri sunun
                    2. Düzenli iletişimi sürdürün
                    3. Referans programına dahil edin
                    """)
    
    elif prediction_type == "Sonraki Alışveriş Tahmini":
        st.write("### Sonraki Alışveriş Tahmini")
        
        # Alışveriş geçmişi simülasyonu
        with st.form("next_purchase_prediction"):
            col1, col2 = st.columns(2)
            with col1:
                avg_purchase_freq = st.number_input("Ortalama Alışveriş Sıklığı (gün)", value=15)
                last_purchase = st.number_input("Son Alışverişten Geçen Gün", value=10)
            with col2:
                total_purchases = st.number_input("Toplam Alışveriş Sayısı", value=20)
                avg_basket = st.number_input("Ortalama Sepet Tutarı", value=250.0)
            
            predict_next = st.form_submit_button("Sonraki Alışverişi Tahmin Et")
            
            if predict_next:
                # Basit tahmin modeli
                expected_days = max(1, int(avg_purchase_freq - last_purchase))
                probability = min(1.0, total_purchases / 100)
                expected_amount = avg_basket * (1 + np.random.normal(0, 0.1))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Tahmini Alışveriş Tarihi",
                        f"{expected_days} gün sonra",
                        f"{expected_days - avg_purchase_freq:+.0f} gün"
                    )
                with col2:
                    st.metric(
                        "Tahmini Sepet Tutarı",
                        f"₺{expected_amount:.2f}",
                        f"{((expected_amount - avg_basket) / avg_basket * 100):+.1f}%"
                    )
                
                # Olasılık göstergesi
                fig_prob = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Alışveriş Olasılığı"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "blue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "darkblue"}
                        ]
                    }
                ))
                st.plotly_chart(fig_prob, use_container_width=True)

elif analysis_type == "Cohort Analizi":
    st.subheader("Detaylı Cohort Analizi")
    
    # Cohort tipi seçimi
    cohort_type = st.selectbox(
        "Cohort Tipi",
        ["Kayıt Tarihi", "İlk Alışveriş", "Harcama Seviyesi"]
    )
    
    # Örnek cohort verisi oluşturma
    cohort_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    cohort_data_list = []  # Liste olarak tutacağız
    
    for cohort in cohort_dates:
        # Her cohort için müşteri sayısı
        initial_customers = np.random.randint(100, 500)
        retention_rates = np.random.uniform(0.6, 1.0, 12) ** np.arange(12)
        
        cohort_row = pd.Series(
            initial_customers * retention_rates,
            index=[f'Ay_{i}' for i in range(12)]
        )
        cohort_data_list.append(cohort_row)
    
    # concat ile dataframe oluşturma
    cohort_data = pd.concat(cohort_data_list, axis=1).T
    cohort_data.index = cohort_dates.strftime('%Y-%m')
    
    # Retention Matrix
    st.write("### Retention Matrix")
    retention_matrix = cohort_data.div(cohort_data['Ay_0'], axis=0) * 100
    
    fig_retention = px.imshow(
        retention_matrix,
        labels=dict(x="Ay", y="Cohort", color="Retention %"),
        aspect="auto",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_retention, use_container_width=True)
    
    # Cohort Analiz Metrikleri
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_retention = retention_matrix.mean().mean()
        st.metric(
            "Ortalama Retention",
            f"{avg_retention:.1f}%",
            f"{avg_retention - 100:.1f}% Başlangıca Göre"
        )
    
    with col2:
        best_cohort = retention_matrix.mean(axis=1).idxmax()
        st.metric(
            "En İyi Cohort",
            f"{best_cohort}",
            "En Yüksek Retention"
        )
    
    with col3:
        worst_cohort = retention_matrix.mean(axis=1).idxmin()
        st.metric(
            "En Düşük Performans",
            f"{worst_cohort}",
            "En Düşük Retention"
        )
    
    # Cohort Trend Analizi
    st.write("### Cohort Trend Analizi")
    
    trend_period = st.selectbox(
        "Trend Periyodu",
        ["3. Ay Retention", "6. Ay Retention", "12. Ay Retention"]
    )
    
    period_map = {
        "3. Ay Retention": "Ay_3",
        "6. Ay Retention": "Ay_6",
        "12. Ay Retention": "Ay_11"
    }
    
    trend_data = retention_matrix[period_map[trend_period]]
    fig_trend = px.line(
        trend_data,
        title=f"{trend_period} Trendi",
        labels={'value': 'Retention %', 'index': 'Cohort'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Cohort Karşılaştırma
    st.write("### Cohort Karşılaştırma")
    selected_cohorts = st.multiselect(
        "Cohortları Seçin",
        retention_matrix.index.tolist(),
        default=retention_matrix.index[:3].tolist()
    )
    
    if selected_cohorts:
        comparison_data = retention_matrix.loc[selected_cohorts]
        fig_comparison = px.line(
            comparison_data.T,
            title="Seçili Cohort Karşılaştırması",
            labels={'value': 'Retention %', 'index': 'Ay'}
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

elif analysis_type == "Özel Rapor Oluşturma":
    st.subheader("Özel Rapor Oluşturma")
    
    # Rapor bileşenleri seçimi
    report_sections = st.multiselect(
        "Rapor Bileşenlerini Seçin",
        ["Temel Metrikler", "Müşteri Segmentasyonu", "Satış Analizi",
         "Cohort Analizi", "Tahminler", "Öneriler"]
    )
    
    # Rapor parametreleri
    with st.form("report_parameters"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Başlangıç Tarihi", 
                                     datetime.now() - timedelta(days=90))
            report_title = st.text_input("Rapor Başlığı", "CRM Analiz Raporu")
        with col2:
            end_date = st.date_input("Bitiş Tarihi", datetime.now())
            include_visuals = st.checkbox("Görselleri Dahil Et", True)
        
        generate_report = st.form_submit_button("Rapor Oluştur")
    
    # Form dışında rapor içeriği ve görselleştirmeler
    if generate_report:
        # Grafikleri ve verileri saklayacağımız sözlük
        figures = {}
        report_data = {}

        st.write(f"## {report_title}")
        st.write(f"*Dönem: {start_date.strftime('%d.%m.%Y')} - "
                f"{end_date.strftime('%d.%m.%Y')}*")
        
        for section in report_sections:
            st.write(f"### {section}")
            
            if section == "Temel Metrikler":
                # Temel metrikler görselleştirmesi
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Toplam Müşteri", "1,234", "+12%")
                with col2:
                    st.metric("Ortalama Sepet", "₺856", "+8%")
                with col3:
                    st.metric("Müşteri Sadakati", "76%", "+5%")
                
                # Detaylı metrikler tablosu
                st.write("#### Detaylı Metrikler")
                metrics_df = pd.DataFrame({
                    'Metrik': ['Aktif Müşteri Sayısı', 'Yeni Müşteri Sayısı', 
                              'Churn Oranı', 'Ortalama Sipariş Değeri',
                              'Müşteri Başına Gelir', 'Müşteri Edinme Maliyeti'],
                    'Değer': ['987', '234', '%3.2', '₺856', '₺2,345', '₺125'],
                    'Değişim': ['+10%', '+15%', '-0.5%', '+8%', '+12%', '-5%']
                })
                st.dataframe(metrics_df, hide_index=True)
                
                # Metrikler trend grafiği
                metrics_trend = pd.DataFrame({
                    'Tarih': pd.date_range(start=start_date, end=end_date, freq='D'),
                    'Aktif_Müşteri': np.random.normal(1000, 50, 
                                                    len(pd.date_range(start=start_date, 
                                                                    end=end_date, freq='D')))
                })
                fig_metrics = px.line(metrics_trend, x='Tarih', y='Aktif_Müşteri',
                                    title='Aktif Müşteri Trendi')
                st.plotly_chart(fig_metrics, use_container_width=True)
                figures['metrics_trend'] = fig_metrics
                report_data['metrics'] = metrics_df
            
            elif section == "Satış Analizi":
                st.write("#### Satış Performans Özeti")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Aylık satış trendi
                    monthly_sales = pd.DataFrame({
                        'Ay': pd.date_range(start=start_date, end=end_date, freq='M'),
                        'Satış': np.random.normal(100000, 15000, 
                                                len(pd.date_range(start=start_date, 
                                                                end=end_date, freq='M')))
                    })
                    fig_sales = px.line(monthly_sales, x='Ay', y='Satış',
                                      title='Aylık Satış Trendi')
                    st.plotly_chart(fig_sales, use_container_width=True)
                    figures['sales_trend'] = fig_sales
                
                with col2:
                    # Kategori bazlı satışlar
                    categories = pd.DataFrame({
                        'Kategori': ['Elektronik', 'Giyim', 'Ev', 'Kozmetik'],
                        'Satış': np.random.normal(50000, 10000, 4)
                    })
                    fig_cat = px.pie(categories, values='Satış', names='Kategori',
                                   title='Kategori Bazlı Satışlar')
                    st.plotly_chart(fig_cat, use_container_width=True)
                    figures['category_sales'] = fig_cat
                
                # Satış özet tablosu
                st.write("#### Satış Özet Tablosu")
                sales_summary = pd.DataFrame({
                    'Metrik': ['Toplam Satış', 'Ortalama Günlük Satış', 
                              'En Yüksek Satış Günü', 'Başarılı İşlem Oranı'],
                    'Değer': ['₺1,234,567', '₺45,678', '15 Ekim 2023', '%94.5'],
                    'Değişim': ['+12%', '+8%', '', '+2.3%']
                })
                st.dataframe(sales_summary, hide_index=True)
                report_data['sales_summary'] = sales_summary
            
            elif section == "Cohort Analizi":
                st.write("#### Cohort Retention Matrisi")
                # Cohort verisi oluşturma
                cohort_data = pd.DataFrame(
                    np.random.uniform(0.6, 1.0, (6, 6)) * 100,
                    columns=['Ay 0', 'Ay 1', 'Ay 2', 'Ay 3', 'Ay 4', 'Ay 5'],
                    index=['2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11']
                )
                
                fig_cohort = px.imshow(cohort_data,
                                     labels=dict(x="Ay", y="Cohort", color="Retention %"),
                                     color_continuous_scale="Blues")
                st.plotly_chart(fig_cohort, use_container_width=True)
                figures['cohort_matrix'] = fig_cohort
                
                # Cohort performans metrikleri
                st.write("#### Cohort Performans Özeti")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("En İyi Cohort", "2023-09", "+5% vs Ortalama")
                with col2:
                    st.metric("Ortalama Retention", "68%", "+3% vs Geçen Dönem")
                
                report_data['cohort_data'] = cohort_data
            
            elif section == "Müşteri Segmentasyonu":
                st.write("#### Segment Dağılımı")
                segments = pd.DataFrame({
                    'Segment': ['VIP', 'Sadık Müşteri', 'Orta Segment', 'Risk Grubu'],
                    'Müşteri Sayısı': [234, 567, 890, 123],
                    'Toplam Gelir': ['₺567,890', '₺890,123', '₺456,789', '₺123,456'],
                    'Ortalama Sepet': ['₺2,345', '₺1,567', '₺890', '₺567']
                })
                st.dataframe(segments, hide_index=True)
                
                if include_visuals:
                    fig_segment = px.pie(segments, values='Müşteri Sayısı', 
                                       names='Segment',
                                       title='Segment Dağılımı')
                    st.plotly_chart(fig_segment, use_container_width=True)
                    figures['segment_dist'] = fig_segment
                
                report_data['segments'] = segments
            
            elif section == "Tahminler":
                st.write("#### Gelecek Dönem Tahminleri")
                predictions = pd.DataFrame({
                    'Metrik': ['Beklenen Satış', 'Tahmini Müşteri Sayısı',
                              'Churn Riski', 'Büyüme Potansiyeli'],
                    'Tahmin': ['₺1,345,678', '1,456', '%4.5', '%8.9'],
                    'Değişim': ['+15%', '+10%', '-0.3%', '+2.1%']
                })
                st.dataframe(predictions, hide_index=True)
                
                # Tahmin grafiği
                forecast_data = pd.DataFrame({
                    'Tarih': pd.date_range(start=end_date, 
                                         periods=30, freq='D'),
                    'Tahmin': np.random.normal(100000, 5000, 30)
                })
                fig_forecast = px.line(forecast_data, x='Tarih', y='Tahmin',
                                     title='30 Günlük Satış Tahmini')
                st.plotly_chart(fig_forecast, use_container_width=True)
                figures['forecast'] = fig_forecast
                report_data['predictions'] = predictions
            
            elif section == "Öneriler":
                st.write("#### Aksiyon Önerileri")
                st.write("""
                1. **VIP Müşteriler İçin:**
                   - Özel indirim kampanyası başlatılmalı
                   - Kişiselleştirilmiş ürün önerileri sunulmalı
                   - Sadakat programı geliştirilmeli
                
                2. **Risk Grubu İçin:**
                   - Müşteri memnuniyet anketi yapılmalı
                   - Win-back kampanyası düzenlenmeli
                   - Özel fiyatlandırma stratejisi geliştirilmeli
                
                3. **Orta Segment İçin:**
                   - Cross-selling fırsatları değerlendirilmeli
                   - Kategori bazlı kampanyalar düzenlenmeli
                   - Müşteri deneyimi iyileştirmeleri yapılmalı
                
                4. **Genel Öneriler:**
                   - Email pazarlama stratejisi güçlendirilmeli
                   - Mobil uygulama kullanımı artırılmalı
                   - Müşteri geri bildirimleri düzenli toplanmalı
                """)
        

def generate_html_report(report_title, start_date, end_date, report_sections, data, figures):
    """HTML rapor oluşturma fonksiyonu"""
    html_content = f"""
    <html>
    <head>
        <title>{report_title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .metric-card {{
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }}
            .table-wrapper {{
                overflow-x: auto;
                margin-bottom: 20px;
            }}
            .plot-wrapper {{
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">{report_title}</h1>
            <p class="text-center mb-4">Dönem: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}</p>
    """

    # Her bölüm için içerik ekleme
    for section in report_sections:
        html_content += f'<div class="section"><h2>{section}</h2>'
        
        if section == "Temel Metrikler":
            # KPI kartları
            html_content += '''
            <div class="row">
                <div class="col-md-4">
                    <div class="metric-card bg-primary text-white">
                        <h4>Toplam Müşteri</h4>
                        <h2>1,234</h2>
                        <p>↑ 12% Geçen Aya Göre</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card bg-success text-white">
                        <h4>Ortalama Sepet</h4>
                        <h2>₺856</h2>
                        <p>↑ 8% Geçen Aya Göre</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card bg-info text-white">
                        <h4>Müşteri Sadakati</h4>
                        <h2>76%</h2>
                        <p>↑ 5% Geçen Aya Göre</p>
                    </div>
                </div>
            </div>
            '''
            
            # Metrikler tablosu
            if 'metrics' in data:
                html_content += '''
                <div class="table-wrapper">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                '''
                for col in data['metrics'].columns:
                    html_content += f'<th>{col}</th>'
                html_content += '</tr></thead><tbody>'
                
                for _, row in data['metrics'].iterrows():
                    html_content += '<tr>'
                    for val in row:
                        html_content += f'<td>{val}</td>'
                    html_content += '</tr>'
                html_content += '</tbody></table></div>'
            
            # Metrik trendi grafiği
            if 'metrics_trend' in figures:
                html_content += f'''
                <div class="plot-wrapper">
                    <div id="metric_trend" style="width:100%;height:400px;"></div>
                    <script>
                        var fig = {figures['metrics_trend'].to_json()}
                        Plotly.newPlot('metric_trend', fig.data, fig.layout);
                    </script>
                </div>
                '''
        
        elif section == "Satış Analizi":
            # Satış trendi grafiği
            if 'sales_trend' in figures:
                html_content += f'''
                <div class="plot-wrapper">
                    <div id="sales_trend" style="width:100%;height:400px;"></div>
                    <script>
                        var fig = {figures['sales_trend'].to_json()}
                        Plotly.newPlot('sales_trend', fig.data, fig.layout);
                    </script>
                </div>
                '''
            
            # Kategori bazlı satışlar
            if 'category_sales' in figures:
                html_content += f'''
                <div class="plot-wrapper">
                    <div id="category_sales" style="width:100%;height:400px;"></div>
                    <script>
                        var fig = {figures['category_sales'].to_json()}
                        Plotly.newPlot('category_sales', fig.data, fig.layout);
                    </script>
                </div>
                '''
        
        elif section == "Müşteri Segmentasyonu":
            if 'segments' in data:
                # Segment tablosu
                html_content += '''
                <div class="table-wrapper">
                    <table class="table table-striped">
                        <thead><tr>
                '''
                for col in data['segments'].columns:
                    html_content += f'<th>{col}</th>'
                html_content += '</tr></thead><tbody>'
                
                for _, row in data['segments'].iterrows():
                    html_content += '<tr>'
                    for val in row:
                        html_content += f'<td>{val}</td>'
                    html_content += '</tr>'
                html_content += '</tbody></table></div>'
            
            # Segment dağılım grafiği
            if 'segment_dist' in figures:
                html_content += f'''
                <div class="plot-wrapper">
                    <div id="segment_dist" style="width:100%;height:400px;"></div>
                    <script>
                        var fig = {figures['segment_dist'].to_json()}
                        Plotly.newPlot('segment_dist', fig.data, fig.layout);
                    </script>
                </div>
                '''
        
        elif section == "Cohort Analizi":
            if 'cohort_matrix' in figures:
                html_content += f'''
                <div class="plot-wrapper">
                    <div id="cohort_matrix" style="width:100%;height:500px;"></div>
                    <script>
                        var fig = {figures['cohort_matrix'].to_json()}
                        Plotly.newPlot('cohort_matrix', fig.data, fig.layout);
                    </script>
                </div>
                '''
        
        elif section == "Öneriler":
            html_content += '''
            <div class="card">
                <div class="card-body">
                    <h4>VIP Müşteriler İçin:</h4>
                    <ul>
                        <li>Özel indirim kampanyası başlatılmalı</li>
                        <li>Kişiselleştirilmiş ürün önerileri sunulmalı</li>
                        <li>Sadakat programı geliştirilmeli</li>
                    </ul>
                    
                    <h4>Risk Grubu İçin:</h4>
                    <ul>
                        <li>Müşteri memnuniyet anketi yapılmalı</li>
                        <li>Win-back kampanyası düzenlenmeli</li>
                        <li>Özel fiyatlandırma stratejisi geliştirilmeli</li>
                    </ul>
                    
                    <h4>Orta Segment İçin:</h4>
                    <ul>
                        <li>Cross-selling fırsatları değerlendirilmeli</li>
                        <li>Kategori bazlı kampanyalar düzenlenmeli</li>
                        <li>Müşteri deneyimi iyileştirmeleri yapılmalı</li>
                    </ul>
                </div>
            </div>
            '''
        
        html_content += '</div>'
    
    html_content += '''
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''
    
    return html_content

# Streamlit uygulamasında kullanımı
if generate_report:
    st.write(f"## {report_title}")
    st.write(f"*Dönem: {start_date.strftime('%d.%m.%Y')} - "
            f"{end_date.strftime('%d.%m.%Y')}*")
    
    # Rapor içeriğini ve grafikleri oluştur
    figures = {}
    report_data = {}
    
    # Her bölüm için veri ve grafik hazırla
    for section in report_sections:
        if section == "Temel Metrikler":
            metrics_df = pd.DataFrame({
                'Metrik': ['Aktif Müşteri', 'Yeni Müşteri', 'Churn Oranı'],
                'Değer': ['987', '234', '%3.2'],
                'Değişim': ['+10%', '+15%', '-0.5%']
            })
            report_data['metrics'] = metrics_df
            
            # Metrik trendi
            trend_data = pd.DataFrame({
                'Tarih': pd.date_range(start=start_date, end=end_date, freq='D'),
                'Değer': np.random.normal(1000, 50, (end_date - start_date).days + 1)
            })
            figures['metrics_trend'] = px.line(trend_data, x='Tarih', y='Değer',
                                             title='Metrik Trendi')
        
        elif section == "Satış Analizi":
            # Satış trendi
            sales_data = pd.DataFrame({
                'Tarih': pd.date_range(start=start_date, end=end_date, freq='D'),
                'Satış': np.random.normal(10000, 2000, (end_date - start_date).days + 1)
            })
            figures['sales_trend'] = px.line(sales_data, x='Tarih', y='Satış',
                                           title='Satış Trendi')
            
            # Kategori bazlı satışlar
            categories = pd.DataFrame({
                'Kategori': ['A', 'B', 'C', 'D'],
                'Satış': np.random.normal(5000, 1000, 4)
            })
            figures['category_sales'] = px.pie(categories, values='Satış', 
                                             names='Kategori',
                                             title='Kategori Bazlı Satışlar')
    
    # HTML rapor oluştur
    html_report = generate_html_report(
        report_title=report_title,
        start_date=start_date,
        end_date=end_date,
        report_sections=report_sections,
        data=report_data,
        figures=figures
    )
    
    # HTML raporu indir
    b64 = base64.b64encode(html_report.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{report_title}.html">HTML Raporu İndir</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Önizleme
    st.write("### Rapor Önizleme")
    st.components.v1.html(html_report, height=600, scrolling=True)

elif analysis_type == "Müşteri Segmentasyonu":
    st.subheader("Müşteri Segmentasyonu Analizi")
    
    # K-means için veri hazırlığı
    features = ['ToplamHarcama', 'AlisverisFrekansi', 'WebsiteZiyareti']
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means modelini eğitme
    n_clusters = st.slider("Segment Sayısı Seçin", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Segment'] = kmeans.fit_predict(X_scaled)
    
    # Segment analizi
    st.write("### Segment Özellikleri")
    segment_stats = data.groupby('Segment').agg({
        'ToplamHarcama': 'mean',
        'AlisverisFrekansi': 'mean',
        'WebsiteZiyareti': 'mean',
        'MusteriID': 'count'
    }).round(2)
    
    segment_stats.columns = ['Ort. Harcama', 'Ort. Alışveriş Frekansı', 
                           'Ort. Website Ziyareti', 'Müşteri Sayısı']
    st.dataframe(segment_stats)
    
    # Segment görselleştirmesi
    fig = px.scatter_3d(data, x='ToplamHarcama', y='AlisverisFrekansi', 
                       z='WebsiteZiyareti', color='Segment',
                       title='3D Segment Görselleştirmesi')
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "RFM Analizi":
    st.subheader("RFM (Recency, Frequency, Monetary) Analizi")
    
    # RFM Skorları hesaplama
    data['R_Score'] = pd.qcut(data['SonAlisverisGunu'], q=5, labels=[5,4,3,2,1])
    data['F_Score'] = pd.qcut(data['AlisverisFrekansi'], q=5, labels=[1,2,3,4,5])
    data['M_Score'] = pd.qcut(data['ToplamHarcama'], q=5, labels=[1,2,3,4,5])
    
    # RFM Segmentleri
    def get_segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'VIP'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Sadık Müşteri'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
            return 'Potansiyel Sadık'
        else:
            return 'Risk Altında'
    
    data['RFM_Segment'] = data.apply(get_segment, axis=1)
    
    # Segment dağılımı
    st.write("### RFM Segment Dağılımı")
    fig_rfm = px.pie(data, names='RFM_Segment', title='RFM Segment Dağılımı')
    st.plotly_chart(fig_rfm, use_container_width=True)
    
    # Segment detayları
    st.write("### Segment Detayları")
    rfm_stats = data.groupby('RFM_Segment').agg({
        'MusteriID': 'count',
        'ToplamHarcama': 'mean',
        'AlisverisFrekansi': 'mean',
        'SonAlisverisGunu': 'mean'
    }).round(2)
    
    st.dataframe(rfm_stats)

elif analysis_type == "A/B Test Analizi":
    st.subheader("A/B Test Analizi")
    
    # Test senaryosu seçimi
    test_scenario = st.selectbox(
        "Test Senaryosu Seçin",
        ["Email Kampanyası", "Website Tasarımı", "Fiyatlandırma", "Özel Senaryo"]
    )
    
    # A/B Test veri simülasyonu
    def generate_ab_test_data(n_samples=1000):
        if test_scenario == "Email Kampanyası":
            control = np.random.normal(0.12, 0.02, n_samples)
            variant = np.random.normal(0.15, 0.02, n_samples)
            metric_name = "Tıklama Oranı"
        elif test_scenario == "Website Tasarımı":
            control = np.random.normal(120, 20, n_samples)
            variant = np.random.normal(135, 20, n_samples)
            metric_name = "Ziyaret Süresi (sn)"
        else:
            control = np.random.normal(100, 20, n_samples)
            variant = np.random.normal(110, 20, n_samples)
            metric_name = "Dönüşüm"
            
        return control, variant, metric_name
    
    control_data, variant_data, metric_name = generate_ab_test_data()
    
    # İstatistiksel analiz
    t_stat, p_value = stats.ttest_ind(control_data, variant_data)
    
    # Sonuçların görselleştirilmesi
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Kontrol vs Varyant Karşılaştırması")
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Box(y=control_data, name="Kontrol Grubu"))
        fig_ab.add_trace(go.Box(y=variant_data, name="Varyant Grubu"))
        fig_ab.update_layout(title=f"{metric_name} Dağılımı")
        st.plotly_chart(fig_ab, use_container_width=True)
    
    with col2:
        st.write("### Test Sonuçları")
        st.write(f"p-değeri: {p_value:.4f}")
        if p_value < 0.05:
            st.success("İstatistiksel olarak anlamlı bir fark bulundu! (p < 0.05)")
        else:
            st.warning("İstatistiksel olarak anlamlı bir fark bulunamadı. (p >= 0.05)")
        
        improvement = ((np.mean(variant_data) - np.mean(control_data)) / 
                      np.mean(control_data) * 100)
        st.metric(
            "İyileştirme Oranı",
            f"{improvement:.1f}%",
            delta=f"{improvement:.1f}%"
        )


elif analysis_type == "Churn Prediction":
    st.subheader("Churn (Müşteri Kaybı) Tahminleme")
    
    # Churn verisi hazırlama
    def prepare_churn_features(df):
        features = df.copy()
        features['IsChurn'] = (features['SonAlisverisGunu'] > 60).astype(int)
        features['CustomerLifetime'] = (datetime.now() - 
                                      features['KayitTarihi']).dt.days
        
        # Kategorik değişkenleri dönüştürme
        le = LabelEncoder()
        features['YasGrubu'] = le.fit_transform(features['YasGrubu'])
        features['Cinsiyet'] = le.fit_transform(features['Cinsiyet'])
        features['Sehir'] = le.fit_transform(features['Sehir'])
        
        return features
    
    churn_data = prepare_churn_features(data)
    
    # Model özellikleri
    feature_cols = ['ToplamHarcama', 'AlisverisFrekansi', 'WebsiteZiyareti',
                   'EmailAcma', 'SepetTerkOrani', 'NPSPuani', 'YasGrubu',
                   'CustomerLifetime']
    
    X = churn_data[feature_cols]
    y = churn_data['IsChurn']
    
    # Model seçimi
    model_type = st.selectbox(
        "Model Seçin",
        ["Random Forest", "XGBoost", "Gradient Boosting"]
    )
    
    # Model eğitimi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier(random_state=42)
    else:
        model = GradientBoostingClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model performansı
    st.write("### Model Performansı")
    col1, col2 = st.columns(2)
    
    with col1:
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig_conf = px.imshow(conf_matrix,
                           labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
                           title="Confusion Matrix")
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("Sınıflandırma Raporu:")
        st.dataframe(pd.DataFrame(report).transpose())
    
    # Özellik önemliliği
    if model_type in ["Random Forest", "XGBoost"]:
        st.write("### Özellik Önemliliği")
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_imp = px.bar(feature_imp, x='importance', y='feature',
                        orientation='h',
                        title="Özellik Önemliliği")
        st.plotly_chart(fig_imp, use_container_width=True)

elif analysis_type == "Özel Rapor Oluşturma":
    st.subheader("Özel Rapor Oluşturma")
    
    # Rapor bileşenleri seçimi
    report_sections = st.multiselect(
        "Rapor Bileşenlerini Seçin",
        ["Temel Metrikler", "Müşteri Segmentasyonu", "Satış Analizi",
         "Cohort Analizi", "Tahminler", "Öneriler"]
    )
    
    # Form yapısı
    with st.form("report_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Başlangıç Tarihi", 
                                     datetime.now() - timedelta(days=90))
            report_title = st.text_input("Rapor Başlığı", "CRM Analiz Raporu")
        with col2:
            end_date = st.date_input("Bitiş Tarihi", datetime.now())
            include_visuals = st.checkbox("Görselleri Dahil Et", True)
        
        submitted = st.form_submit_button("Rapor Oluştur")
    
    if submitted:
        st.write(f"## {report_title}")
        st.write(f"*Dönem: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}*")
        
        # Veri ve grafikleri saklayacak sözlükler
        figures = {}
        report_data = {}
        
        for section in report_sections:
            st.write(f"### {section}")
            
            if section == "Müşteri Segmentasyonu":
                # RFM Segmentasyonu
                rfm_data = pd.DataFrame({
                    'Segment': ['VIP', 'Sadık Müşteri', 'Orta Segment', 'Risk Grubu'],
                    'Müşteri Sayısı': [234, 567, 890, 123],
                    'Toplam Gelir': ['₺567,890', '₺890,123', '₺456,789', '₺123,456'],
                    'Ortalama Sepet': ['₺2,345', '₺1,567', '₺890', '₺567'],
                    'Retention Oranı': ['92%', '85%', '67%', '45%']
                })
                st.dataframe(rfm_data)
                
                # Segment dağılım grafiği
                fig_segment = px.pie(rfm_data, values='Müşteri Sayısı', names='Segment',
                                   title='Müşteri Segment Dağılımı')
                st.plotly_chart(fig_segment)
                figures['segment_dist'] = fig_segment
                
                # Segment karşılaştırma grafiği
                fig_segment_comp = px.bar(rfm_data, x='Segment', y='Müşteri Sayısı',
                                        title='Segment Bazlı Müşteri Dağılımı')
                st.plotly_chart(fig_segment_comp)
                figures['segment_comp'] = fig_segment_comp
                
                report_data['segments'] = rfm_data
            
            elif section == "Cohort Analizi":
                # Cohort matrisi oluşturma
                months = ['Ay 0', 'Ay 1', 'Ay 2', 'Ay 3', 'Ay 4', 'Ay 5']
                cohorts = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
                
                cohort_data = pd.DataFrame(
                    np.random.uniform(0.6, 1.0, (len(cohorts), len(months))) * 100,
                    columns=months,
                    index=cohorts
                )
                
                # Cohort heatmap
                fig_cohort = px.imshow(cohort_data,
                                     labels=dict(x="Ay", y="Cohort", color="Retention %"),
                                     color_continuous_scale="Blues",
                                     title="Cohort Retention Analizi (%)")
                st.plotly_chart(fig_cohort)
                figures['cohort_matrix'] = fig_cohort
                
                # Cohort metrikleri
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ortalama Retention", "68%", "+3%")
                with col2:
                    st.metric("En İyi Cohort", "2024-03", "92%")
                with col3:
                    st.metric("Son Ay Retention", "73%", "+5%")
                
                # Cohort trend analizi
                cohort_trend = pd.DataFrame({
                    'Cohort': cohorts,
                    'Retention': [75, 78, 82, 79, 81, 83]
                })
                fig_trend = px.line(cohort_trend, x='Cohort', y='Retention',
                                  title='Cohort Retention Trendi')
                st.plotly_chart(fig_trend)
                figures['cohort_trend'] = fig_trend
                
                report_data['cohort_data'] = cohort_data
            
            elif section == "Tahminler":
                # Gelecek dönem tahminleri
                predictions = pd.DataFrame({
                    'Metrik': ['Beklenen Satış', 'Tahmini Müşteri Sayısı',
                              'Churn Riski', 'Büyüme Potansiyeli', 
                              'CLV Tahmini', 'Sepet Değeri Tahmini'],
                    'Tahmin': ['₺1,345,678', '1,456', '%4.5', '%8.9',
                              '₺12,345', '₺945'],
                    'Değişim': ['+15%', '+10%', '-0.3%', '+2.1%',
                               '+7%', '+4%']
                })
                st.dataframe(predictions)
                
                # Tahmin grafikleri
                forecast_data = pd.DataFrame({
                    'Tarih': pd.date_range(start=end_date, periods=90, freq='D'),
                    'Tahmin': np.random.normal(100000, 5000, 90),
                    'Alt Sınır': np.random.normal(90000, 5000, 90),
                    'Üst Sınır': np.random.normal(110000, 5000, 90)
                })
                
                fig_forecast = go.Figure([
                    go.Scatter(x=forecast_data['Tarih'], y=forecast_data['Üst Sınır'],
                             fill=None, mode='lines', line_color='rgba(0,100,80,0.2)',
                             name='Üst Sınır'),
                    go.Scatter(x=forecast_data['Tarih'], y=forecast_data['Alt Sınır'],
                             fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)',
                             name='Alt Sınır'),
                    go.Scatter(x=forecast_data['Tarih'], y=forecast_data['Tahmin'],
                             mode='lines', line_color='rgb(0,100,80)',
                             name='Tahmin')
                ])
                fig_forecast.update_layout(title='90 Günlük Satış Tahmini')
                st.plotly_chart(fig_forecast)
                figures['forecast'] = fig_forecast
                
                # Churn riski dağılımı
                churn_data = pd.DataFrame({
                    'Risk Seviyesi': ['Düşük', 'Orta', 'Yüksek'],
                    'Müşteri Sayısı': [567, 234, 123]
                })
                fig_churn = px.pie(churn_data, values='Müşteri Sayısı', 
                                 names='Risk Seviyesi',
                                 title='Churn Risk Dağılımı')
                st.plotly_chart(fig_churn)
                figures['churn_dist'] = fig_churn
                
                report_data['predictions'] = predictions
            
            elif section == "Temel Metrikler":
                # [Mevcut Temel Metrikler kodu]
                pass
            
            elif section == "Satış Analizi":
                # [Mevcut Satış Analizi kodu]
                pass
            
            elif section == "Öneriler":
                # [Mevcut Öneriler kodu]
                pass
        
        # HTML rapor oluşturma ve indirme butonu
        if st.button("HTML Rapor İndir"):
            html_content = f"""
            <html>
                <head>
                    <title>{report_title}</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    <style>
                        body {{ padding: 20px; }}
                        .section {{ margin-bottom: 30px; }}
                        .metric-card {{
                            padding: 15px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            margin-bottom: 15px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1 class="text-center mb-4">{report_title}</h1>
                        <p class="text-center mb-4">
                            Dönem: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}
                        </p>
            """
            
            # Her bölüm için içerik ekleme
            for section in report_sections:
                html_content += f'<div class="section"><h2>{section}</h2>'
                
                # Her bölüm için özel içerik ekleme
                if section in figures:
                    for fig_name, fig in figures[section].items():
                        html_content += f"""
                        <div class="plot-wrapper">
                            <div id="{fig_name}" style="width:100%;height:400px;"></div>
                            <script>
                                var fig = {fig.to_json()};
                                Plotly.newPlot('{fig_name}', fig.data, fig.layout);
                            </script>
                        </div>
                        """
                
                if section in report_data:
                    html_content += report_data[section].to_html(
                        classes='table table-striped',
                        index=False
                    )
                
                html_content += '</div>'
            
            html_content += """
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                </body>
            </html>
            """
            
            # HTML raporu indir
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{report_title}.html">HTML Raporu İndir</a>'
            st.markdown(href, unsafe_allow_html=True)

# Export seçenekleri
export_format = st.sidebar.selectbox(
    "Export Formatı",
    ["CSV", "Excel", "PDF", "JSON"]
)

if st.sidebar.button("Analiz Verilerini Export Et"):
    if export_format == "CSV":
        csv = data.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="CSV İndir",
            data=csv,
            file_name="crm_analiz.csv",
            mime="text/csv"
        )


# Footer
st.markdown("""
<style>
    footer {
        visibility: hidden;
    }
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to right, rgba(17, 24, 39, 0.7), rgba(31, 41, 55, 0.7));
        backdrop-filter: blur(10px);
        color: #9ca3af;
        text-align: center;
        padding: 12px;
        font-size: 0.8rem;
        border-top: 1px solid rgba(156, 163, 175, 0.2);
        z-index: 999;
    }
    .social-links {
        margin: 8px 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        flex-wrap: wrap;
    }
    .social-links a {
        display: inline-flex;
        align-items: center;
        padding: 8px 15px;
        border-radius: 8px;
        text-decoration: none;
        color: #9ca3af;
        background: rgba(17, 24, 39, 0.3);
        border: 1px solid rgba(156, 163, 175, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .social-links a:hover {
        transform: translateY(-2px);
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    .social-links i {
        font-size: 18px;
        margin-right: 8px;
        transition: transform 0.3s ease;
    }
    /* Özel ikon renkleri */
    .fa-linkedin { color: #0077b5; }
    .fa-github { color: #f5f5f5; }
    .fa-medium { color: #00ab6c; }
    .fa-chart-bar { color: #03ef62; }
    .fa-kaggle { color: #20beff; }
    .fa-envelope { color: #ff4444; }
    
    .social-links a:hover i {
        transform: scale(1.2);
    }
    .copyright {
        margin-top: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        letter-spacing: 0.5px;
        background: linear-gradient(45deg, #9ca3af, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        opacity: 0.9;
    }
    .custom-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(to right, 
            transparent, 
            rgba(59, 130, 246, 0.5), 
            transparent
        );
    }
    @media (max-width: 768px) {
        .social-links {
            gap: 10px;
        }
        .social-links a {
            padding: 6px 12px;
            font-size: 0.75rem;
        }
    }
    /* Tooltip stili */
    .social-links a::before {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        border-radius: 4px;
        font-size: 12px;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
    }
    .social-links a:hover::before {
        opacity: 1;
        visibility: visible;
        bottom: calc(100% + 10px);
    }
</style>

<div class="custom-footer">
    <div class="social-links">
        <a href="https://www.linkedin.com/in/Ysntns" target="_blank" data-tooltip="LinkedIn Profilim">
            <i class="fab fa-linkedin"></i> LinkedIn
        </a>
        <a href="https://github.com/Ysntns" target="_blank" data-tooltip="GitHub Projelerim">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a href="https://medium.com/@Ysntns" target="_blank" data-tooltip="Medium Yazılarım">
            <i class="fab fa-medium"></i> Medium
        </a>
        <a href="https://www.datacamp.com/portfolio/ysntnss" target="_blank" data-tooltip="DataCamp Portfolyom">
            <i class="fas fa-chart-bar"></i> DataCamp
        </a>
        <a href="https://www.kaggle.com/ysntnss" target="_blank" data-tooltip="Kaggle Profilim">
            <i class="fab fa-kaggle"></i> Kaggle
        </a>
        <a href="mailto:ysn.tnss@gmail.com" data-tooltip="Mail Gönder">
            <i class="fas fa-envelope"></i> Email
        </a>
    </div>
    <div class="copyright">
        © 2024 CRM Analitik Dashboard | Version 1.0.0 | Tüm hakları Yasin TANIŞ'a aittir.
    </div>
</div>

<script src="https://kit.fontawesome.com/your-font-awesome-kit.js" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)

# Font Awesome ve Google Fonts için head kısmına eklenecek linkler
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
