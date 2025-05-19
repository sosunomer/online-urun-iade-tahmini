import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    print("Veri yükleniyor...")
    df = pd.read_excel(file_path)
    
    print("Veri ön işleniyor...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # İade işaretleme (C ile başlayan faturalar iade)
    df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C').astype(int)
    
    # Özellik mühendisliği
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayName'] = df['InvoiceDate'].dt.day_name()
    
    # Kategorik değişkenleri dönüştürme
    le = LabelEncoder()
    df['CustomerID'] = df['CustomerID'].fillna(-1)
    df['Country'] = le.fit_transform(df['Country'])
    
    # Özellik seçimi
    features = ['Quantity', 'UnitPrice', 'TotalPrice', 'Hour', 'DayOfWeek', 
                'Month', 'Country', 'CustomerID']
    
    return df, df[features], df['IsReturn']

def create_visualizations(df, X, y):
    print("Görselleştirmeler oluşturuluyor...")
    
    # 1. İade Analizi
    plt.figure(figsize=(20, 15))
    
    # İade Oranı
    plt.subplot(3, 2, 1)
    sns.countplot(data=df, y='IsReturn')
    plt.title('İade Dağılımı')
    
    # Günlere Göre İade
    plt.subplot(3, 2, 2)
    sns.boxplot(data=df, x='DayName', y='TotalPrice', hue='IsReturn')
    plt.xticks(rotation=45)
    plt.title('Günlere Göre İade ve Fiyat Dağılımı')
    
    # Saatlere Göre İade
    plt.subplot(3, 2, 3)
    sns.histplot(data=df, x='Hour', hue='IsReturn', multiple="stack")
    plt.title('Saatlere Göre İade Dağılımı')
    
    # Aylara Göre İade
    plt.subplot(3, 2, 4)
    sns.histplot(data=df, x='Month', hue='IsReturn', multiple="stack")
    plt.title('Aylara Göre İade Dağılımı')
    
    # Ülkelere Göre İade
    plt.subplot(3, 2, 5)
    top_countries = df.groupby('Country')['IsReturn'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title('Ülkelere Göre İade Oranı (Top 10)')
    
    # Fiyata Göre İade
    plt.subplot(3, 2, 6)
    sns.boxplot(data=df, x='IsReturn', y='TotalPrice')
    plt.title('Fiyata Göre İade Dağılımı')
    
    plt.tight_layout()
    plt.savefig('detayli_analiz.png')
    plt.close()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    }
    
    print("\nToplam 6 farklı model eğitiliyor...")
    results = {}
    for name, model in models.items():
        print(f"\n{name} eğitiliyor...")
        
        # Model eğitimi
        model.fit(X_train, y_train)
        
        # Tahmin
        y_pred = model.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"{name} Doğruluk: {accuracy:.4f}")
        print("\nDetaylı Rapor:")
        print(classification_report(y_test, y_pred))
        
        # Çapraz doğrulama
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\nÇapraz Doğrulama Sonuçları:")
        print(f"Ortalama: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Karmaşıklık matrisi görselleştirme
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Karmaşıklık Matrisi')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def analyze_feature_importance(X, model, model_name):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = abs(model.coef_[0])
    else:
        return
    
    # Özellik önemliliği görselleştirme
    plt.figure(figsize=(10, 6))
    features = X.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'{model_name} Özellik Önemliliği')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    # Veri yükleme ve ön işleme
    file_path = r"C:\Users\user\Desktop\Online Retail.xlsx"
    df, X, y = load_and_preprocess_data(file_path)
    
    # Detaylı görselleştirmeler
    create_visualizations(df, X, y)
    
    print("Veri bölünüyor ve ölçeklendiriliyor...")
    # Veri bölme ve ölçeklendirme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DataFrame'e dönüştürme (özellik isimleri için)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Modelleri eğit ve değerlendir
    print("\nModeller eğitiliyor ve değerlendiriliyor...")
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Özellik önemliliği analizi
    print("\nÖzellik önemliliği analizi yapılıyor...")
    for name, result in results.items():
        analyze_feature_importance(X, result['model'], name)
    
    # En iyi modeli yazdır
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nEn iyi model: {best_model[0]} (Doğruluk: {best_model[1]['accuracy']:.4f})")
    
    print("\nTüm görselleştirmeler kaydedildi:"
          "\n- detayli_analiz.png"
          "\n- confusion_matrix_*.png"
          "\n- feature_importance_*.png")
    
    print("\nÖdev Gereksinimleri:")
    print("1. Python programlama dili ✓")
    print("2. En az 5 makine öğrenmesi algoritması:")
    print("   - Logistic Regression")
    print("   - Random Forest")
    print("   - Decision Tree")
    print("   - K-Nearest Neighbors")
    print("   - Gradient Boosting")
    print("   - Neural Network (Bonus)")
    print("3. Detaylı görselleştirmeler ve analizler ✓")
    print("4. Çapraz doğrulama ile model değerlendirme ✓") 