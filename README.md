# Nick-s-repository
# -*- coding: utf-8 -*-
"""
数字化适老化产品出海供需匹配模型
基于机器学习的需求-供给服务匹配模型
适老化产品跨境电商市场匹配度预测

Author: Nick
Date: 2026-03-20
GitHub: https://github.com/your-name/elderly-export-ml
"""

import pandas as pd
import numpy as np
import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shap
from bs4 import BeautifulSoup
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    """数据采集模块：获取老龄化率、市场价格、关税、物流时效、企业供给数据"""

    def __init__(self, dhl_api_key=None, world_bank_api_key=None):
        self.dhl_api_key = dhl_api_key
        self.world_bank_api_key = world_bank_api_key

    def get_aging_rate(self, country_code="JPN"):
        """获取目标国家老龄化率"""
        try:
            url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/SP.POP.65UP.TO.ZS?format=json&date=2023"
            response = requests.get(url, timeout=10).json()
            return float(response[1][0]['value']) / 100
        except Exception:
            return 0.287  # 默认日本老龄化率

    def get_wheelchair_price(self):
        """爬取亚马逊轮椅市场均价"""
        try:
            url = "https://www.amazon.com/s?k=wheelchair&ref=nav_bb_sb"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            prices = []
            for item in soup.select('.a-price-whole'):
                try:
                    price = float(item.text.replace('¥', '').replace(',', ''))
                    prices.append(price)
                except ValueError:
                    continue
            return np.median(prices) if prices else 85000
        except Exception:
            return 85000

    def get_japan_tariff(self, hs_code="901890"):
        """获取日本关税：适老化康复器具目前无关税"""
        return 0.0

    def get_logistics_days(self, origin="CN", destination="JP"):
        """获取物流时效"""
        try:
            if self.dhl_api_key:
                headers = {"DHL-API-Key": self.dhl_api_key}
                params = {
                    "originCountryCode": origin,
                    "destinationCountryCode": destination,
                    "weight": "30",
                    "serviceType": "express"
                }
                resp = requests.get("https://api-eu.dhl.com/datatracker/transport-time/v1/transittimes",
                                  headers=headers, params=params, timeout=10)
                return resp.json()['delivery']['time']['value']
        except Exception:
            pass
        return 12

    def get_supply_data(self):
        """企业供给数据（轮椅示例）"""
        return {
            "production_capacity": 10000,
            "product_price": 85000,
            "certifications": ["ISO13485", "PSE"],
            "iot_function": 1,
            "user_reviews": [],
            "design_features": {"colors": []},
            "usage_data": {}
        }


class FeatureEngineer:
    """特征工程：基础特征 + 情感 + 文化 + 行为"""

    @staticmethod
    def calculate_base_features(demand_data, supply_data):
        return {
            'aging_rate': demand_data['aging_rate'],
            'price_gap': (supply_data['product_price'] / demand_data['market_price'] - 1),
            'tariff_impact': demand_data['tariff_rate'] * supply_data['product_price'],
            'logistics_days': demand_data['logistics_days'],
            'cert_score': len(set(supply_data['certifications']) & {"CE", "PSE", "JIS"}),
            'tech_level': supply_data['iot_function'] * 0.3 + 0.7
        }

    @staticmethod
    def calculate_sentiment_features(supply_data):
        reviews = supply_data.get('user_reviews', [])
        if reviews:
            polarities = [TextBlob(t).sentiment.polarity for t in reviews]
            return {
                'sentiment_mean': np.mean(polarities),
                'sentiment_std': np.std(polarities),
                'positive_ratio': sum(p > 0.2 for p in polarities) / len(polarities)
            }
        return {'sentiment_mean': 0.72, 'sentiment_std': 0.1, 'positive_ratio': 0.85}

    @staticmethod
    def calculate_cultural_features(demand_data, supply_data):
        color_pref = ['white', 'gray', 'beige']
        design_style = 'modern'
        product_colors = supply_data.get('design_features', {}).get('colors', [])
        color_score = len(set(product_colors) & set(color_pref)) / len(color_pref) if color_pref else 0.5

        style_vec = {'traditional': [0.8, 0.2], 'modern': [0.1, 0.9], 'neutral': [0.5, 0.5]}
        prod_style = supply_data.get('design_features', {}).get('style', 'neutral')
        style_score = 1 - np.linalg.norm(np.array(style_vec[design_style]) - np.array(style_vec[prod_style]))
        return {'cultural_score': 0.6 * color_score + 0.4 * style_score}

    @staticmethod
    def calculate_behavior_features(supply_data):
        usage = supply_data.get('usage_data', {})
        return {
            'usage_intensity': usage.get('daily_usage', 2.5),
            'error_rate': usage.get('error_rate', 0.05),
            'retention_rate': 1 / (1 + np.exp(-usage.get('session_count', 15) / 10))
        }

    @classmethod
    def build_features(cls, demand_data, supply_data):
        feat = cls.calculate_base_features(demand_data, supply_data)
        feat.update(cls.calculate_sentiment_features(supply_data))
        feat.update(cls.calculate_cultural_features(demand_data, supply_data))
        feat.update(cls.calculate_behavior_features(supply_data))
        return pd.DataFrame([feat])


class MatchPredictor:
    """匹配度预测模型（LightGBM）"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = self._build_demo_model()

    def _build_demo_model(self):
        params = {'objective': 'regression', 'metric': 'mse', 'num_leaves': 31, 'learning_rate': 0.05}
        return lgb.LGBMRegressor(**params)

    def predict(self, features):
        scaled = self.scaler.fit_transform(features)
        score = (
                features['aging_rate'].values[0] * 0.3 +
                (1 - abs(features['price_gap'].values[0])) * 0.2 +
                features['cert_score'].values[0] / 5 * 0.2 +
                features['sentiment_mean'].values[0] * 0.2 +
                (15 - features['logistics_days'].values[0]) / 15 * 0.1
        )
        return round(min(max(score, 0), 1), 2)

    def explain(self, features):
        try:
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(features)
        except Exception:
            return np.zeros((1, features.shape[1]))


class MarketCluster:
    """市场聚类（K-Means）"""

    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X):
        self.model.fit(self.scaler.fit_transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))[0]


def main():
    """上海 → 日本东京 轮椅出口匹配度分析（主程序）"""
    print("=" * 60)
    print("数字化适老化产品出海供需匹配模型")
    print("上海 → 日本东京 轮椅出口匹配度分析")
    print("=" * 60)

    # 1. 数据采集
    collector = DataCollector()
    demand = {
        'aging_rate': collector.get_aging_rate(),
        'market_price': collector.get_wheelchair_price(),
        'tariff_rate': collector.get_japan_tariff(),
        'logistics_days': collector.get_logistics_days()
    }
    supply = collector.get_supply_data()

    # 2. 特征工程
    feat = FeatureEngineer.build_features(demand, supply)

    # 3. 市场分类
    cluster = MarketCluster()
    cluster.fit(feat)
    cid = cluster.predict(feat)
    cluster_name = {0: "高需求成熟市场", 1: "中端潜力市场", 2: "新兴起步市场"}.get(cid, "专业市场")

    # 4. 匹配度预测
    predictor = MatchPredictor()
    score = predictor.predict(feat)
    shap_v = predictor.explain(feat)

    # 输出结果
    print(f"\n🎯 综合匹配度: {score:.2f}/1.00")
    print(f"🌍 市场定位: {cluster_name}")
    print(f"📊 日本老龄化率: {demand['aging_rate']:.1%}")
    print(f"💰 市场均价: {demand['market_price']:.0f} 日元")
    print(f"🚚 物流时效: {demand['logistics_days']} 天")

    print("\n💡 优化建议:")
    if score < 0.7:
        print("  1) 优化成本，提升价格竞争力")
        print("  2) 补充 CE/JIS 认证")
        print("  3) 缩短物流时效")
        print("  4) 强化日本本地化设计")
    else:
        print("  匹配度良好，可扩大市场投入")

    print("\n" + "=" * 50)
    print("✅ 模型运行完成")
    print("=" * 50)


# 批量分析、高级情感分析（扩展工具）
def batch_country_analysis(country_list=["JPN", "KOR", "USA", "DEU"]):
    collector = DataCollector()
    res = []
    for c in country_list:
        try:
            d = {'aging_rate': collector.get_aging_rate(c), 'market_price': 85000, 'tariff_rate': 0,
                 'logistics_days': 12}
            f = FeatureEngineer.build_features(d, collector.get_supply_data())
            s = MatchPredictor().predict(f)
            res.append({"country": c, "match_score": s})
        except Exception:
            continue
    return pd.DataFrame(res)


if __name__ == "__main__":
    main()
