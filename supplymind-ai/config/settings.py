"""Central configuration for the SupplyMind AI platform."""

import os
from dotenv import load_dotenv

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    _CUDA_AVAILABLE = False

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPICS = {
    "news": "news_stream",
    "social": "social_stream",
    "supply": "supply_stream",
}
STREAM_DELAY = float(os.getenv("STREAM_DELAY", 0.05))

# ── Dataset file lists ───────────────────────────────────────────────────────
NEWS_FILES = ["all-data.csv", "financial_news_events.csv"]
SOCIAL_FILES = [
    "submissions_reddit_finance.csv",
    "submissions_reddit_stockmarket.csv",
    "submissions_reddit_wallstreet_bets.csv",
]
SUPPLY_FILES = [
    "global_supply_chain_disruption_v1.csv",
    "global_supply_chain_risk_2026.csv",
]

# ── Delta Lake paths ─────────────────────────────────────────────────────────
DELTA_DIR = os.path.join(BASE_DIR, "delta_lake")
DELTA_PATHS = {
    "silver_news": os.path.join(DELTA_DIR, "silver_news"),
    "silver_social": os.path.join(DELTA_DIR, "silver_social"),
    "silver_supply": os.path.join(DELTA_DIR, "silver_supply"),
    "gold_alerts": os.path.join(DELTA_DIR, "gold_alerts"),
}
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── NLP models ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NER_MODEL = "dslim/bert-base-NER"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
RISK_LABELS = ["low risk", "medium risk", "high risk", "critical risk"]

# ── GPU settings ─────────────────────────────────────────────────────────────
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true" and _CUDA_AVAILABLE
DEVICE = "cuda" if USE_GPU else "cpu"
BATCH_SIZE_GPU = 256
BATCH_SIZE_CPU = 64
BATCH_SIZE = BATCH_SIZE_GPU if USE_GPU else BATCH_SIZE_CPU

# ── Text column auto-detection keywords ──────────────────────────────────────
TEXT_COLUMN_KEYWORDS = [
    "sentence", "headline", "title", "text", "description",
    "event", "disruption", "summary", "selftext",
]

# ── Known entities for NER augmentation ──────────────────────────────────────
KNOWN_COMPANIES = [
    # Semiconductors & Electronics
    "TSMC", "Samsung", "Intel", "Nvidia", "AMD", "Qualcomm", "Broadcom",
    "Texas Instruments", "Micron", "SK Hynix", "ASML", "GlobalFoundries",
    "MediaTek", "Infineon", "NXP", "STMicroelectronics", "Foxconn",
    "Huawei", "Xiaomi", "SMIC",
    # Tech Giants
    "Apple", "Microsoft", "Google", "Alphabet", "Meta", "Amazon", "IBM",
    "Oracle", "SAP", "Cisco", "Dell", "HP", "Lenovo", "Sony",
    # Automotive
    "Toyota", "Volkswagen", "Tesla", "Ford", "GM", "General Motors",
    "BMW", "Mercedes-Benz", "Hyundai", "Honda", "Stellantis", "BYD",
    "Rivian", "Lucid", "NIO", "CATL", "Panasonic",
    # Aerospace & Defense
    "Boeing", "Airbus", "Lockheed Martin", "Raytheon", "Northrop Grumman",
    "General Electric", "GE", "Rolls-Royce", "BAE Systems",
    # Retail & Consumer
    "Walmart", "Costco", "Target", "Nike", "Adidas", "LVMH",
    "Procter & Gamble", "Unilever", "Nestle", "PepsiCo", "Coca-Cola",
    # Energy & Materials
    "ExxonMobil", "Shell", "BP", "Chevron", "Saudi Aramco",
    "Rio Tinto", "BHP", "Vale", "Glencore", "ArcelorMittal",
    # Pharma & Healthcare
    "Pfizer", "Johnson & Johnson", "Roche", "Novartis", "Merck",
    "AstraZeneca", "Moderna", "Abbott", "Medtronic",
    # Logistics & Shipping
    "Maersk", "MSC", "CMA CGM", "COSCO", "FedEx", "UPS", "DHL",
    "Hapag-Lloyd", "Evergreen", "ZIM", "XPO Logistics",
    # Agriculture & Food
    "Cargill", "ADM", "Bunge", "John Deere", "BASF", "Syngenta",
    "Tyson Foods", "JBS",
]
KNOWN_REGIONS = [
    # Asia
    "Taiwan", "China", "Shenzhen", "Shanghai", "Beijing", "Guangzhou",
    "Hong Kong", "Japan", "Tokyo", "South Korea", "Seoul", "Singapore",
    "Vietnam", "Hanoi", "Ho Chi Minh", "India", "Mumbai", "Chennai",
    "Bangladesh", "Dhaka", "Thailand", "Bangkok", "Malaysia", "Indonesia",
    "Philippines", "Myanmar",
    # Europe
    "Rotterdam", "Hamburg", "Antwerp", "Germany", "France", "UK",
    "London", "Berlin", "Italy", "Spain", "Poland", "Turkey", "Istanbul",
    "Netherlands", "Belgium", "Switzerland", "Sweden",
    # Americas
    "Los Angeles", "Long Beach", "New York", "Houston", "Chicago",
    "Savannah", "Mexico", "Brazil", "Sao Paulo", "Panama Canal", "Panama",
    # Middle East & Africa
    "Suez", "Suez Canal", "Dubai", "Saudi Arabia", "Iran",
    "Strait of Hormuz", "Red Sea", "Cape Town", "South Africa", "Egypt",
    # Oceania
    "Australia", "Melbourne", "Sydney",
]

# ── Geo coordinates for the risk map ─────────────────────────────────────────
GEO_COORDS = {
    # Asia
    "Taiwan": (23.6978, 120.9605),
    "China": (35.8617, 104.1954),
    "Shenzhen": (22.5431, 114.0579),
    "Shanghai": (31.2304, 121.4737),
    "Beijing": (39.9042, 116.4074),
    "Hong Kong": (22.3193, 114.1694),
    "Japan": (36.2048, 138.2529),
    "Tokyo": (35.6762, 139.6503),
    "South Korea": (35.9078, 127.7669),
    "Seoul": (37.5665, 126.9780),
    "Singapore": (1.3521, 103.8198),
    "Vietnam": (14.0583, 108.2772),
    "India": (20.5937, 78.9629),
    "Mumbai": (19.0760, 72.8777),
    "Bangladesh": (23.6850, 90.3563),
    "Thailand": (15.8700, 100.9925),
    "Malaysia": (4.2105, 101.9758),
    "Indonesia": (-0.7893, 113.9213),
    "Philippines": (12.8797, 121.7740),
    # Europe
    "Rotterdam": (51.9225, 4.4792),
    "Hamburg": (53.5511, 9.9937),
    "Antwerp": (51.2194, 4.4025),
    "Germany": (51.1657, 10.4515),
    "France": (46.6034, 1.8883),
    "UK": (55.3781, -3.4360),
    "London": (51.5074, -0.1278),
    "Italy": (41.8719, 12.5674),
    "Turkey": (38.9637, 35.2433),
    "Istanbul": (41.0082, 28.9784),
    # Americas
    "Los Angeles": (34.0522, -118.2437),
    "Long Beach": (33.7701, -118.1937),
    "New York": (40.7128, -74.0060),
    "Houston": (29.7604, -95.3698),
    "Mexico": (23.6345, -102.5528),
    "Brazil": (-14.2350, -51.9253),
    "Panama": (8.5380, -80.7821),
    # Middle East & Africa
    "Suez": (29.9668, 32.5498),
    "Suez Canal": (30.4574, 32.3499),
    "Dubai": (25.2048, 55.2708),
    "Saudi Arabia": (23.8859, 45.0792),
    "Iran": (32.4279, 53.6880),
    "Red Sea": (20.2801, 38.5126),
    "South Africa": (-30.5595, 22.9375),
    "Egypt": (26.8206, 30.8025),
    # Oceania
    "Australia": (-25.2744, 133.7751),
}
