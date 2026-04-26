import requests
import re
import joblib
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ACCESS_TOKEN = "YOUR_THREADS_ACCESS_TOKEN"
API_URL = "https://graph.threads.net/v1/posts/search"
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

CITY_LIST = [
    "Астана", "Алматы", "Шымкент", "Караганда",
    "Актобе", "Тараз", "Павлодар", "Усть-Каменогорск"
]

train_texts = [
    "дороги ужасные","ямы везде","ехать невозможно","асфальт развалился","дороги разбиты",
    "воду отключили","нет воды снова","вода пропала","воды нет","перебои с водой",
    "свет мигает","нет электричества","снова отключили свет","перебои со светом","электричество пропало",
    "мусор не вывозят","контейнеры переполнены","воняет во дворе","мусор лежит неделями","двор грязный",
    "ограбление рядом","кража на районе","опасно вечером","страшно ходить ночью","полиция была снова",
    "фонари не работают","темно во дворе","освещения нет","улица без света","ничего не видно",
    "вода ржавая","вода грязная","странный запах воды","вода мутная","из крана плохо",
    "дороги плохие","дороги как после войны","разбитая дорога","асфальта нет","грунтовка вместо дороги",
    "район тихий","ничего не происходит","просто новости читаю","обычный день","всё нормально",
    "перебои постоянно","каждый день проблемы","ничего не работает","система ломается","всё нестабильно",
    "район опасный","много краж","частые преступления","страшный район","небезопасно"
]

train_labels = [
    "roads","roads","roads","roads","roads",
    "water","water","water","water","water",
    "electricity","electricity","electricity","electricity","electricity",
    "garbage","garbage","garbage","garbage","garbage",
    "crime","crime","crime","crime","crime",
    "electricity","electricity","electricity","electricity","electricity",
    "water","water","water","water","water",
    "roads","roads","roads","roads","roads",
    "other","other","other","other","other",
    "electricity","electricity","electricity","electricity","electricity",
    "crime","crime","crime","crime","crime"
]

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)

X_train = vectorizer.fit_transform(train_texts)
model.fit(X_train, train_labels)

joblib.dump(vectorizer, "threads_vectorizer.pkl")
joblib.dump(model, "threads_model.pkl")

def fetch_posts(query="Казахстан", limit=100):
    r = requests.get(API_URL, headers=HEADERS, params={"q": query, "limit": limit})
    if r.status_code != 200:
        return []
    data = r.json()
    return [p["text"] for p in data.get("posts", [])]

def extract_location(text):
    t = text.lower()
    for city in CITY_LIST:
        if city.lower() in t:
            return city
    return "Unknown"

def classify(text):
    x = vectorizer.transform([normalize(text)])
    return model.predict(x)[0]

def analyze(posts):
    res = defaultdict(list)
    for p in posts:
        loc = extract_location(p)
        cat = classify(p)
        res[loc].append(cat)
    return res

def summarize(data):
    return {k: dict(Counter(v)) for k, v in data.items()}

if __name__ == "__main__":
    posts = fetch_posts("Казахстан", 100)
    results = analyze(posts)
    summary = summarize(results)
    for loc, probs in summary.items():
        print(loc)
        for k, v in probs.items():
            print(k, v)