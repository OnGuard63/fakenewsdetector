
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

app = Flask(__name__)

# Function to download NLTK resources (if not already available)
def download_nltk_resources():
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        nltk.download('wordnet')

# Custom stopwords list (fallback if NLTK download fails or if you prefer not to use NLTK)
CUSTOM_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

# Function to extract keywords from the user's input
def extract_keywords(text, use_nltk=True):
    if use_nltk:
        download_nltk_resources()
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = CUSTOM_STOPWORDS

    lemmatizer = WordNetLemmatizer()
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return filtered_words

# Function to fetch recent headlines from a given news site, with retries
def fetch_news_headlines(url, site_name, tags, headers=None, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = soup.find_all(tags)
            return [(headline.get_text(strip=True).lower(), site_name) for headline in headlines if headline.get_text(strip=True)]
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return []

# Function to check if most of the keywords from user input appear in a headline
def check_headline_for_keywords(user_keywords, headlines, threshold=0.3):
    if not headlines:
        return []
    user_input_str = ' '.join(user_keywords)
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [user_input_str] + [headline[0] for headline in headlines]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    matched_headlines = []
    for i, score in enumerate(cosine_similarities):
        if score > threshold:
            matched_headlines.append(f"Headline match found from {headlines[i][1]}: {headlines[i][0]}")

    return matched_headlines

# Flask route to render the form and process the user's input
@app.route("/", methods=["GET", "POST"])
def index():
    matched_headlines = []
    if request.method == "POST":
        user_input = request.form["user_input"]
        user_keywords = extract_keywords(user_input, use_nltk=True)

        news_sources = {
            "BBC": {"url": "https://www.bbc.co.uk", "tags": ['h3', 'h2']},
            "The Guardian": {"url": "https://www.theguardian.com/observer", "tags": ['h3']},
            "CNN": {"url": "https://www.cnn.com", "tags": ['h3', 'span']},
            "Al Jazeera": {"url": "https://www.aljazeera.com", "tags": ['h3', 'h2', 'h1']},
            "Associated Press": {"url": "https://www.apnews.com", "tags": ['h3', 'h2', 'a']},
            "ABC News": {"url": "https://abcnews.go.com", "tags": ['h3', 'h2']},
            "New York Times": {"url": "https://www.nytimes.com", "tags": ['h3', 'h2']},
        }

        all_headlines = []
        for site_name, info in news_sources.items():
            headlines = fetch_news_headlines(info['url'], site_name, info['tags'])
            all_headlines.extend(headlines)

        matched_headlines = check_headline_for_keywords(user_keywords, all_headlines)

    return render_template("index.html", matched_headlines=matched_headlines)

if __name__ == "__main__":
    app.run(debug=True)
