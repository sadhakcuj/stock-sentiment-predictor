import requests
import pandas as pd
from datetime import datetime, timedelta

print("=" * 60)
print("STOCK SENTIMENT PREDICTOR - NEWS SCRAPER")
print("=" * 60)

# Finnhub API Configuration
FINNHUB_API_KEY = "d88mp51r01qq4343qe6gd88mp51r01qq4343qe70"

def scrape_stock_news(ticker, days=7):
    """
    Fetch recent stock news using Finnhub API
    """

    print(f"\n📰 Scraping news for {ticker}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"

    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }

    articles = []

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:

            data = response.json()

            if isinstance(data, list):
                print(f"✅ Found {len(data)} articles")

                for item in data:
                    articles.append({
                        "title": item.get("headline", ""),
                        "description": item.get("summary", ""),
                        "publishedAt": datetime.fromtimestamp(
                            item.get("datetime", 0)
                        ).strftime("%Y-%m-%d %H:%M:%S"),

                        "url": item.get("url", ""),

                        "source": {
                            "name": item.get("source", "Unknown")
                        }
                    })
            else:
                print("⚠️ Unexpected API response format")

        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"⚠️ Error fetching news: {e}")

    return articles

def analyze_sentiment_simple(text):
    """
    Simple sentiment analysis using keyword matching
    """

    positive_words = [
        'gain', 'surge', 'bull', 'growth', 'profit', 'earnings',
        'strong', 'record', 'rally', 'rise', 'soar', 'optimism',
        'bullish', 'upgrade', 'beat', 'achieve', 'exceed',
        'momentum', 'outperform', 'excellent'
    ]

    negative_words = [
        'loss', 'fall', 'drop', 'crash', 'decline', 'bear',
        'weak', 'poor', 'fail', 'miss', 'risk', 'concern',
        'bearish', 'downgrade', 'plunge', 'slump', 'struggle',
        'challenge', 'underperform', 'warning', 'sell'
    ]

    text_lower = text.lower()

    positive_count = sum(
        1 for word in positive_words if word in text_lower
    )

    negative_count = sum(
        1 for word in negative_words if word in text_lower
    )

    if positive_count > negative_count:
        return 'positive', positive_count / (
            positive_count + negative_count + 1
        )

    elif negative_count > positive_count:
        return 'negative', negative_count / (
            positive_count + negative_count + 1
        )

    else:
        return 'neutral', 0.5

def process_news(articles, ticker):

    print(f"\n💭 Analyzing sentiment for {len(articles)} articles...")

    processed = []

    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }

    for article in articles:

        title = article.get('title', '')
        description = article.get('description', '')

        text = f"{title} {description}"

        sentiment, confidence = analyze_sentiment_simple(text)

        sentiment_counts[sentiment] += 1

        processed.append({
            'title': title,
            'source': article.get('source', {}).get('name', 'Unknown'),
            'date': article.get('publishedAt', ''),
            'url': article.get('url', ''),
            'sentiment': sentiment,
            'confidence': confidence
        })

    print(f"✅ Sentiment Analysis Complete:")
    print(f"   Positive: {sentiment_counts['positive']}")
    print(f"   Negative: {sentiment_counts['negative']}")
    print(f"   Neutral:  {sentiment_counts['neutral']}")

    return processed, sentiment_counts

def save_news_data(articles, ticker):

    df = pd.DataFrame(articles)

    filename = f'news_{ticker}.csv'

    df.to_csv(filename, index=False)

    print(f"\n💾 Saved {len(df)} articles to {filename}")

    return filename

# MAIN EXECUTION

print("\n📊 Stock Sentiment Analysis Tool")
print("=" * 60)

ticker = "AAPL"

articles = scrape_stock_news(ticker, days=7)

if len(articles) == 0:
    print("❌ No articles found")
    exit()

processed_articles, sentiment_counts = process_news(
    articles,
    ticker
)

save_news_data(processed_articles, ticker)

total = sum(sentiment_counts.values())

sentiment_score = (
    (sentiment_counts['positive'] * 1.0) +
    (sentiment_counts['neutral'] * 0.5) +
    (sentiment_counts['negative'] * 0.0)
) / total if total > 0 else 0.5

print(f"\n📈 Overall Sentiment Score: {sentiment_score:.2f}/1.0")

if sentiment_score > 0.6:
    print("   ⬆️  BULLISH - More positive news")

elif sentiment_score < 0.4:
    print("   ⬇️  BEARISH - More negative news")

else:
    print("   ➡️  NEUTRAL - Mixed sentiment")

print("\n" + "=" * 60)
print("✨ News scraping complete!")
print("=" * 60)