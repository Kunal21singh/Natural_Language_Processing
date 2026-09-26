from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text using VADER sentiment analysis.

    Parameters:
    text (str): The input text to analyze.

    Returns:
    dict: A dictionary containing the sentiment scores.
    """
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

print(analyze_sentiment("I don't love this product! It's can be improved."))