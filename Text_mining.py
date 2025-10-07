import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets"""
    nltk_downloads = [
        'punkt',
        'stopwords', 
        'wordnet',
        'omw-1.4',
        'punkt_tab'
    ]
    
    for resource in nltk_downloads:
        try:
            nltk.data.find(resource)
            print(f" {resource} already available")
        except LookupError:
            print(f" Downloading {resource}...")
            nltk.download(resource, quiet=True)
            print(f" {resource} downloaded successfully")

print("Checking NLTK resources...")
download_nltk_data()
print("All NLTK resources ready!\n")


def load_data():
    try:
        df = pd.read_csv('amazon_product_reviews.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('Reviews.csv')
        except FileNotFoundError:
            try:
                df = pd.read_csv('amazon_reviews.csv')
            except FileNotFoundError:
                print("Please download the dataset from Kaggle and update the file path")
                print("Dataset: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews")
                return None
    
    print(f"Dataset shape: {df.shape}")
    print("\nDataset columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    return df

# Load the data
df = load_data()

if df is not None:
    # Check if required columns exist
    required_columns = ['Score', 'Text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
    else:
        print("\n All required columns found!")
        print(f"Using 'Text' for reviews and 'Score' for ratings")
        
        # Keep only the columns we need and drop missing values
        df = df[['Score', 'Text', 'Summary', 'ProductId', 'UserId']].copy()
        df = df.dropna(subset=['Text', 'Score'])
        
        print(f"Data after cleaning: {df.shape}")
        
        # Display rating distribution
        print("\nScore distribution:")
        score_dist = df['Score'].value_counts().sort_index()
        print(score_dist)

        class TextPreprocessor:
            def __init__(self):
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                # Add custom stopwords for product reviews
                self.stop_words.update(['br', 'href', 'http', 'https', 'com', 'www', 'amazon', 'product', 'item'])
                
            def clean_text(self, text):
                if pd.isna(text):
                    return ""
                
                # Convert to string and lowercase
                text = str(text).lower()
                
                # Remove URLs
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'www\S+', '', text)
                
                # Remove HTML tags
                text = re.sub(r'<.*?>', '', text)
                
                # Remove punctuation and numbers (keep words only)
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords and lemmatize
                cleaned_tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token not in self.stop_words and len(token) > 2
                ]
                
                return ' '.join(cleaned_tokens)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        print("Cleaning text data... (This may take a while for large datasets)")
        
        # Sample the data if it's too large for demonstration
        original_size = len(df)
        if len(df) > 15000:
            df = df.sample(15000, random_state=42)
            print(f"Sampled from {original_size} to {len(df)} records for faster processing")
        
        # Clean the text
        df['cleaned_text'] = df['Text'].apply(preprocessor.clean_text)
        
        # Also clean summary for additional analysis
        df['cleaned_summary'] = df['Summary'].apply(lambda x: preprocessor.clean_text(x) if pd.notna(x) else "")
        
        # Remove empty reviews after cleaning
        df = df[df['cleaned_text'].str.len() > 10]  # At least 10 characters after cleaning
        
        # Create sentiment labels
        def create_sentiment(score):
            if score >= 4:
                return 'positive'
            elif score == 3:
                return 'neutral'
            else:
                return 'negative'
        
        df['sentiment'] = df['Score'].apply(create_sentiment)
        
        print(f"\nFinal dataset size: {df.shape}")
        print("\nSentiment distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        print(sentiment_dist)

        def plot_enhanced_analysis(df):
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
            axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                         autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            
            # 2. Score distribution
            score_counts = df['Score'].value_counts().sort_index()
            axes[0,1].bar(score_counts.index, score_counts.values, 
                         color=['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71'])
            axes[0,1].set_title('Score Distribution', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Score (1-5)')
            axes[0,1].set_ylabel('Number of Reviews')
            
            # 3. Review length distribution
            df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
            axes[0,2].hist(df['review_length'], bins=50, edgecolor='black', alpha=0.7)
            axes[0,2].set_title('Distribution of Review Length', fontsize=14, fontweight='bold')
            axes[0,2].set_xlabel('Number of Words')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].axvline(df['review_length'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {df["review_length"].mean():.1f}')
            axes[0,2].legend()
            
            # 4. Average score by sentiment
            sentiment_score = df.groupby('sentiment')['Score'].mean()
            axes[1,0].bar(sentiment_score.index, sentiment_score.values, 
                         color=['#e74c3c', '#f39c12', '#2ecc71'])
            axes[1,0].set_title('Average Score by Sentiment', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('Average Score')
            
            # 5. Top products by review count
            top_products = df['ProductId'].value_counts().head(10)
            axes[1,1].bar(range(len(top_products)), top_products.values)
            axes[1,1].set_title('Top 10 Products by Review Count', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('Product Index')
            axes[1,1].set_ylabel('Number of Reviews')
            
            # 6. Helpfulness ratio (if available)
            if 'HelpfulnessNumerator' in df.columns and 'HelpfulnessDenominator' in df.columns:
                df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
                df['helpfulness_ratio'] = df['helpfulness_ratio'].replace([np.inf, -np.inf], np.nan)
                helpfulness_by_score = df.groupby('Score')['helpfulness_ratio'].mean()
                axes[1,2].bar(helpfulness_by_score.index, helpfulness_by_score.values)
                axes[1,2].set_title('Average Helpfulness by Score', fontsize=14, fontweight='bold')
                axes[1,2].set_xlabel('Score')
                axes[1,2].set_ylabel('Helpfulness Ratio')
            else:
                # Top reviewers instead
                top_reviewers = df['UserId'].value_counts().head(10)
                axes[1,2].bar(range(len(top_reviewers)), top_reviewers.values)
                axes[1,2].set_title('Top 10 Reviewers by Activity', fontsize=14, fontweight='bold')
                axes[1,2].set_xlabel('User Index')
                axes[1,2].set_ylabel('Number of Reviews')
            
            plt.tight_layout()
            plt.show()
            
            # Print some statistics
            print(f"\n Dataset Statistics:")
            print(f"   - Average review length: {df['review_length'].mean():.1f} words")
            print(f"   - Median review length: {df['review_length'].median():.1f} words")
            print(f"   - Most common score: {df['Score'].mode().iloc[0]}")
            print(f"   - Number of unique products: {df['ProductId'].nunique()}")
            print(f"   - Number of unique users: {df['UserId'].nunique()}")
        
        plot_enhanced_analysis(df)

        def generate_enhanced_wordclouds(df):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Positive reviews
            positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'])
            if positive_text:
                wordcloud = WordCloud(width=600, height=300, background_color='white',
                                    max_words=100, colormap='viridis').generate(positive_text)
                axes[0,0].imshow(wordcloud, interpolation='bilinear')
                axes[0,0].set_title(' Positive Reviews - Word Cloud', fontsize=14, fontweight='bold')
                axes[0,0].axis('off')
            
            # Negative reviews
            negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'])
            if negative_text:
                wordcloud = WordCloud(width=600, height=300, background_color='white',
                                    max_words=100, colormap='Reds').generate(negative_text)
                axes[0,1].imshow(wordcloud, interpolation='bilinear')
                axes[0,1].set_title(' Negative Reviews - Word Cloud', fontsize=14, fontweight='bold')
                axes[0,1].axis('off')
            
            # Neutral reviews
            neutral_text = ' '.join(df[df['sentiment'] == 'neutral']['cleaned_text'])
            if neutral_text:
                wordcloud = WordCloud(width=600, height=300, background_color='white',
                                    max_words=100, colormap='Oranges').generate(neutral_text)
                axes[1,0].imshow(wordcloud, interpolation='bilinear')
                axes[1,0].set_title(' Neutral Reviews - Word Cloud', fontsize=14, fontweight='bold')
                axes[1,0].axis('off')
            
            # Summary word cloud
            summary_text = ' '.join(df['cleaned_summary'])
            if summary_text:
                wordcloud = WordCloud(width=600, height=300, background_color='black',
                                    max_words=100, colormap='plasma').generate(summary_text)
                axes[1,1].imshow(wordcloud, interpolation='bilinear')
                axes[1,1].set_title(' Review Summaries - Word Cloud', fontsize=14, fontweight='bold')
                axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        generate_enhanced_wordclouds(df)

        print("Creating features...")
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=5,           # Ignore terms that appear in less than 5 documents
            max_df=0.8          # Ignore terms that appear in more than 80% of documents
        )
        
        X = tfidf.fit_transform(df['cleaned_text'])
        y = df['sentiment']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(tfidf.get_feature_names_out())}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")

        def train_and_evaluate_models(X_train, X_test, y_train, y_test):
            models = {
                'Naive Bayes': MultinomialNB(),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            results = {}
            
            for name, model in models.items():
                print(f"\n Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f" {name} Accuracy: {accuracy:.4f}")
                print(f" Classification Report for {name}:")
                print(classification_report(y_test, y_pred))
                
                # Plot confusion matrix
                plt.figure(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=np.unique(y), 
                           yticklabels=np.unique(y))
                plt.title(f'Confusion Matrix - {name}', fontweight='bold')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.show()
            
            return results
        
        print(" Training sentiment analysis models...")
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        def perform_topic_modeling(df, num_topics=5):
            print(f"\n Performing Topic Modeling with {num_topics} topics...")
            
            # Use CountVectorizer for LDA
            count_vectorizer = CountVectorizer(
                max_features=2000,
                min_df=10,
                max_df=0.8,
                stop_words='english'
            )
            
            count_data = count_vectorizer.fit_transform(df['cleaned_text'])
            
            # Create and fit LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            
            lda.fit(count_data)
            
            # Display topics
            feature_names = count_vectorizer.get_feature_names_out()
            
            print("\n Discovered Topics:")
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-15:][::-1]  # Top 15 words
                top_words = [feature_names[i] for i in top_words_idx]
                print(f"   Topic #{topic_idx + 1}: {', '.join(top_words[:10])}...")
            
            return lda, count_vectorizer
        
        lda_model, vectorizer = perform_topic_modeling(df)

        def generate_comprehensive_insights(df, model_results, lda_model):
            print("\n" + "="*70)
            print("KEY INSIGHTS AND BUSINESS RECOMMENDATIONS")
            print("="*70)
            
            # Basic statistics
            total_reviews = len(df)
            positive_pct = (df['sentiment'] == 'positive').sum() / total_reviews * 100
            negative_pct = (df['sentiment'] == 'negative').sum() / total_reviews * 100
            neutral_pct = (df['sentiment'] == 'neutral').sum() / total_reviews * 100
            
            print(f"\n1. SENTIMENT OVERVIEW:")
            print(f"   - Total Reviews Analyzed: {total_reviews:,}")
            print(f"   - Positive Sentiment: {positive_pct:.1f}%")
            print(f"   - Neutral Sentiment: {neutral_pct:.1f}%")
            print(f"   - Negative Sentiment: {negative_pct:.1f}%")
            
            # Best performing model
            best_model_name, best_model_info = max(model_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"   - Best Prediction Model: {best_model_name} ({best_model_info['accuracy']:.1%} accuracy)")
            
            # Most common words analysis
            def get_top_words_by_sentiment(sentiment, top_n=8):
                text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                words = text.split()
                return pd.Series(words).value_counts().head(top_n)
            
            negative_top_words = get_top_words_by_sentiment('negative')
            positive_top_words = get_top_words_by_sentiment('positive')
            
            print(f"\n2. CUSTOMER PAIN POINTS (Top words in negative reviews):")
            for word, freq in negative_top_words.items():
                print(f"   - '{word}' (appears {freq} times)")
            
            print(f"\n3. CUSTOMER SATISFACTION DRIVERS (Top words in positive reviews):")
            for word, freq in positive_top_words.items():
                print(f"   - '{word}' (appears {freq} times)")
            
            # Product analysis
            top_products = df['ProductId'].value_counts().head(5)
            print(f"\n4. TOP PRODUCTS BY REVIEW VOLUME:")
            for product, count in top_products.items():
                product_sentiment = df[df['ProductId'] == product]['sentiment'].value_counts(normalize=True)
                positive_rate = product_sentiment.get('positive', 0) * 100
                print(f"   - Product {product[:15]}...: {count} reviews, {positive_rate:.1f}% positive")
            
            print(f"\n5. BUSINESS RECOMMENDATIONS:")
            if negative_pct > 20:
                print(f"   - HIGH PRIORITY: {negative_pct:.1f}% negative reviews need immediate attention")
            else:
                print(f"   - Good: Only {negative_pct:.1f}% negative reviews")
                
            if positive_pct > 70:
                print(f"   - Excellent: {positive_pct:.1f}% customer satisfaction rate")
            else:
                print(f"   - Opportunity: Improve from {positive_pct:.1f}% to >70% positive reviews")
            
            print(f"   - Focus on addressing: {', '.join(negative_top_words.index[:3])}")
            print(f"   - Highlight in marketing: {', '.join(positive_top_words.index[:3])}")
            print(f"   - Use topic modeling insights for product development")
            print(f"   - Train customer support on common complaint patterns")
        
        generate_comprehensive_insights(df, model_results, lda_model)

        def predict_sentiment(new_review, best_model, tfidf_vectorizer, preprocessor):
            # Preprocess the new review
            cleaned_review = preprocessor.clean_text(new_review)
            
            # Transform using the fitted TF-IDF vectorizer
            review_tfidf = tfidf_vectorizer.transform([cleaned_review])
            
            # Predict sentiment
            prediction = best_model.predict(review_tfidf)[0]
            probabilities = best_model.predict_proba(review_tfidf)[0]
            
            # Create probability dictionary
            prob_dict = {best_model.classes_[i]: prob for i, prob in enumerate(probabilities)}
            
            return prediction, prob_dict, cleaned_review
        
        # Test with sample reviews
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS")
        print("="*70)
        
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = model_results[best_model_name]['model']
        
        sample_reviews = [
            "This product is absolutely amazing! It works perfectly and exceeded all my expectations. The quality is outstanding.",
            "The item was okay, nothing special but it gets the job done. Could be better but not terrible.",
            "Terrible quality. Broke after one week of use. Complete waste of money. Would not recommend to anyone.",
            "I love this! It's exactly what I needed and works great. Very happy with my purchase.",
            "Disappointed with the product. It didn't meet my expectations and the quality is poor for the price."
        ]
        
        for i, review in enumerate(sample_reviews, 1):
            sentiment, probabilities, cleaned = predict_sentiment(review, best_model, tfidf, preprocessor)
            print(f"\n Sample Review {i}:")
            print(f"   Original: '{review}'")
            print(f"   Cleaned: '{cleaned}'")
            print(f"   Predicted Sentiment: {sentiment.upper()}")
            print(f"   Confidence: {max(probabilities.values()):.1%}")
            print(f"   Probabilities: {probabilities}")
        
        print("\n" + "="*70)
        print("TEXT MINING PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)

else:
    print("Dataset not loaded. Please check the file path.")