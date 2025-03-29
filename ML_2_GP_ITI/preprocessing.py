import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing HTML tags,
    punctuation, numbers, and extra whitespace.
    
    Args:
        text (str): Raw text to be preprocessed
        
    Returns:
        str: Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def tokenize_text(text):
    """
    Tokenize text into individual words.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of tokens
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of tokens with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their base form.
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of lemmatized tokens
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(token) for token in tokens]

def tokens_to_text(tokens):
    """
    Convert tokens back to text.
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        str: Space-joined text
    """
    return ' '.join(tokens)

def remove_first_n_words(text, n=2):
    """
    Remove the first n words from text.
    
    Args:
        text (str): Input text
        n (int): Number of words to remove from beginning
        
    Returns:
        str: Text with first n words removed
    """
    return ' '.join(text.split()[n:])

def full_preprocessing_pipeline(text):
    """
    Apply the complete preprocessing pipeline to a text.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Fully processed text
    """
    processed_text = preprocess_text(text)
    tokens = tokenize_text(processed_text)
    tokens_without_stopwords = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens_without_stopwords)
    clean_text = tokens_to_text(lemmatized_tokens)
    clean_text = remove_first_n_words(clean_text, 2)
    return clean_text