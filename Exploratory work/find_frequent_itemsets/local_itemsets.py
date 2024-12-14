import json
from collections import defaultdict

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, mode='r', encoding='utf-8') as file:
        return json.load(file)

def filter_single_characters(words):
    """Filter out single-character words."""
    return [word for word in words if len(word) > 1]

def find_co_occurring_words(data, target_word, excluded_words=None):
    """Find words that co-occur with the target word across posts."""
    co_occurrence_counts = defaultdict(int)
    
    for post in data:
        # Assuming post is a list of words directly
        if target_word in post:
            for word in post:
                # Check if word is not single-character, not the target word, and not in excluded words
                if len(word) > 1 and word != target_word and (not excluded_words or word not in excluded_words):
                    co_occurrence_counts[word] += 1
    
    return co_occurrence_counts

def get_top_co_occurring_word(co_occurrence_counts):
    """Return the most frequent co-occurring word from a count dictionary."""
    if co_occurrence_counts:
        return max(co_occurrence_counts, key=co_occurrence_counts.get)
    return None

def find_frequent_set(file_path, initial_word, max_set_size=6):
    """Find a set of six frequent co-occurring words with the initial word."""
    data = load_data(file_path)
    current_set = [initial_word]
    print(f"Starting with initial word: {initial_word}")

    for _ in range(1, max_set_size):
        co_occurrence_counts = find_co_occurring_words(data, target_word=current_set[-1], excluded_words=current_set)
        top_word = get_top_co_occurring_word(co_occurrence_counts)
        
        if top_word:
            current_set.append(top_word)
            print(f"Added '{top_word}' to the set. Current set: {current_set}")
        else:
            print("No more frequent co-occurring words found.")
            break

    return current_set

def count_posts_with_set(data, word_set):
    """Count the number of posts containing all words in the given set."""
    count = 0
    for post in data:
        if all(word in post for word in word_set):
            count += 1
    return count

def process_all_keywords(file_path, keywords, max_set_size=6):
    """Process each keyword and find the frequent set with their occurrences."""
    data = load_data(file_path)
    for keyword in keywords:
        print(f"\nProcessing keyword: {keyword}")
        frequent_set = find_frequent_set(file_path, initial_word=keyword, max_set_size=max_set_size)
        post_count = count_posts_with_set(data, frequent_set)
        print(f"The frequent set for '{keyword}': {frequent_set}")
        print(f"Number of posts containing all words in the set '{frequent_set}': {post_count}")

if __name__ == "__main__":
    # File path to the dataset
    file_path = r"D:\Download\cleaned_data_final.json"

    # List of keywords to process
    keywords = ["民主", "公义", "公平", "投票", "利益", "保衛", "普選",
                "示威者", "示威", "抗争", "文宣", "打壓", "对抗", "落街",
                "政府", "特首", "黑社会", "滲透", "極權"]

    # Process all keywords and find frequent sets
    process_all_keywords(file_path, keywords, max_set_size=6)
