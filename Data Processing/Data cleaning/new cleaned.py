import json
import re

def clean_segmented_text(segmented_texts):
    cleaned_texts = []

    # Define regular expressions for matching unwanted characters like punctuation
    unwanted_chars_re = re.compile(r'[，。、？！：；“”‘’《》【】()（）…—]', re.UNICODE)
    extra_whitespace_re = re.compile(r'\s+')

    for text in segmented_texts:
        cleaned_post = []
        for word in text:
            # Remove unwanted characters
            word = re.sub(unwanted_chars_re, '', word)
            # Remove extra spaces
            word = re.sub(extra_whitespace_re, '', word)
            # Keep words that are not empty
            if word:
                cleaned_post.append(word)
        
        cleaned_texts.append(cleaned_post)

    return cleaned_texts

def process_segmented_data(input_file_path, output_file_path):
    try:
        # Read JSON file content
        with open(input_file_path, mode='r', encoding='utf-8') as json_file:
            segmented_data = json.load(json_file)

        # Clean the data
        cleaned_data = clean_segmented_text(segmented_data)

        # Save the cleaned data to a new JSON file
        with open(output_file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(cleaned_data, output_file, ensure_ascii=False, indent=4)

        print(f"Data cleaning completed. Cleaned data saved to {output_file_path}")

    except FileNotFoundError:
        print("Error: JSON file not found. Please check the file path.")
    except json.JSONDecodeError:
        print("Error: JSON decoding failed. Please check if the file content is a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Input and output file paths
    input_file_path = r"D:\Download\extracted_data.json"
    output_file_path = r"D:\Download\cleaned_data_final_final.json"

    # Process the segmented data
    process_segmented_data(input_file_path, output_file_path)
