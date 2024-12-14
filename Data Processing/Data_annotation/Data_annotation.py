import openai
from typing import Dict, List
import json
import pandas as pd
import os
from datetime import datetime

class PostAnnotator:
    def __init__(self, api_key: str, output_dir: str = "annotations"):
        openai.api_key = api_key
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def annotate_post(self, post: str) -> Dict[str, str]:
        """Annotate a single post for criticism and vagueness using GPT-4."""
        criticism_prompt = '''In the context of a Hong Kong political forum, determine if this post contains criticism of the government, government leaders, or related policies. Label as "Criticism" if any discontent, dissatisfaction, or questioning of government actions, policies, or officials is present, even if implied indirectly or sarcastically. Label as "Non-Criticism" if it does not criticize the government or only discusses general topics.

Post: {post}
Label:'''
        
        vagueness_prompt = '''In the context of a Hong Kong political forum, identify if this post uses vague or indirect expressions to imply criticism of the government without directly stating it. Label as "Vague" if the language is ambiguous, sarcastic, or indirect. Label as "Non-Vague" if the expression is clear and straightforward.

Post: {post}
Label:'''
        
        criticism_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": criticism_prompt.format(post=post)}]
        )
        criticism_label = criticism_response.choices[0].message.content.strip()
        
        vagueness_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": vagueness_prompt.format(post=post)}]
        )
        vagueness_label = vagueness_response.choices[0].message.content.strip()
        
        return {
            "post": post,
            "criticism": criticism_label,
            "vagueness": vagueness_label
        }

    def batch_annotate(self, posts: List[str], batch_size: int = 100, save_interval: int = 100) -> None:
        """Annotate multiple posts in batches and save results periodically."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        
        try:
            for i in range(0, len(posts), batch_size):
                batch = posts[i:i + batch_size]
                batch_results = [self.annotate_post(post) for post in batch]
                results.extend(batch_results)
                
                # Save results periodically
                if (i + batch_size) % save_interval == 0:
                    self._save_results(results, f"annotations_batch_{i}_{timestamp}")
                    print(f"Saved batch {i} results")
                
        except Exception as e:
            # Save results in case of error
            self._save_results(results, f"annotations_error_{timestamp}")
            print(f"Error occurred: {str(e)}")
            raise e
        
        # Save final results
        self._save_results(results, f"annotations_final_{timestamp}")
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, f"annotations_final_{timestamp}.csv"), 
                 index=False, encoding='utf-8')

    def _save_results(self, results: List[Dict], filename: str) -> None:
        """Save results to JSON file."""
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    annotator = PostAnnotator("api-key-here")
    
    # Load your posts from a file
    with open('posts.txt', 'r', encoding='utf-8') as f:
        posts = f.readlines()
    
    # Start annotation
    annotator.batch_annotate(posts, batch_size=100, save_interval=100)