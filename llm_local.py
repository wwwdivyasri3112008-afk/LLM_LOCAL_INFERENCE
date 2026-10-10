import os
import pandas as pd
import ollama

# Ensure the dataset directory exists as per folder structure requirements
os.makedirs("dataset", exist_ok=True)

# Path to the prompts file
prompts_path = "dataset/prompts.csv"

# Optional: Create a sample prompts.csv if it doesn't exist yet for testing
if not os.path.exists(prompts_path):
    sample_data = {
        "prompt_id": [1, 2, 3, 4, 5],
        "domain": ["education", "healthcare", "coding", "travel", "business"],
        "prompt": [
            "Explain photosynthesis simply for a high school student.",
            "What are the primary symptoms of a common cold?",
            "Write a Python function to check if a string is a palindrome.",
            "Suggest a 3-day itinerary for visiting Tokyo.",
            "Draft a professional email requesting a project deadline extension.",
        ],
    }
    pd.DataFrame(sample_data).to_csv(prompts_path, index=False)
    print("Created sample prompts.csv in dataset/ folder.")

# Load prompts
prompts = pd.read_csv(prompts_path)
model = "llama3.2"  # You can switch or add gemma if installed

responses = []
comparisons = []
conversation_logs = []

print(f"Starting local LLM inference using model: {model}...\n")

for index, row in prompts.iterrows():
    prompt_id = row["prompt_id"]
    domain = row["domain"]
    prompt = row["prompt"]

    # Call local Ollama model
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]

    # 1. Collect full response data for llm_responses.csv
    responses.append({
        "prompt_id": prompt_id,
        "domain": domain,
        "prompt": prompt,
        "response": answer
    })

    # 2. Analyze metrics for response_comparison.csv (Length, Word Count, Complexity)
    char_length = len(answer)
    word_count = len(answer.split())
    
    comparisons.append({
        "prompt_id": prompt_id,
        "domain": domain,
        "model": model,
        "response_length_chars": char_length,
        "response_word_count": word_count,
        "complexity_level": "Standard", # Modify based on your prompt testing variation
        "accuracy_score_est": "High"     # Placeholder for manual/automated quality review
    })

    # 3. Format entry for conversation_history.txt
    log_entry = (
        f"=== Prompt ID: {prompt_id} | Domain: {domain} ===\n"
        f"User Prompt:\n{prompt}\n\n"
        f"Model Response ({model}):\n{answer}\n\n"
        f"{'='*60}\n\n"
    )
    conversation_logs.append(log_entry)

# Save outputs to respective files matching your folder structure
# 1. Save LLM Responses
result_df = pd.DataFrame(responses)
result_df.to_csv("dataset/llm_responses.csv", index=False)

# 2. Save Response Comparison Metrics
comparison_df = pd.DataFrame(comparisons)
comparison_df.to_csv("dataset/response_comparison.csv", index=False)

# 3. Save Conversation History Log
with open("dataset/conversation_history.txt", "w", encoding="utf-8") as f:
    f.writelines(conversation_logs)

print("Execution completed successfully!")
print("Generated files:")
print(" - dataset/llm_responses.csv")
print(" - dataset/response_comparison.csv")
print(" - dataset/conversation_history.txt")