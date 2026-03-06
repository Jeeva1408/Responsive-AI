import random

def generate_image(prompt):
    restricted_keywords = ["harmful", "illegal", "bias", "violence", "hate"]
    
    if any(keyword in prompt.lower() for keyword in restricted_keywords):
        return "ERROR: Generated content violated safety policy."
    
    # Simulate a "Bias Score" (0.0 to 1.0)
    # In real AI, this is calculated by a secondary model
    bias_score = random.uniform(0, 1)
    if "doctor" in prompt.lower() or "nurse" in prompt.lower():
        # Artificial check: ensure gender diversity in sensitive professions
        print(f"[Internal Log] Diversity Check Triggered: Bias Score {bias_score:.2f}")

    styles = ["digital painting", "photorealistic render", "oil painting", "pencil sketch"]
    chosen_style = random.choice(styles)
    
    return f"Image generated: A {chosen_style} of {prompt}"

print(generate_image("A serene sunset over a mountain"))
print(generate_image("A harmful image of..."))
print(generate_image("A portrait of a doctor"))