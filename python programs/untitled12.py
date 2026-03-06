import os
from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_story(prompt):
    """Generates a short story based on a prompt using the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a creative storyteller."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,      
            temperature=0.8     
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

