# Code from kaggle 
import torch
import transformers
from transformers import pipeline
!pip install openai
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("openai_api_key") 
client = OpenAI(api_key=api_key)

def get_theme():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Generate a theme for a short joke."}]
    )
    theme = response.choices[0].message.content
    return theme

theme = get_theme() 
print("Theme:", theme)

def get_response_llama(model_path, theme):
    model_pipeline = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    prompt = f"You are a comedian at a comedy. Generate your best short joke based on this theme: {theme}"
    sequences = model_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=0.85,
        num_return_sequences=1,
        eos_token_id=model_pipeline.tokenizer.eos_token_id,
        max_length=150,
        truncation=True
    )
    
    joke = sequences[0]["generated_text"][len(prompt):].strip()
    return joke

def get_response_mistral(model_path, theme):
    model_pipeline = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    prompt = f"You are a comedian at a comedy. Generate your best short joke based on this theme: {theme}"
    sequences = model_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=0.85,
        num_return_sequences=1,
        eos_token_id=model_pipeline.tokenizer.eos_token_id,
        max_length=200,
        truncation=True
    )
    
    joke = sequences[0]["generated_text"][len(prompt):].strip()
    return joke


llama_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
mistral_model = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"

joke_llama = get_response_llama(llama_model, theme)
joke_mistral = get_response_mistral(mistral_model, theme)

print("Joke from LLaMA:", joke_llama)
print("Joke from Mistral:", joke_mistral)


def get_rating(joke1, joke2):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[{"role": "user", "content": f"Here are two jokes:\n1. {joke_llama}\n2. {joke_mistral}\nCan you compare these jokes and rate them on a scale of 1 to 10?"}],
        max_tokens=150  
    )
    rating = response.choices[0].message.content
    return rating

rating = get_rating(joke_llama, joke_mistral)
print("Response:", rating)