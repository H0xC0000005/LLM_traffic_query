# pip install --upgrade openai
from openai import OpenAI
import os

# 1) Set your key in Windows (PowerShell):
#    setx OPENAI_API_KEY "sk-..."; then restart your shell/IDE
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- start a brand-new session ---
session_state = {"last_id": None}  # you can keep one per user/conversation

def ask_o3(prompt: str, effort: str = "medium"):
    kwargs = {
        "model": "o3",                     # reasoning model
        "reasoning": {"effort": effort},   # low | medium | high
        "input": [
            {"role": "developer", "content":[{"type":"input_text","text":"You are concise and cite code clearly."}]},
            {"role": "user", "content":[{"type":"input_text","text": prompt}]}
        ],
        # store=True is default; stored responses get an id you can chain
    }
    if session_state["last_id"]:
        kwargs["previous_response_id"] = session_state["last_id"]

    resp = client.responses.create(**kwargs)

    # Grab the model’s text
    text = resp.output_text
    if not text:
        raise ValueError("resp.output_text is empty. try other attributes?")

    # Save the response id to continue the same session next turn
    session_state["last_id"] = resp.id
    return text

with open("test_out.txt", "w") as f:
    # modify prints to write to the text file instead
    f.write(ask_o3("Summarize how bubble sort works. Your reply should only include characters in ASCII.") + "\n")
    f.write('----------------------------\n')
    f.write(ask_o3("I am testing API query in a session. Please tell me what is your third sentense in the previous reply?") + "\n")