from flask import Flask, request, jsonify, render_template
import pandas as pd
from dotenv import load_dotenv
import os
import openai
from rapidfuzz import process, fuzz


load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("API Key Loaded Successfully!")
    openai.api_key = openai_api_key
else:
    print("Error: OPENAI_API_KEY not found")

app = Flask(__name__, template_folder='templates')

conversation_history = {}

SCORE_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", 70))

intent_triggers = {
    "cause": ["cause", "causes", "why", "reason"],
    "symptom": ["symptom", "symptoms", "signs", "feel"],
    "treatment": ["treat", "treatment", "cure", "manage"],
    "prevention": ["prevent", "prevention", "avoid", "avoidance"],
    "risk": ["risk", "risks", "prone", "likely"],
    "incubation": ["incubation", "period"],
}

def detect_intent(s):
    s = s.lower()
    for intent, triggers in intent_triggers.items():
        if any(t in s for t in triggers):
            return intent
    return None

def best_match(user_msg, questions, intent=None):
    msg = user_msg.lower()
    

    if intent:
        candidates = [q for q in questions if any(t in q for t in intent_triggers[intent])]
    else:
        candidates = questions

    match = process.extractOne(
        msg,
        candidates,
        scorer=fuzz.WRatio,
        score_cutoff=SCORE_THRESHOLD
    )
    return match[0] if match else None

# Load data from CSV
def load_qa_data(csv_path):
    try:
        df = pd.read_csv(csv_path)

        data = {}
        for _, row in df.iterrows():
            question = str(row["QUESTION"]).strip()
            answer = str(row["ANSWER"]).strip()
            data[question.lower()] = answer

        print("✅ CSV data loaded successfully.")
        return data

    except Exception as e:
        print("❌ Error loading CSV file:", e)
        return {}

qa_data = load_qa_data("DATASET1.csv")  

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].lower().strip()


    user_id = request.json.get('user_id', 'default_user')
    user_context = conversation_history.get(user_id, [])


    current_intent = detect_intent(user_message)

    if user_context:

        previous_intent = user_context[-1].get('intent')
        best_answer = best_match(user_message, qa_data.keys(), intent=current_intent or previous_intent)
    else:
        best_answer = best_match(user_message, qa_data.keys(), intent=current_intent)


    if best_answer:
        response = qa_data.get(best_answer.lower())
    else:
        try:
            gpt_reply = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful medical assistant."},
                          {"role": "user", "content": user_message}]
            )
            response = gpt_reply['choices'][0]['message']['content'].strip()
        except Exception as e:
            response = "Sorry, I couldn't fetch an answer right now."

    conversation_history[user_id] = conversation_history.get(user_id, []) + [{'intent': current_intent, 'response': response}]
    
    return jsonify({'reply': response})

@app.route("/test-key")
def test_key():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello from browser route!"}]
        )
        reply = response.choices[0].message["content"]
        return f"<h2>✅ API Key is working!</h2><p><strong>Response:</strong> {reply}</p>"
    except Exception as e:
        return f"<h2>❌ API Key Error:</h2><pre>{e}</pre>"

if __name__ == '__main__':
    app.run(debug=True)
