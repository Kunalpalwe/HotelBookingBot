import gradio as gr
import google.generativeai as genai
import os
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ------ Configuration -----
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    'gemini-1.5-flash',
)


# ------- JSON structure for Gemini's response ------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "next_action": {
            "type": "string",
            "description": "The next logical step/stage in the conversation flow.",
             "enum": ["ASK_AGE", "ASK_EMAIL", "CONFIRM_BOOKING", "REJECT_BOOKING", "REQUEST_CLARIFICATION", "END_CONVERSATION"]
        },
        "bot_message": {
            "type": "string",
            "description": "The conversational message to display to the user."
        },
        "extracted_data": {
            "type": "object",
            "description": "Any data successfully extracted or confirmed in this turn (optional).",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            }
        }
    },
    "required": ["next_action", "bot_message"]
}


# --- Excel File Configuration ---
EXCEL_FILE_PATH = Path("bookings.xlsx") # Use Path for better path handling
EXCEL_COLUMNS = ["Name", "Age", "Email", "Booking Timestamp"]


# --- State Management ---
INITIAL_STATE = {
    "stage": "ASK_NAME",
    "user_name": None,
    "user_age": None,
    "user_email": None,
    "error_count": 0,
}
MAX_ERRORS = 2


# --- Helper Functions ---
def is_valid_email(email):
    """Basic email format check."""
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

def clean_json_response(raw_text):
    """Attempts to extract valid JSON from Gemini's potentially messy output."""
    match = re.search(r'```json\s*({.*?})\s*```', raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        brace_match = re.search(r'({.*?})', raw_text, re.DOTALL)
        if brace_match:
             json_str = brace_match.group(1)
        else:
             return None
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw JSON string attempted: {json_str}")
        return None

def generate_structured_gemini_response(prompt):
    """Generates a response using the Gemini API, expecting structured JSON output."""
    full_prompt = f"""{prompt}

    Respond STRICTLY in the following JSON format. Do not include any text outside the JSON structure.
    ```json
    {{
      "next_action": "ACTION_ENUM",
      "bot_message": "Your response message here.",
      "extracted_data": {{ ... }}
    }}
    ```
    Ensure 'next_action' is one of: {JSON_SCHEMA['properties']['next_action']['enum']}
    """
    try:
        response = model.generate_content(full_prompt)
        if response.candidates and response.candidates[0].content.parts:
            raw_text = response.text
            print(f"--- Gemini Raw Response ---\n{raw_text}\n--------------------------")
            parsed_json = clean_json_response(raw_text)
            if parsed_json and 'next_action' in parsed_json and 'bot_message' in parsed_json:
                return parsed_json
            else:
                 print("Warning: Gemini response missing required JSON fields or failed parsing.")
                 return None
        else:
             print("Warning: Gemini response was empty or blocked.")
             return None
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


# --- Save Booking to Excel ---
def save_booking_to_excel(name, age, email):
    """Appends a new booking record to the Excel file."""
    try:
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        new_booking_data = {"Name": [name], "Age": [age], "Email": [email], "Booking Timestamp": [timestamp]}
        new_booking_df = pd.DataFrame(new_booking_data)

        if EXCEL_FILE_PATH.exists():
            try:
                existing_df = pd.read_excel(EXCEL_FILE_PATH)
                for col in EXCEL_COLUMNS:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                existing_df = existing_df[EXCEL_COLUMNS]
            except Exception as read_err:
                 print(f"Could not read existing Excel file, creating new one. Error: {read_err}")
                 existing_df = pd.DataFrame(columns=EXCEL_COLUMNS)

            combined_df = pd.concat([existing_df, new_booking_df], ignore_index=True)
        else:
            print(f"Excel file '{EXCEL_FILE_PATH}' not found. Creating new file.")
            combined_df = new_booking_df

        combined_df.to_excel(EXCEL_FILE_PATH, index=False, columns=EXCEL_COLUMNS)
        print(f"Booking for {name} saved successfully to '{EXCEL_FILE_PATH}'.")

    except PermissionError:
         print(f"Error: Permission denied to write to '{EXCEL_FILE_PATH}'. Make sure the file is not open and you have write permissions.")
    except Exception as e:
        print(f"An error occurred while saving to Excel: {e}")


# --- Core Chat Logic ---

def hotel_booking_logic(user_input, history, state):
    """Handles the conversation flow using structured output from Gemini."""
    bot_message = "Sorry, something went wrong. Let's try restarting. What is your name?"
    next_stage = state.get("stage", "ASK_NAME")
    history = history or []
    current_stage = state.get("stage", "ASK_NAME")
    prompt = ""

    if current_stage == "ASK_NAME":
        if user_input and len(user_input.strip()) > 1:
             potential_name = user_input.strip()
             prompt = f"""
             User provided their name: '{potential_name}'.
             Task: Acknowledge the name '{potential_name}'. Ask the user for their age as the next step required for hotel booking.
             Set next_action to 'ASK_AGE'. Include the name in extracted_data.
             """
        else:
             bot_message = "I need a valid name to proceed. Please tell me your name."
             next_stage = "ASK_NAME"; state["error_count"] +=1
             if state["error_count"] >= MAX_ERRORS: bot_message = "Having trouble getting your name. Let's restart."; state = INITIAL_STATE.copy(); next_stage = state["stage"]
             history.append([user_input, bot_message]); return history, state

    elif current_stage == "ASK_AGE":
        try:
            age = int(user_input.strip())
            if age <= 0: raise ValueError("Age must be positive")
            state["user_age"] = age; user_name = state.get('user_name', 'there')
            if age >= 18:
                prompt = f"""
                User '{user_name}' provided their age: {age} (which is >= 18).
                Task: Acknowledge the age. Ask the user for their email address as the next step.
                Set next_action to 'ASK_EMAIL'. Include the age in extracted_data.
                """
            else:
                prompt = f"""
                User '{user_name}' provided their age: {age} (which is < 18).
                Task: Politely explain that the user cannot book a hotel because the minimum age is 18. Do not ask further questions.
                Set next_action to 'REJECT_BOOKING'. Include the age in extracted_data.
                """
        except ValueError:
             state["error_count"] += 1
             if state["error_count"] >= MAX_ERRORS: bot_message = "Trouble with age input. Let's restart. Name?"; state = INITIAL_STATE.copy()
             else: bot_message = "Invalid age. Please enter a number (e.g., 25)."
             next_stage = "ASK_AGE"; history.append([user_input, bot_message]); return history, state

    elif current_stage == "ASK_EMAIL":
        email_input = user_input.strip(); user_name = state.get('user_name', 'Guest')
        if is_valid_email(email_input):
            state["user_email"] = email_input
            prompt = f"""
            User '{user_name}' (age {state.get('user_age')}) provided their email: '{email_input}'.
            Task: Generate a friendly hotel booking confirmation message. State the booking is successful and details will be sent to the email.
            Set next_action to 'CONFIRM_BOOKING'. Include the email in extracted_data.
            """
        else:
             state["error_count"] += 1
             if state["error_count"] >= MAX_ERRORS: bot_message = "Trouble with email. Let's restart. Name?"; state = INITIAL_STATE.copy()
             else: bot_message = "Invalid email format. Please enter a valid email (e.g., name@example.com)."
             next_stage = "ASK_EMAIL"; history.append([user_input, bot_message]); return history, state

    elif current_stage in ["BOOKING_COMPLETE", "AGE_REJECTED", "CONFIRM_BOOKING", "REJECT_BOOKING"]:
        previous_outcome = state.get("stage", "ended")
        if "start again" in user_input.lower() or "restart" in user_input.lower() or "new booking" in user_input.lower():
             state = INITIAL_STATE.copy(); bot_message = "Okay, let's start over! What is your name?"; next_stage = "ASK_NAME"
             history.append([user_input, bot_message]); return history, state
        else:
            prompt = f"""
            The user said: '{user_input}'. The previous interaction ended with state: '{previous_outcome}'.
            Task: Respond politely. Ask if there's anything else or if they'd like to start a new booking request.
            Set next_action to 'END_CONVERSATION' unless they clearly want to restart.
            """

    # --- Call Gemini and Process Structured Response ---
    if prompt:
        print(f"--- Sending Prompt to Gemini ---\n{prompt}\n-----------------------------")
        structured_response = generate_structured_gemini_response(prompt)

        if structured_response:
            bot_message = structured_response.get("bot_message", "I seem to have lost my train of thought...")
            action_received = structured_response.get("next_action")

            extracted = structured_response.get("extracted_data", {})
            if "name" in extracted: state["user_name"] = extracted["name"]
            if "age" in extracted: state["user_age"] = extracted["age"]
            if "email" in extracted: state["user_email"] = extracted["email"]

            if action_received in JSON_SCHEMA['properties']['next_action']['enum']:
                next_stage = action_received
                state["stage"] = next_stage
                state["error_count"] = 0

                # --- SAVE TO EXCEL on CONFIRM_BOOKING  ---
                if next_stage == "CONFIRM_BOOKING":
                    if state.get("user_name") and state.get("user_age") and state.get("user_email"):
                        save_booking_to_excel(
                            state["user_name"],
                            state["user_age"],
                            state["user_email"]
                        )
                    else:
                        print("Warning: Could not save booking to Excel because some user data was missing in the state.")

            else:
                 print(f"Warning: Unexpected next_action: {action_received}")
                 next_stage = state['stage']
                 bot_message += "\nUnsure on the next step. Could you clarify?"
        else:
            state["error_count"] += 1
            if state["error_count"] >= MAX_ERRORS:
                 bot_message = "Processing error. Let's restart. Name?"; state = INITIAL_STATE.copy()
            else:
                 bot_message = "Sorry, encountered an issue. Try again?"
            next_stage = state["stage"]
    else:
        print("Warning: No prompt generated for Gemini call.")

    history.append([user_input, bot_message])
    print(f"--- Current State ---\n{state}\n---------------------")
    return history, state


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("""
    # Hotel Booking Chatbot 
    Talk to the bot to book your hotel! Follow the steps: Name -> Age -> Email.
    """)
    chatbot_state = gr.State(value=INITIAL_STATE.copy())
    chatbot_display = gr.Chatbot(label="Hotel Bot", value=[[None, "Hello! I'm the Hotel Booking Bot. To start, please tell me your name."]], height=500)
    user_textbox = gr.Textbox(placeholder="Type your message here...", label="Your Message", lines=2)
    clear_btn = gr.Button("Clear Chat & Restart")

    def clear_chat():
        return [[None, "Hello! I'm the Hotel Booking Bot. To start, please tell me your name."]], INITIAL_STATE.copy()

    user_textbox.submit(hotel_booking_logic, [user_textbox, chatbot_display, chatbot_state], [chatbot_display, chatbot_state])
    clear_btn.click(clear_chat, None, [chatbot_display, chatbot_state], queue=False)
    user_textbox.submit(lambda: gr.update(value=''), None, user_textbox, queue=False)


# --- Run the App ---
if __name__ == "__main__":
    print(f"Starting Gradio app... Bookings will be saved to '{EXCEL_FILE_PATH.resolve()}'.")
    demo.queue().launch(debug=True)