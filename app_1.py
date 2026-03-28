"""
TalentScout - AI-Powered Hiring Assistant Chatbot
==================================================
A conversational recruitment assistant built with Streamlit and a LOCAL LLM via Ollama.

LLM Backend: Ollama (http://localhost:11434) running LLaMA 3

Features:
- Collects candidate information step-by-step
- Parses tech stack from natural language
- Generates adaptive technical questions (difficulty based on experience)
- Maintains full conversation context via session_state
- Graceful exit and fallback handling

Setup & Run:
    1. Install Ollama:       https://ollama.com/download
    2. Pull LLaMA 3 model:   ollama pull llama3
    3. Start Ollama server:  ollama serve          (runs on http://localhost:11434)
    4. Install Python deps:  pip install streamlit requests
    5. Run the app:          streamlit run app.py
"""

import re
import json
import requests
import streamlit as st


OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

STAGES = [
    "greeting",
    "collect_name",
    "collect_email",
    "collect_phone",
    "collect_experience",
    "collect_position",
    "collect_location",
    "collect_techstack",
    "generate_questions",
    "interview",
    "farewell",
]

STAGE_PROMPTS = {
    "collect_name":       "What is your **full name**?",
    "collect_email":      "What is your **email address**?",
    "collect_phone":      "What is your **phone number**?",
    "collect_experience": "How many **years of professional experience** do you have?",
    "collect_position":   "What **position** are you applying for?",
    "collect_location":   "What is your **current location** (city, country)?",
    "collect_techstack":  (
        "Please list your **tech stack** — programming languages, frameworks, "
        "databases, tools, cloud platforms, etc. (comma-separated or free text is fine)"
    ),
}

EXIT_KEYWORDS = {"exit", "bye", "goodbye", "quit", "stop", "end"}


# ─────────────────────────────────────────────
# ★  LOCAL LLM CALL  (Ollama / LLaMA 3)
# ─────────────────────────────────────────────

def llm_call(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to the locally-running Ollama LLaMA 3 model and return the
    generated text response.

    How it works
    ------------
    Ollama's /api/generate endpoint accepts a single ``prompt`` string, so we
    concatenate the system instructions and the user message with a clear
    separator before sending the request.

    The request is made with ``stream=False`` so Ollama returns the entire
    completion in one JSON response instead of a token-by-token stream.

    Parameters
    ----------
    system     : str   Instruction / role prompt for the model.
    user       : str   The actual user message or task.
    max_tokens : int   Approximate upper limit on output tokens (passed as
                       ``num_predict`` to Ollama).

    Returns
    -------
    str  The model's generated text, stripped of leading/trailing whitespace.
         Returns a user-friendly error string if Ollama is unreachable.

    Ollama API reference
    --------------------
    POST http://localhost:11434/api/generate
    Body (JSON):
        {
            "model":       "llama3",
            "prompt":      "<combined prompt>",
            "stream":      false,
            "options": {
                "num_predict": <max_tokens>,
                "temperature": 0.3
            }
        }
    Response (JSON):
        { "response": "<generated text>", ... }
    """


    combined_prompt = (
        f"[SYSTEM]\n{system.strip()}\n\n"
        f"[USER]\n{user.strip()}\n\n"
        f"[ASSISTANT]\n"
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": combined_prompt,
        "stream": False,                
        "options": {
            "num_predict": max_tokens,    
            "temperature": 0.3,           
        },
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,         
        )
        resp.raise_for_status()   
        data = resp.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "Local LLM is not running. "
            "Please start Ollama with: ollama serve "
            "and ensure the llama3 model is pulled: ollama pull llama3."
        )
    except requests.exceptions.Timeout:
        return (
            "Request timed out. "
            "The model is taking too long to respond. "
            "Try again or check your Ollama server."
        )
    except requests.exceptions.HTTPError as e:
        return f"Ollama API error: {e}"
    except Exception as e:
        return f"Unexpected error contacting local LLM: {e}"



def init_session():
    """Initialise all session-state keys on first load."""
    defaults = {
        "stage": "greeting",
        "messages": [],           
        "candidate": {           
            "name": None,
            "email": None,
            "phone": None,
            "experience": None,
            "position": None,
            "location": None,
            "tech_stack": [],
        },
        "questions": {},         
        "q_index": 0,            
        "flat_questions": [],    
        "answers": [],          
        "greeted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def append_message(role: str, content: str):
    """Append a message to the conversation history."""
    st.session_state.messages.append({"role": role, "content": content})


def advance_stage():
    """Move to the next stage in the STAGES list."""
    current = st.session_state.stage
    idx = STAGES.index(current)
    if idx + 1 < len(STAGES):
        st.session_state.stage = STAGES[idx + 1]




def parse_tech_stack(raw_input: str) -> list[str]:
    """
    Use the local LLM to extract a clean list of technologies from free-form
    user input.

    Prompt engineering note:
        We instruct the model to return ONLY a JSON array of strings so we
        can parse it deterministically.  A fallback comma-split is used if
        JSON parsing fails.

    Parameters
    ----------
    raw_input : str  Free-form tech stack description from the candidate.

    Returns
    -------
    list[str]  Deduplicated list of technology names.
    """
    system = """You are a tech-stack extraction assistant.
Your ONLY job is to extract technology names (languages, frameworks, databases, tools, cloud platforms)
from the user's input and return them as a JSON array of strings.

Rules:
- Return ONLY a valid JSON array, e.g. ["Python", "Django", "PostgreSQL"]
- No markdown, no backticks, no explanation, no extra text.
- Normalize capitalisation: "python" -> "Python", "reactjs" -> "React".
- Remove duplicates.
- If nothing recognisable is found, return [].
- Your entire response must be ONLY the JSON array and nothing else."""

    raw = llm_call(system, raw_input, max_tokens=512)

  
    error_indicators = ["Local LLM is not running", "Request timed out", "Ollama API error", "Unexpected error"]
    if any(indicator in raw for indicator in error_indicators):
  
        return [t.strip().capitalize() for t in re.split(r"[,;/]", raw_input) if t.strip()]

    try:
        
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
       
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        technologies = json.loads(clean)
        if isinstance(technologies, list):
            return [str(t).strip() for t in technologies if t]
    except (json.JSONDecodeError, ValueError):
        pass

  
    return list(set([t.strip().capitalize() for t in re.split(r"[,;/]", raw_input) if t.strip()]))


def generate_technical_questions(
    technologies: list[str],
    years_of_experience: int | float,
    position: str,
) -> dict[str, list[str]]:
    """
    Generate 3-5 technical interview questions per technology using the local LLM.

    Prompt engineering note:
        - Difficulty is calibrated to years_of_experience:
            0-2 yrs  -> basic/foundational
            3-5 yrs  -> intermediate/applied
            6+  yrs  -> advanced/architectural
        - Output is a strict JSON object for deterministic parsing.

    Parameters
    ----------
    technologies        : list of technology names
    years_of_experience : candidate's experience in years
    position            : role being applied for

    Returns
    -------
    dict[str, list[str]]  {technology: [question, ...]}
    """
    if years_of_experience <= 2:
        difficulty = "basic to intermediate (focus on fundamentals, syntax, core concepts)"
    elif years_of_experience <= 5:
        difficulty = "intermediate (focus on applied usage, design patterns, best practices)"
    else:
        difficulty = "intermediate to advanced (focus on architecture, performance, trade-offs)"

    tech_list = ", ".join(technologies)

    system = f"""You are a senior technical interviewer conducting a screening for a {position} role.
Generate practical technical interview questions for the technologies provided.

Difficulty level: {difficulty} (candidate has {years_of_experience} years of experience)

Return ONLY a valid JSON object where keys are technology names and values are arrays of 3 to 5 question strings.
Example format:
{{
  "Python": ["What is a list comprehension?", "Explain GIL.", "How does async/await work?"],
  "SQL":    ["What is a JOIN?", "Explain ACID properties."]
}}

Rules:
- No markdown, no backticks, no explanation outside the JSON.
- Questions must be practical, clear, and relevant to real-world usage.
- Mix conceptual and problem-solving questions.
- Questions should be answerable in 1 to 3 sentences (no coding exercises).
- Your entire response must be ONLY the JSON object and nothing else."""

    user = f"Technologies: {tech_list}"
    raw = llm_call(system, user, max_tokens=2048)


    error_indicators = ["Local LLM is not running", "Request timed out", "Ollama API error", "Unexpected error"]
    if any(indicator in raw for indicator in error_indicators):
        return {tech: [f"Tell me about your experience with {tech}."] for tech in technologies}

    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
     
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        questions = json.loads(clean)
        if isinstance(questions, dict):
            return {k: v for k, v in questions.items() if isinstance(v, list)}
    except (json.JSONDecodeError, ValueError):
        pass


    return {tech: [f"Tell me about your experience with {tech}."] for tech in technologies}



def validate_email(email: str) -> bool:
    """Basic regex email validation."""
    return bool(re.match(r"^[\w.\-+]+@[\w\-]+\.[a-zA-Z]{2,}$", email.strip()))


def validate_phone(phone: str) -> bool:
    """Accept digits, spaces, dashes, parentheses, plus sign (7-15 digits)."""
    digits = re.sub(r"[\s\-()+]", "", phone)
    return digits.isdigit() and 7 <= len(digits) <= 15


def parse_experience(text: str) -> float | None:
    """Extract a numeric year value from text like '5 years', '3.5', 'two'."""
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    lower = text.lower().strip()
    for word, val in words.items():
        if word in lower:
            return float(val)
    match = re.search(r"\d+(\.\d+)?", lower)
    if match:
        return float(match.group())
    return None


def is_exit(text: str) -> bool:
    """Return True if the user wants to end the conversation."""
    return text.strip().lower() in EXIT_KEYWORDS



def handle_greeting():
    """Send the opening greeting message once."""
    if not st.session_state.greeted:
        msg = (
            "👋 **Welcome to TalentScout!**\n\n"
            "I'm your AI-powered recruitment assistant running on a **local LLM (LLaMA 3 via Ollama)**. "
            "I'll guide you through a quick screening process — collecting some basic information "
            "and then asking a few technical questions tailored to your experience.\n\n"
            "You can type **exit** at any time to end the conversation.\n\n"
            "Let's get started! " + STAGE_PROMPTS["collect_name"]
        )
        append_message("assistant", msg)
        st.session_state.stage = "collect_name"
        st.session_state.greeted = True


def handle_collect_name(user_input: str):
    name = user_input.strip()
    if len(name.split()) < 1 or len(name) < 2:
        append_message("assistant", "I didn't quite catch that. Could you please share your **full name**?")
        return
    st.session_state.candidate["name"] = name
    advance_stage()
    append_message("assistant", f"Nice to meet you, **{name}**! 😊\n\n{STAGE_PROMPTS['collect_email']}")


def handle_collect_email(user_input: str):
    email = user_input.strip()
    if not validate_email(email):
        append_message("assistant", "That doesn't look like a valid email address. Could you double-check and try again?")
        return
    st.session_state.candidate["email"] = email
    advance_stage()
    append_message("assistant", f"Got it! ✉️\n\n{STAGE_PROMPTS['collect_phone']}")


def handle_collect_phone(user_input: str):
    phone = user_input.strip()
    if not validate_phone(phone):
        append_message("assistant", "That phone number looks a bit off. Please enter a valid phone number (7-15 digits).")
        return
    st.session_state.candidate["phone"] = phone
    advance_stage()
    append_message("assistant", f"Perfect! 📱\n\n{STAGE_PROMPTS['collect_experience']}")


def handle_collect_experience(user_input: str):
    years = parse_experience(user_input)
    if years is None or years < 0:
        append_message("assistant", "I couldn't parse that. Please tell me your years of experience as a number, e.g. **3** or **5.5**.")
        return
    st.session_state.candidate["experience"] = years
    advance_stage()
    append_message("assistant", f"Great — **{years} year(s)** of experience noted! 🏆\n\n{STAGE_PROMPTS['collect_position']}")


def handle_collect_position(user_input: str):
    position = user_input.strip()
    if len(position) < 2:
        append_message("assistant", "Could you please be more specific about the position you're applying for?")
        return
    st.session_state.candidate["position"] = position
    advance_stage()
    append_message("assistant", f"Awesome! Applying for **{position}**. 💼\n\n{STAGE_PROMPTS['collect_location']}")


def handle_collect_location(user_input: str):
    location = user_input.strip()
    if len(location) < 2:
        append_message("assistant", "Could you share your current city or country?")
        return
    st.session_state.candidate["location"] = location
    advance_stage()
    append_message("assistant", f"📍 **{location}** — noted!\n\n{STAGE_PROMPTS['collect_techstack']}")


def handle_collect_techstack(user_input: str):
    with st.spinner("🔍 Analysing your tech stack with local LLM..."):
        tech_stack = parse_tech_stack(user_input)

    if not tech_stack:
        append_message("assistant",
            "I couldn't identify any technologies from your input. "
            "Could you list them again? For example: *Python, Django, PostgreSQL, Docker*.\n\n"
            "*(If Ollama is not running, start it with: `ollama serve`)*")
        return

    st.session_state.candidate["tech_stack"] = tech_stack
    tech_display = ", ".join(f"**{t}**" for t in tech_stack)
    append_message("assistant",
        f"Got it! I've identified the following technologies: {tech_display}\n\n"
        "⏳ Generating personalised technical questions — this may take a moment...")
    advance_stage()

    handle_generate_questions()


def handle_generate_questions():
    """Generate technical questions and prepare the flat question list."""
    candidate = st.session_state.candidate
    with st.spinner("🤖 LLaMA 3 is crafting your technical questions..."):
        questions = generate_technical_questions(
            technologies=candidate["tech_stack"],
            years_of_experience=candidate["experience"],
            position=candidate["position"],
        )

    st.session_state.questions = questions


    flat = []
    for tech, qs in questions.items():
        for q in qs:
            flat.append((tech, q))
    st.session_state.flat_questions = flat
    st.session_state.q_index = 0


    summary_lines = [f"- **{tech}**: {len(qs)} question(s)" for tech, qs in questions.items()]

    name = candidate["name"].split()[0]
    append_message("assistant",
        f"✅ Questions ready, {name}! Here's what we'll cover:\n\n"
        + "\n".join(summary_lines)
        + "\n\nLet's begin the technical interview. Take your time with each answer.\n\n"
        + _format_next_question()
    )
    advance_stage() 


def _format_next_question() -> str:
    """Return the next question formatted with its index and technology label."""
    flat = st.session_state.flat_questions
    idx = st.session_state.q_index
    if idx >= len(flat):
        return ""
    tech, question = flat[idx]
    total = len(flat)
    return f"**[{idx + 1}/{total}] {tech}:** {question}"


def handle_interview(user_input: str):
    """Record the candidate's answer and advance to the next question or farewell."""
    flat = st.session_state.flat_questions
    idx = st.session_state.q_index
    tech, question = flat[idx]


    st.session_state.answers.append((tech, question, user_input.strip()))
    st.session_state.q_index += 1
    new_idx = st.session_state.q_index

    if new_idx >= len(flat):
  
        advance_stage()
        handle_farewell()
    else:
    
        acknowledgements = [
            "Thanks for your answer!",
            "Got it, moving on!",
            "Great response! Next one:",
            "Noted! Here's the next question:",
            "Appreciated! Let's continue:",
        ]
        ack = acknowledgements[new_idx % len(acknowledgements)]
        append_message("assistant", f"{ack}\n\n{_format_next_question()}")


def handle_farewell():
    """Display a warm closing message with a candidate summary."""
    c = st.session_state.candidate
    total_q = len(st.session_state.flat_questions)
    total_a = len(st.session_state.answers)

    append_message("assistant",
        f"🎉 **Interview complete!** Thank you, **{c['name']}**!\n\n"
        f"Here's a summary of your session:\n"
        f"- 📧 **Email:** {c['email']}\n"
        f"- 📞 **Phone:** {c['phone']}\n"
        f"- 🏙️ **Location:** {c['location']}\n"
        f"- 💼 **Position:** {c['position']}\n"
        f"- ⏳ **Experience:** {c['experience']} year(s)\n"
        f"- 🛠️ **Tech Stack:** {', '.join(c['tech_stack'])}\n"
        f"- ✅ **Questions answered:** {total_a}/{total_q}\n\n"
    
        "Best of luck! 🚀 Type **exit** or simply close this tab whenever you're ready."
    )


def handle_user_input(user_input: str):
    """Route user input to the correct stage handler."""
    stage = st.session_state.stage

    
    if is_exit(user_input):
        name = st.session_state.candidate.get("name") or "there"
        first = name.split()[0] if name != "there" else name
        append_message("assistant",
            f"👋 Goodbye, **{first}**! Thanks for chatting with TalentScout. "
            "We hope to connect with you soon. Take care! 😊")
        st.session_state.stage = "farewell"
        return

  
    handlers = {
        "collect_name":       handle_collect_name,
        "collect_email":      handle_collect_email,
        "collect_phone":      handle_collect_phone,
        "collect_experience": handle_collect_experience,
        "collect_position":   handle_collect_position,
        "collect_location":   handle_collect_location,
        "collect_techstack":  handle_collect_techstack,
        "interview":          handle_interview,
    }

    if stage in handlers:
        handlers[stage](user_input)
    elif stage == "farewell":
        append_message("assistant",
            "The interview session has ended. Refresh the page to start a new session. 😊")
    else:
    
        append_message("assistant",
            "I'm not sure how to process that right now. "
            "Could you please rephrase, or type **exit** to end the session?")



def setup_page():
    """Configure Streamlit page and inject custom CSS."""
    st.set_page_config(
        page_title="TalentScout — Hiring Assistant",
        page_icon="🎯",
        layout="centered",
    )
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Grotesk:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Header ── */
    .ts-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .ts-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .ts-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ── Chat container ── */
    .chat-wrap {
        max-width: 760px;
        margin: 0 auto;
        padding: 0 1rem 7rem;
    }

    /* ── Message bubbles ── */
    .msg-row {
        display: flex;
        margin-bottom: 1.1rem;
        animation: fadeUp 0.3s ease;
    }
    .msg-row.user  { justify-content: flex-end; }
    .msg-row.bot   { justify-content: flex-start; }

    .bubble {
        max-width: 78%;
        padding: 0.8rem 1.1rem;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .bubble.bot {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        border-top-left-radius: 4px;
    }
    .bubble.user {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: #fff;
        border-top-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(109,40,217,0.4);
    }
    .avatar {
        width: 34px; height: 34px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .msg-row.bot  .avatar { background: rgba(99,102,241,0.25); margin-right: 0.6rem; }
    .msg-row.user .avatar { background: rgba(109,40,217,0.4);  margin-left:  0.6rem; }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Input area ── */
    .input-dock {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: rgba(15, 12, 41, 0.92);
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(255,255,255,0.08);
        padding: 0.9rem 1rem;
        z-index: 100;
    }
    .input-inner {
        max-width: 760px;
        margin: 0 auto;
        display: flex;
        gap: 0.6rem;
        align-items: center;
    }

    /* ── Streamlit input overrides ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1.5px solid rgba(139,92,246,0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        white-space: nowrap !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(109,40,217,0.5) !important;
    }

    /* ── Stage pill ── */
    .stage-pill {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-family: 'Space Grotesk', sans-serif;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    /* ── Ollama status badge ── */
    .ollama-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.3);
        color: #6ee7b7;
        font-size: 0.72rem;
        font-family: 'Space Grotesk', sans-serif;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-left: 0.5rem;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; }
    </style>
    """, unsafe_allow_html=True)



def render_header():
    st.markdown("""
    <div class="ts-header">
        <h1>🎯 TalentScout</h1>
        <p>
            AI-Powered Hiring Assistant
            <span class="ollama-badge">⚡ Local LLM &middot; LLaMA 3 via Ollama</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_messages():
    """Render all chat messages as styled bubbles."""
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)


    stage = st.session_state.stage
    readable = stage.replace("_", " ").title()
    st.markdown(
        f'<div style="text-align:center"><span class="stage-pill">Stage: {readable}</span></div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "assistant":
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="avatar">🤖</div>
                <div class="bubble bot">{_md_to_html(content)}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row user">
                <div class="bubble user">{_escape(content)}</div>
                <div class="avatar">👤</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _escape(text: str) -> str:
    """HTML-escape user text to prevent XSS."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>"))


def _md_to_html(text: str) -> str:
    """
    Minimal markdown to HTML converter for bold, italic, inline code,
    newlines, and bullet points.
    Avoids pulling in a full markdown library.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    text = re.sub(
        r"`(.+?)`",
        r"<code style='background:rgba(255,255,255,0.1);padding:0 4px;border-radius:4px'>\1</code>",
        text,
    )

    lines = text.split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul style='margin:0.4rem 0 0.4rem 1.2rem;padding:0'>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(line)
    if in_list:
        html_lines.append("</ul>")
    return "<br>".join(html_lines)



def main():
    setup_page()
    init_session()

    render_header()


    if st.session_state.stage == "greeting":
        handle_greeting()

    render_messages()

    st.markdown('<div class="input-dock"><div class="input-inner">', unsafe_allow_html=True)

    is_done = st.session_state.stage == "farewell"
    placeholder = "Type your answer here..." if not is_done else "Session complete. Refresh to restart."

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        if "clear_input" in st.session_state and st.session_state["clear_input"]:
            st.session_state["user_input_field"] = ""
            st.session_state["clear_input"] = False

        user_text = st.text_input(

            label="user_input",
            placeholder=placeholder,
            label_visibility="collapsed",
            disabled=is_done,
            key="user_input_field",
        )

    with col_btn:
        send_clicked = st.button("Send ➤", disabled=is_done)

    st.markdown('</div></div>', unsafe_allow_html=True)

    
    if send_clicked:
        if user_text.strip():
            append_message("user", user_text.strip())
            handle_user_input(user_text.strip())

            st.session_state["clear_input"] = True
            st.rerun()


if __name__ == "__main__":
    main()
