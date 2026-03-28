# TalentScout — AI-Powered Hiring Assistant

## Overview

TalentScout is a conversational recruitment chatbot built using Streamlit and a local Large Language Model (LLM) via Ollama. The application conducts an initial candidate screening by collecting user details and generating technical interview questions based on the candidate’s experience and technology stack.

The system is designed to work without paid APIs by leveraging a locally hosted LLM.

---

## Features

* Step-by-step conversational data collection
* Input validation for email, phone, and experience
* Tech stack extraction from free-form user input
* Dynamic technical question generation based on experience level
* Local LLM integration using Ollama
* Fallback logic for handling unreliable LLM outputs
* Stateful conversation using Streamlit session management
* Exit option available at any stage

---

## Tech Stack

* Python
* Streamlit
* Ollama (Local LLM runtime)
* LLaMA / Phi3 models
* Requests library
* Regex and JSON parsing

---

## System Workflow

1. User interacts with chatbot through Streamlit interface
2. Candidate details are collected step-by-step
3. Tech stack is extracted using LLM with fallback logic
4. Technical questions are generated dynamically
5. Candidate answers are recorded and summarized

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd TalentScout
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install from:
https://ollama.com

### 4. Run Local Model

```bash
ollama run phi3
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

### 6. Open in Browser

http://localhost:8501

---

## Example Usage

* Enter candidate details (name, email, phone, etc.)
* Provide tech stack (e.g., Python, SQL, Power BI)
* System generates technical questions
* Candidate answers questions interactively
* Final summary is displayed

---

## Key Implementation Details

* LLM integration is handled via HTTP requests to Ollama API
* Prompt engineering ensures structured JSON outputs
* Fallback parsing is implemented when LLM responses fail
* Session state is used to maintain conversation flow
* Input validation prevents incorrect data entry



---

## Author

Punith Reddy
Data Analytics Enthusiast
