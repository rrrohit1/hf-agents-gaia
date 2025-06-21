---
title: HF Agents GAIA Benchmark
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Final project for the Hugging Face Agents course — building and evaluating a custom AI agent on the GAIA benchmark. 
Check out the gradio [app](https://huggingface.co/spaces/rrrohit/hf-agents-gaia) hosted at Huggingface Spaces.

## HF Agents Gaia

A modular agent framework built using **LangChain** and **Gemini 2-flash models**, designed for deploying intelligent, vision-enabled, and instruction-following agents on **Hugging Face Spaces**.

This repository contains my final project for the Hugging Face Agents Course. The goal of the assignment is to design, build, and evaluate an AI agent capable of solving a set of tasks based on a subset of the GAIA benchmark. The agent must achieve a performance score of 30% or higher to successfully complete the course and earn certification.

## Key Features

* **LangGraph-style Agent Orchestration** – Modular structure for defining complex agent workflows.
* **Multimodal Gemini Integration** – Supports image and text inputs using `gemini-2.0-flash` and `gemini-2.5-flash-preview-05-20`.
* **Custom Tooling** – Integrates web search, Wikipedia, tabular response formatting, and local file I/O.
* **Audio + Vision Support** – Includes Whisper for transcription and PIL for vision I/O.
* **Built-in Tracing** – Supports LangSmith or console tracing for debugging agent reasoning.

## Agent Overview

The agent (`GaiaAgent`) is designed to answer grounded, factual queries using **Langgraph** + **Gemini** based tools.

## Tools Used

| Tool Name                 | Description                                |
| ------------------------- | ------------------------------------------ |
| `DuckDuckGoSearchResults` | Real-time web search                       |
| `WikipediaQueryRun`       | Structured Wikipedia access                |
| `Whisper`                 | Audio transcription (OpenAI)               |
| `Gemini`                  | Multimodal reasoning (text + image)        |
| `PIL` + `numpy`           | Image preprocessing                        |
| `Tabulate`                | Converts dict results into readable tables |

## Directory Structure

```md
hf-agents-gaia/
│
├── app.py               # Hugging Face Space entry point
├── agent.py             # Main GaiaAgent logic with tool & model config
├── tools.py             # Custom tools (e.g., tabulate, file tools)
├── README.md            # This file
└── requirements.txt     # All dependencies
```

## Setup Instructions

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Add secrets (optional for Spaces):**
   Create a `.env` file:

   ```txt
   GEMINI_API_KEY=your_key
   GEMINI_MODEL=your_model_name
   ```

3. **Run locally:**

   ```bash
   python app.py
   ```

---

## Acknowledgments

This project builds upon the initial template provided by the [Hugging Face Agents Course Template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template/tree/main) repository.