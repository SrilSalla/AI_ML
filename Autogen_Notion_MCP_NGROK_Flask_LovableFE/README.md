# Autogen Notion MCP NGROK Flask LovableFE

## Overview
This project integrates Notion's MCP (Model Context Protocol) with OpenAI and Flask, providing a backend API that can create and manage Notion pages using natural language tasks. It also exposes endpoints via Flask, with ngrok support for public tunneling, and is ready for frontend integration.

## Features
- Create and manage Notion pages using natural language tasks
- Integrates with OpenAI for LLM-powered task execution
- Flask API with CORS enabled
- ngrok integration for public URL exposure
- Secure secret management using a `.env` file

## Requirements
- Python 3.8+
- Node.js (for npx and mcp-remote)
- Notion API integration (API key)
- OpenAI API key

## Setup

### 1. Clone the repository
```sh
git clone https://github.com/SrilSalla/AI_ML.git
cd AI_ML/Autogen_Notion_MCP_NGROK_Flask_LovableFE
```

### 2. Create and activate a virtual environment
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Install Node.js
Download and install Node.js from [https://nodejs.org/](https://nodejs.org/). Ensure `npx` is available in your PATH.

### 5. Set up environment variables
Create a `.env` file in the project root with the following content:
```
OPENAI_API_KEY=your_openai_api_key
NOTION_API_KEY=your_notion_api_key
```

### 6. Run the Flask server
```sh
python final.py
```

## API Endpoints

### Health Check
```
GET /health
```
Response:
```json
{"status": "ok", "message": "Notion MCP Flask App is live"}
```

### Root
```
GET /
```
Response:
```json
{"message": "MCP Notion app is live, use /health or /run to work "}
```

### Run a Task
```
POST /run
Content-Type: application/json
Body: {"task": "create a page 'final'"}
```
Example (PowerShell):
```powershell
Invoke-WebRequest -Uri http://localhost:7001/run -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"task": "create a page ''final''"}'
```

## Notion Page ID
To create a page under a specific parent, provide the Notion page ID in your task, e.g.:
```
{"task": "Create a new page titled 'PageFromMCINotion' under parent page with ID 'YOUR_PAGE_ID'"}
```
Find your Notion page ID by copying the last part of the page URL (remove dashes).

## Troubleshooting
- If you see `[WinError 193] %1 is not a valid Win32 application`, ensure you use `npx.cmd` on Windows in your Python code.
- Make sure all environment variables are set and valid.
- Ensure Node.js and npx are installed and available in your PATH.
- For push errors to GitHub, use a feature branch and open a pull request.

## License
This project is for educational and demonstration purposes.


## Create Front-End using Lovable, below is the prompt

https://5b6f09e31949.ngrok-free.app

# curl -X POST http://localhost:7001/run \
#   -H "Content-Type: application/json" \
#   -d '{"task": "create a page 'HelloFromLove' "}'

Can you create a solid quick Front end in which we are having a chatbot to connect to my notion page, further also fields and buton to test out my end points on above give url of ngrok 

/health
/
/run(I have given you the curl for this which I could be able to edit.
