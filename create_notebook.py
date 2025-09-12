#!/usr/bin/env python3
"""
Script to convert the Python file to a well-structured Jupyter notebook
"""

import json
import re

def create_notebook_from_py():
    """
    Create a Jupyter notebook from the Python file with proper sections
    """
    
    # Read the Python file
    with open('langgraph_chatbot_poc.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Split content by markdown sections
    sections = re.split(r'# %% \[markdown\]\n# (.+)', content)
    
    # Add main title
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LangGraph Chatbot POC\n",
            "\n",
            "This notebook demonstrates a production-ready chatbot using LangGraph that:\n",
            "1. Takes user queries and maintains conversation context\n",
            "2. Determines intent and relevance using LLM\n",
            "3. Converts natural language to complex Snowflake SQL with JOINs\n",
            "4. Executes queries against real Snowflake database\n",
            "5. Formats responses back to natural language\n",
            "6. Handles follow-up questions intelligently\n",
            "\n",
            "## Key Features:\n",
            "- **Multi-Table Support**: Handles complex queries across multiple related tables\n",
            "- **Follow-up Questions**: Maintains conversation context for natural interactions\n",
            "- **Schema File Integration**: Loads table schemas from external file\n",
            "- **Production Ready**: Real database connections with proper error handling"
        ]
    })
    
    # Process sections
    for i in range(1, len(sections), 2):
        if i < len(sections):
            title = sections[i].strip()
            code_content = sections[i+1] if i+1 < len(sections) else ""
            
            # Add markdown cell for section title
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": f"## {title}"
            })
            
            # Clean up code content
            code_lines = code_content.split('\n')
            clean_code = []
            
            for line in code_lines:
                # Skip markdown comments and empty lines at start
                if line.strip().startswith('# %%') or (not clean_code and not line.strip()):
                    continue
                clean_code.append(line)
            
            # Remove trailing empty lines
            while clean_code and not clean_code[-1].strip():
                clean_code.pop()
            
            if clean_code:
                notebook["cells"].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": clean_code
                })
    
    # Write the notebook
    with open('langgraph_chatbot_poc.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Jupyter notebook created successfully!")
    print("📁 File: langgraph_chatbot_poc.ipynb")

if __name__ == "__main__":
    create_notebook_from_py()