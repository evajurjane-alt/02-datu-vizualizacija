import requests
import time
import logging
import os
import sys

app_location = os.path.dirname(os.path.abspath(__file__))

def file_path_resolver(file_path: str, is_parent: bool = False, sub_folder: str = None):
    if type(file_path).__name__ == 'StringIO':
        file_path.seek(0)
    else:
        dir_path = app_location
        if is_parent: dir_path = os.path.dirname(dir_path)
        if sub_folder is not None: dir_path = os.path.join(dir_path, sub_folder)
        file_path = os.path.join(dir_path, file_path)
    return file_path

def logtofile(message, level: str = 'info', mode: str = 'a'):
    # For mode 'a' to append and 'w' to overwrite the log file.
    logfile_path = file_path_resolver('app.log')
    max_message_length = 1000 
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_handler = logging.FileHandler(logfile_path, mode=mode)
    console_handler = logging.StreamHandler(sys.stdout)
    message = str(message)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[file_handler, console_handler]
    )
    if len(message) > max_message_length:
        message = message[:max_message_length] + '... [truncated]'
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)

def ask_gemini_with_fallback(prompt: str, api_key: str) -> str:
    """
    Queries the Gemini API using standard HTTP requests.
    Includes rate-limit fallbacks across free models and retries for unexpected server errors.
    """
    
    # Ordered list of free/preview Gemini models, from most to least capable
    models = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    max_retries = 3
    backoff_factor = 2 # Seconds to multiply wait time by on each retry
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                
                # 1. Success: Extract and return the text
                if response.status_code == 200:
                    data = response.json()
                    try:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    except (KeyError, IndexError) as e:
                        logtofile(f"Error parsing response from {model}: {e}", level='error')
                        return
                
                # 2. Rate Limited / Quota Exceeded: Stop retrying this model, fallback to next
                elif response.status_code == 429 or response.status_code == 403:
                    logtofile(f"[{model}] Quota exceeded (429) or invalid API key (403). Falling back to the next model...", level='warning')
                    break 
                
                # 3. Unexpected Server Errors: Wait and retry
                elif response.status_code >= 500:
                    logtofile(f"[{model}] Server error ({response.status_code}). Retrying {attempt + 1}/{max_retries}...", level='warning')
                    time.sleep(backoff_factor ** attempt)
                    continue
                
                # 4. Fatal Client Errors (e.g., 400 Bad Request): Abort entirely
                else:
                    logtofile(f"[{model}] Fatal error ({response.status_code}): {response.text}", level='error')
                    return
                    
            except requests.exceptions.RequestException as e:
                # 5. Network/Timeout Errors: Wait and retry
                logtofile(f"[{model}] Network error: {e}. Retrying {attempt + 1}/{max_retries}...", level='warning')
                time.sleep(backoff_factor ** attempt)
                continue
                
        logtofile(f"--- Exhausted options for {model}, moving down the list. ---", level='info')

    # If the loop finishes without returning, all models failed or hit limits
    logtofile("All available free models failed or exhausted their quota.", level='error')
    return

# --- Usage Example ---
if __name__ == "__main__":
    API_KEY = "KEY_HERE"
    PROMPT = "PROMPT_HERE"
    
    result = ask_gemini_with_fallback(PROMPT, API_KEY)
    print("\nFinal Output:\n")
    print(result)
