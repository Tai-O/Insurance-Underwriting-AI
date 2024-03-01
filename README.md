# Insurance-Underwriting-AI

PoC to show how we can use LLM's to analyze risk on an Insurance application form using Insurance underwriting guidelines. Built using Streamlit

1. Steps involved

- Provide an Underwriting Guideline along with Insurance application Form
- Text Summarization model summarizes Underwriting guidelines document
- Summarized Underwriting guidelines along with Insurance application form are used to query GPT 3.5 turbo model
- Model returns the highlighted risks on the Insurance application

2. Get Open AI API KEY
- Create an account on https://platform.openai.com/api-keys and get an API KEY (Needed to access GPT 3.5 turbo model)
- Open command prompt
- ```bash
    setx OPENAI_API_KEY "your-api-key-here"
    ```
- Permanent setup: To make the setup permanent, add the variable through the system properties as follows:

Right-click on 'This PC' or 'My Computer' and select 'Properties'.
- Click on 'Advanced system settings'.
- Click the 'Environment Variables' button.
- In the 'System variables' section, click 'New...' and enter OPENAI_API_KEY as the variable name and your API key as the variable value.
- Verification: To verify the setup, reopen the command prompt and type the command below. It should display your API key: echo %OPENAI_API_KEY%
 
3. Install Requiremnents:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the code:
    ```bash
    streamlit run app.py
    ```


