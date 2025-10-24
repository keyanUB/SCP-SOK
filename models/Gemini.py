import time
from .BaseModel import BaseModel 
import google.generativeai as genai
import traceback

class GeminiPro(BaseModel):
        
    def __init__(self, api_key=None) -> None:
        genai.configure(api_key=api_key)
        safe = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safe)
        self.ignore_lang_for_code_extraction = False

    def _convert_to_gemini_history(self, history):

        gemini_history = []
        for message in history:
           
            if message["role"] == "user":
                gemini_history.append({
                    "parts": [message["content"]],
                    "role": "user"
                })
            elif message["role"] == "assistant":
                gemini_history.append({
                    "parts": [message["content"]],
                    "role": "model"
                })
            elif message["role"] == "system":
                gemini_history.extend([{
                    "parts": [message["content"]],
                    "role": "user"
                }, {
                    "parts": ["Understood"],
                    "role": "model"
                }])

        return gemini_history
      
    def _generate(self, messages, system_prompt=None, retry=0):
        
        # print("********************************************\n\nMESSAGES")
        # print(messages)
        # print("-------NEW--------")
        messages_gemini = self._convert_to_gemini_history(messages)
        # for message in messages_gemini:
        #     print("ROLE:", message["role"])
        #     print("CONTENT:", message["parts"][0])
        #     if (message["parts"][0].strip()==""):
        #         return ""
        try:
            completion = self.model.generate_content(messages_gemini)
            # print("OUTPUT ")
            # print(completion.text)
            if (completion):
                return completion.text
            return "Error"
        except Exception:
            traceback.print_exc()
            print("There was an error here")
            time.sleep(70)
            if retry<3:
                return self._generate(messages, system_prompt, retry=retry+1)
            return "```\nerror\n```"
