from .BaseModel import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque, defaultdict
import torch
import os
import re
from utils import extract_code_from_completion, truncate_completion, remove_comments

class Codegen25Instruct(BaseModel):
        
    def __init__(self, api_key=None) -> None:
        self.iteration = 0
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-instruct_P", device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-instruct_P", device_map="auto", trust_remote_code=True)
        self.ignore_lang_for_code_extraction = False

    def check_and_implement_scp(self, scp, scenario, current_code, output_scenario_dir, lang, relevance_threshold):
        
        practice = scp["practice"]
        scp_id = scp["id"]

        ## Rate first
#         relevance_prompt = f"""```{lang}\n{current_code}\n```\n\nFor the code above, consider the Secure Coding Practice "{practice}" and rate its applicability and relevance to the code. The ratings should be one of 1 (Not relevant), 2 (Slightly relevant), 3 (Moderately relevant), 4 (Relevant) or 5 (Very relevant). Give an explanation after your rating. Your output should be in the following format:
# Rating: [1-5]
# Explanation: [Your explanation here]

# ### Response:
# """
        commented_scenario = ""
        # if lang == 'py':
        #     commented_scenario = "# " + scenario
        # else:
        #     commented_scenario = "// " + scenario
        relevance_and_revision_prompt = f"""```\n{current_code}\n```\n\nThe code above implements the prompt, "{scenario}". Consider the Secure Coding Practice (SCP) "{practice}". Explain your thought process for why the Practice is or is not relevant/applicable to the prompt and code. Then give a relevance rating of 1 (Not relevant), 2 (Slightly relevant), 3 (Moderately relevant), 4 (Relevant) or 5 (Very relevant). Then, if it is found to be relevant and is not yet implemented, integrate the SCP into the code. Do not add new functionality to the code. If there are changes in the code, ensure that the original prompt is still addressed. If there are no changes then, under the "Code" section in the format below write "No changes". Your output should be in the following format:\nExplanation of relevancy and applicability of SCP: [Your explanation here]\nRelevancy Score: [Relevancy Score 1-5]\nUpdated New Code: [Your Updated Code]"""

        scp_relevancy_and_revised_code = self._generate(relevance_and_revision_prompt, min_new_tokens=600, max_new_tokens=800)
        scp_relevancy_and_revised_code = scp_relevancy_and_revised_code.split("### Response:")
        if (len(scp_relevancy_and_revised_code)<=1):
            return -1, current_code # Retry because it cannot be parsed correctly
        scp_relevancy_and_revised_code = scp_relevancy_and_revised_code[1]
        pattern = r"Relevan.*?:\s*\**(\d+)"
        matches = re.findall(pattern, scp_relevancy_and_revised_code, re.DOTALL)
        relevance_score = -1
        if not len(matches):
            # Use a regular expression to find the text after "Relevance:" until the new line
            match = re.search(r'Relevan.*?:\s*(.*)', scp_relevancy_and_revised_code)
            if match:
                result = match.group(1)
                if "not relevant" in result.lower():
                    relevance_score = 1
                elif "very relevant" in result.lower():
                    relevance_score = 5
                elif "moderately relevant" in result.lower():
                    relevance_score = 3
                elif "slighlty relevant" in result.lower():
                    relevance_score = 2
                elif "relevant" in result.lower():
                    relevance_score = 4
            else:
                return -1, current_code # Retry
        else:
            relevance_score = int(matches[0])
        print("THE RELEVANCY SCORE IS", relevance_score)
        self.scp_counter+=1
        scp_counter = self.scp_counter
        with open(os.path.join(output_scenario_dir, f"iteration_{scp_counter} (SCP_{scp_id}).txt"), 'w') as f:
            f.write(relevance_and_revision_prompt+"\n\n"+scp_relevancy_and_revised_code)
       
        
        if relevance_score >= relevance_threshold:
            # pattern = r"nUpdated New Code: (.*)"
            # match = re.search(pattern, scp_relevancy_and_revised_code, re.DOTALL)
            # if match:
            #     scp_relevancy_and_revised_code = match.group(1).strip()
            # else:
            #     return -1, current_code # Retry
            raw_code = extract_code_from_completion(scp_relevancy_and_revised_code, lang)
            raw_code = remove_comments(raw_code, lang)
            if "no changes" not in scp_relevancy_and_revised_code.lower() and len(raw_code.replace(' ', '')):
                current_code = raw_code
            return 1, current_code
    
        return 0, current_code
    
    def _scp_guided_generation(self, scenario, output_scenario_dir, lang, relevance_threshold=3, num_return_sequences=25):
        

        # Save outputs for verification
        output_scenario_dir_original = output_scenario_dir
        """
            Step 1: First we get the standard model's answer
        """
     
        vanilla_plan_and_codes =  self._generate(scenario, num_return_sequences) #messages, num_return_sequences=25, max_new_tokens=300, system_prompt=None):

        # TODO: Check that this is a list
        completions = []
        for i in range(len(vanilla_plan_and_codes)):
            output_scenario_dir = os.path.join(output_scenario_dir_original, "tree_search_logs", "iteration_"+str(i))
            vanilla_plan_and_code = vanilla_plan_and_codes[i]
            os.makedirs(output_scenario_dir)
            if self.tokenizer.eos_token in vanilla_plan_and_code:
                vanilla_plan_and_code = vanilla_plan_and_code[:vanilla_plan_and_code.find(self.tokenizer.eos_token)]
            vanilla_code = truncate_completion(vanilla_plan_and_code, lang)
            vanilla_code = scenario + vanilla_code        
            current_code = vanilla_code

            with open(os.path.join(output_scenario_dir, f"vanilla_code.txt"), 'w') as f:
                f.write(vanilla_code)

            # We need a description of the code since this is a code completion prompt

            prompt_get_description = f"""
```
{vanilla_code}
```

Write a short description of the code above. Then write a instruction prompt that would generate this code. Do not use code in your prompt. Finally, output the original code. Your output should be in the following format:
Description: [ Your description here ]
Prompt: [ Your prompt here ]
Code: [ The original code here ]
"""
            description = self._generate(prompt_get_description)
            with open(os.path.join(output_scenario_dir, f"vanilla_description.txt"), 'w') as f:
                f.write(description)
                
            # Define the regex pattern to match the "Prompt" section
            pattern = r"Prompt:\s*(.*?)(?:\n[A-Z][a-z]+:|\Z)"

            # Search for the pattern in the text
            match = re.search(pattern, description, re.DOTALL)
            prompt_section = ""
            # Extract the "Prompt" section if a match is found
            if match:
                prompt_section = match.group(1).strip()
                print("Prompt Section:")
                print(prompt_section)
            else:
                print("Prompt section not found.")
                continue
        
            """
                Step 2: Now, we perform a BFS of the graph and iteratively implement the SCPs
            """
            
            scps = self._convert_to_kv_format()
            incoming_edges = defaultdict(set)
            visited = set([0])

            for node_id, node_data in scps.items():
                for child in node_data["children"]:
                    incoming_edges[child].add(node_id)
            
          
            self.get_longest_paths(scps, incoming_edges)
            
            next_nodes = []
            for s in scps[0]["children"]:
                next_nodes.append(scps[s])
            queue = deque(next_nodes)
            
            self.scp_counter = 0
            while queue:
                queue = deque(sorted(queue, key=lambda x: x["depth"]))

                scp = queue.popleft()

                print("Processing node ", scp)
                
                if scp["id"] in visited:
                    continue

                # scp = scps[current]
                is_relevant, raw_code = self.check_and_implement_scp(scp, prompt_section, current_code, output_scenario_dir, lang, relevance_threshold)
                
                visited.add(scp["id"])
                if (is_relevant==-1):
                    next_nodes = []
                    for s in scps[scp["id"]]["children"]:
                        next_nodes.append(scps[s])
                    queue.extend(next_nodes)
                    continue
                
                current_code = raw_code

                if is_relevant==1:
                    next_nodes = []
                    for s in scps[scp["id"]]["children"]:
                        next_nodes.append(scps[s])
                    queue.extend(next_nodes)

            """
            Step 3: Once we get the code, there may be too many changes from the iterative process. We simply ask the LLM
            to rectify any potential bugs
            """


            bug_fix_messages = f"```\n{current_code}\n```" + "\n Are there any bugs in this code? Fix them."

            raw_code = self._generate(bug_fix_messages)
            with open(os.path.join(output_scenario_dir, f"final_output.txt"), 'w') as f:
                f.write(raw_code)
            current_code = extract_code_from_completion(raw_code, lang)
            completions.append(current_code)
        return completions
    
    def sample(self, scenario=None, num_return_sequences=25, lang=None, history=None, prompt_type=None, output_scenario_dir=None):
        
        completions, messages = [], []
          

        if (prompt_type=="SCPGoT"):
            completions = self._scp_guided_generation(scenario, output_scenario_dir, lang, num_return_sequences=num_return_sequences)
        else:
            completion = self._generate(messages, num_return_sequences=1)
            completions.append(completion)

        return completions

    def _generate(self, messages, num_return_sequences=1, min_new_tokens=300, max_new_tokens=300, system_prompt=None):
        
        print("INPUT")
        print(messages)
        device = torch.cuda.current_device()
        inputs = self.tokenizer(messages, return_tensors="pt").to(device)
        generated_tokens = self.model.generate(inputs.input_ids, temperature=0.4, do_sample=True,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,top_p=0.95, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)
        alltokens = self.tokenizer.batch_decode(generated_tokens)[0]
        tokens = generated_tokens[:, inputs.input_ids.shape[1]:, ...]
        if (num_return_sequences==1):
            completions = self.tokenizer.batch_decode(tokens)[0]
        else:
            completions = self.tokenizer.batch_decode(tokens)
        # Get answer
        # completion = completion.split('### Response')[1]
        print("OUTPUT ")
        print(completions)
        print("\n\n**********************\n\n")
        return completions
