'''
./medagentaudit/core/base_agent.py
'''
from typing import Dict, Any, Tuple
import time
from openai import OpenAI
import sys
from pathlib import Path
import json
current_file_path = Path(__file__).resolve()
utils_root = current_file_path.parents[1] / "utils"
sys.path.append(str(utils_root))
from config_loader import get_config
class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
                 agent_id: str,
                 agent_type,
                 config_path,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor or Coordinator)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []

        self.llm = get_config(config_path, active_llm=model_key).llm

        self.client = OpenAI(
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
            timeout = self.llm.timeout, # if time out then atonomously report error
        )
        self.model_name = self.llm.model_name
    def call_llm(self,
                 system_message: Dict[str, str],
                 user_message: Dict[str, Any],
                 max_retries: int = 3, 
                 response_format : Dict | None = None) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing question and optional image
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple containing:
            - LLM response text
            - The system message sent to the LLM
            - The user message sent to the LLM
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...")
                print(f'the llm model name is {self.model_name}')
                request_kwargs = {
                    "model": self.model_name,
                    "stream": self.llm.stream,
                    "timeout": self.llm.timeout,
                }

                if self.model_key in ['gpt-5.2-high','o3','gpt-5.2']: # specifically for openai close source model
                    request_kwargs["messages"] = [system_message, user_message]
                    request_kwargs['response_format'] = response_format
                    request_kwargs["reasoning_effort"] = getattr(self.llm, 'reasoning_effort', 'medium')
                
                elif 'gemini' in self.model_key.lower():
                    # Gemini Special Handling
                    request_kwargs["messages"] = [system_message, user_message]
                    request_kwargs['response_format'] = response_format
                    thinking_config = {"include_thoughts": True}
                    
                    raw_effort = getattr(self.llm, 'reasoning_effort', "high")
                    
                    if "2.5" in self.model_name:
                        # Gemini 2.5 use thinking_budget (int)
                        if isinstance(raw_effort, int):
                            thinking_config["thinking_budget"] = raw_effort
                        elif raw_effort == "low":
                            thinking_config["thinking_budget"] = 1024
                        elif raw_effort == "medium":
                            thinking_config["thinking_budget"] = 8192
                        elif raw_effort == "high":
                            thinking_config["thinking_budget"] = 24576
                        else:
                            # Default to dynamic thinking (-1)
                            thinking_config["thinking_budget"] = -1 
                    else:
                        # Gemini 3 use thinking_level (str)
                        thinking_config["thinking_level"] = raw_effort if isinstance(raw_effort, str) else "high"

                    request_kwargs["extra_body"] = {
                        "google": {
                            "thinking_config": thinking_config
                        }
                    }
                
                else: # qwen
                    request_kwargs["messages"] = [system_message, user_message]
                    request_kwargs['response_format'] = response_format
                    # Qwen / Default handling
                    request_kwargs["extra_body"] = {"enable_thinking": True}

                completion = self.client.chat.completions.create(**request_kwargs)

                if not self.llm.stream:
                    data_dict = completion.model_dump()
                    print(json.dumps(data_dict, indent=2, ensure_ascii=False)) # Debug usage
                    message = completion.choices[0].message
                    response = message.content
                    reasoning_content = getattr(message, 'reasoning_content', None)
                else:
                    response_chunks = []
                    reasoning_chunks = []
                    for chunk in completion:
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            response_chunks.append(delta.content)
                        # defensive programming for reasoning content
                        r_content = getattr(delta, "reasoning_content", None)
                        if r_content is not None:
                            reasoning_chunks.append(r_content)
                    response = "".join(response_chunks)
                    reasoning_content = "".join(reasoning_chunks)

                # check if the response is empty
                if not response.strip():
                    raise ValueError("Empty response received from LLM")
                
                if reasoning_content is not None and len(reasoning_content) > 50:
                    print(f"Agent {self.agent_id} received reasoning: {reasoning_content[:50]}...")

                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response, reasoning_content, system_message, user_message
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    # don't return error message ,just raise exception
                    raise RuntimeError(f"CRITICAL: Agent {self.agent_id} failed after {max_retries} attempts. Reason: {str(e)}")
                time.sleep(1)