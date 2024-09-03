# Basic LLM node that calls for a Large Language Model for completion.
import os

import openai

from nodes.node import Node
from nodes.node_cofig import *
from utils.util import *
# from alpaca.lora import AlpacaLora

from llm.chat_llm import ChatLLM

# openai.api_key = os.environ["OPENAI_API_KEY"]


class LLMNode(Node):
    def __init__(self, name="BaseLLMNode", model_name="text-davinci-003", stop=None, input_type=str, output_type=str):
        super().__init__(name, input_type, output_type)
        self.model_name = model_name
        self.stop = stop

        # Initialize to load shards only once
        # if self.model_name in LLAMA_WEIGHTS:
        #     self.al = AlpacaLora(lora_weights=self.model_name)

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        response = self.call_llm(input, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def call_llm(self, prompt, stop):
        if self.model_name in LLM_COMPLETION_MODELS:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                top_p=LLM_CONFIG["top_p"],
                frequency_penalty=LLM_CONFIG["frequency_penalty"],
                presence_penalty=LLM_CONFIG["presence_penalty"],
                stop=stop
            )
            return {"input": prompt,
                    "output": response["choices"][0]["text"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"]}
        elif self.model_name in LLM_CHAT_MODELS:
            messages = [{"role": "user", "content": prompt}]
            # response = openai.ChatCompletion.create(
            #     model=self.model_name,
            #     messages=messages,
            #     temperature=LLM_CONFIG["temperature"],
            #     max_tokens=LLM_CONFIG["max_tokens"],
            #     top_p=LLM_CONFIG["top_p"],
            #     frequency_penalty=LLM_CONFIG["frequency_penalty"],
            #     presence_penalty=LLM_CONFIG["presence_penalty"],
            #     stop=stop
            # )

            chat_llm = ChatLLM(
                temperature=LLM_CONFIG["temperature"],
                api_key=ZHIPUAI_API_KEY,
                model=self.model_name
            )

            response = chat_llm.invoke(
                # model=self.model_name,
                input=messages,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                top_p=LLM_CONFIG["top_p"],
                # frequency_penalty=LLM_CONFIG["frequency_penalty"],
                # presence_penalty=LLM_CONFIG["presence_penalty"],
                stop=stop
            )

            return {"input": prompt,
                    # "output": response["choices"][0]["message"]["content"],
                    # "prompt_tokens": response["usage"]["prompt_tokens"],
                    # "completion_tokens": response["usage"]["completion_tokens"],
                    "output": response.content}
        elif self.model_name in LLAMA_WEIGHTS:
            instruction, input = prompt[0], prompt[1]
            output, prompt = self.al.lora_generate(instruction, input)
            return {"input": prompt,
                    "output": output,
                    "prompt_tokens": len(prompt)/4,
                    "completion_tokens": len(output)/4
            }

        else:
            raise ValueError("Model not supported")
