import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import List, Any
from warnings import warn

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/path/to/huggingface")

import openai
from vllm import LLM, SamplingParams

from evalplus.gen.util.api_request import make_auto_request

EOS = ["<|endoftext|>", "<|endofmask|>", "</s>"]


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        conversational: bool = False,
        max_conversational_new_tokens: int = 1024,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = (
            max_conversational_new_tokens if conversational else max_new_tokens
        )
        self.conversational = conversational

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass
    
    def score_logprob(
        self, generations: List[str], num_samples: int = 200
    ) -> List[Any]:
        pass
    
    def construct_instruction(
        self, prompt: str,
    ) -> str:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", "1")),
            "enforce_eager": True,
            }
        if "CodeLlama" in name:
            kwargs["dtype"] = "bfloat16"
        elif "WizardCoder" in name:
            kwargs["dtype"] = "float16"
        elif "deepseek" in name:
            kwargs["dtype"] = "bfloat16"

        # before we set 4096, I set it to 2048.
        # okay, I set it to 4096.
        # I change trust_remote_code to True bc I want to use DeepSeekV2 models.
        print(f"kwargs = {kwargs}")
        self.llm = LLM(model=name, max_model_len=4096, trust_remote_code=True, **kwargs)

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs
    
    def score_logprob(
        self, prompt: List[str], num_samples: int = 200
    ) -> List[Any]:
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            prompt[:batch_size],
            SamplingParams(
                temperature=0,
                max_tokens=1,
                top_p=1.0,
                stop=self.eos,
                prompt_logprobs=0,
            ),
            use_tqdm=False,
        )

        return [x.prompt_logprobs for x in vllm_outputs]


class CodeLlamaInstruct70B(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""'<s>Source: system

 You are a helpful and honest code assistant expert in Python. Please, provide all answers to programming questions in Python.
 <step> Source: user

 Provide a self-contained Python script that solves the following problem:
```python
{prompt}
```
 <step> Source: assistant

 Here is a Python script that solves the problem:
```python
"""

        return VLlmDecoder.codegen(self, input, do_sample, num_samples)
    
    def construct_instruction(self, prompt: str) -> str:
        input = f"""'<s>Source: system

 You are a helpful and honest code assistant expert in Python. Please, provide all answers to programming questions in Python.
 <step> Source: user

 Provide a self-contained Python script that solves the following problem:
```python
{prompt}
```
 <step> Source: assistant

 Here is a Python script that solves the problem:
```python
"""

        return input


class CodeLlamaInstructSmall(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
```python
{prompt}
```
[/INST]
```python
"""

        return VLlmDecoder.codegen(self, input, do_sample, num_samples)
    
    def construct_instruction(self, prompt: str) -> str:
        input = f"""[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
```python
{prompt}
```
[/INST]
```python
"""

        return input


class DeepSeekInstruct(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Please complete the following Python function in a markdown style code block:
```python
{prompt}
```
### Response:
```python
"""

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)
    
    def construct_instruction(self, prompt: str) -> str:
        prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Please complete the following Python function in a markdown style code block:
```python
{prompt}
```
### Response:
```python
"""
        return prompt


class QwenInstruct(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
### Instruction:
Please complete the following Python function in a markdown style code block:
```python
{prompt}
```
### Response:
```python
"""

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)
    
    def construct_instruction(self, prompt: str) -> str:
        prompt = f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
### Instruction:
Please complete the following Python function in a markdown style code block:
```python
{prompt}
```
### Response:
```python
"""
        return prompt


class Alpaca(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes request.

### Instruction:
Create a Python script for this problem:
{prompt}

### Response:
```python
"""

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)
    
    def construct_instruction(self, prompt: str) -> str:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes request.

### Instruction:
Create a Python script for this problem:
{prompt}

### Response:
```python
"""
        return prompt


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI()

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        # construct prompt
        fmt = "json_object" if self.name == "gpt-4-1106-preview" else "text"
        if fmt == "json_object":
            message = r'Please complete the following code snippet by generating JSON like {"code": ""}'
        else:
            message = r"Please generate code to complete the following problem:"

        message += f"\n```python\n{prompt.strip()}\n```\nSure, here is the code to complete the given problem:```python"

        ret = make_auto_request(
            self.client,
            message=message,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            response_format={"type": fmt},
        )

        outputs = []
        for item in ret.choices:
            content = item.message.content
            # if json serializable
            if fmt == "json_object":
                try:
                    json_data = json.loads(content)
                    if json_data.get("code", None) is not None:
                        outputs.append(prompt + "\n" + json_data["code"])
                        continue

                    print(f"'code' field not found in: {json_data}")
                except Exception as e:
                    print(e)
            outputs.append(content)

        return outputs


def make_model(name: str, batch_size: int = 1, temperature: float = 0.8):
    if name.startswith("gpt-3.5-") or name.startswith("gpt-4-"):
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("code-llama-"):
        if name.endswith("instruct"):
            nb = name.split("-")[2]
            assert nb.endswith("b")
            if nb == "70b":
                return CodeLlamaInstruct70B(
                    batch_size=batch_size,
                    name=f"codellama/CodeLlama-70B-Instruct-hf",
                    temperature=temperature,
                    conversational=True,
                )
            else:
                return CodeLlamaInstructSmall(
                    batch_size=batch_size,
                    name=f"codellama/CodeLlama-{nb}-Instruct-hf",
                    temperature=temperature,
                    conversational=True,
                )
        assert name.endswith("b")
        nb = name.split("-")[-1]
        return VLlmDecoder(
            batch_size=batch_size,
            name=f"codellama/CodeLlama-{nb}-Python-hf",
            temperature=temperature,
        )
    elif name == "deepseek-coder-16b-instruct":
        return DeepSeekInstruct(
            batch_size=batch_size,
            name=f"deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("deepseek-coder"):
        import re

        # format deepseek-coder-{nb}b-{base|instruct}-[v{version}]
        pattern = re.compile(r"deepseek-coder-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = float(matches[0])
        if nb.is_integer():
            nb = int(nb)

        if "instruct" in name:
            # if version is specified, use it
            version = matches[1].split("-")[-1]
            version_suffix = f"-{version}" if version.startswith("v") else ""
            return DeepSeekInstruct(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-instruct{version_suffix}",
                temperature=temperature,
                conversational=True,
            )
        else:
            version = matches[1].split("-")[-1]
            version_suffix = f"-{version}" if version.startswith("v") else ""
            return VLlmDecoder(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-base{version_suffix}",
                temperature=temperature,
            )
    elif name.startswith("qwen"):
        import re

        assert "2.5" in name, "We only consider Qwen 2.5 for now"
        # format qwen2.5-coder-{nb}b-{|instruct}
        nb = name.split("-")[2]
        assert nb.endswith("b")
        nb.replace("b", "B")

        if "instruct" in name:
            return QwenInstruct(
                batch_size=batch_size,
                name=f"Qwen/Qwen2.5-Coder-{nb}-Instruct",
                temperature=temperature,
                conversational=True,
            )
        else:
            raise ValueError(f"We currently do not support base models for Qwen2.5")
    elif name == "wizardcoder-34b":
        return Alpaca(
            batch_size=batch_size,
            name="WizardLMTeam/WizardCoder-Python-34B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-15b":
        return Alpaca(
            batch_size=batch_size,
            name="WizardLMTeam/WizardCoder-15B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-13b":
        return Alpaca(
            batch_size=batch_size,
            name="WizardLMTeam/WizardCoder-Python-13B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-7b":
        return Alpaca(
            batch_size=batch_size,
            name="WizardLMTeam/WizardCoder-Python-7B-V1.0",
            temperature=temperature,
            conversational=True,
        )

    raise ValueError(f"Invalid model name: {name}")
