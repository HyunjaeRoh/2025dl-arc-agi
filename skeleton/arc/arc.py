from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import message_templates
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

import os
import json
from tqdm.auto import tqdm


class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None, is_training=False):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
            is_training (bool): mode of training (set use_cache option for model)
        """
        # config_path = "artifacts/config/config.yml" @@ TODO: search config_setting
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )

        # Load pre-trained model
        use_cache = False if is_training else True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allow the model to use custom code from the repository
            quantization_config=bnb_config,  # Apply the 4-bit quantization configuration
            attn_implementation='sdpa',  # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16,  # Set the data type for the model
            use_cache=use_cache,  # at training, disable caching to save memory
            device_map='auto',  # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        # Load tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        # Define token IDs for ARC grid and pixels (0-10) and row seperator
        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.separator = self.tokenizer.encode('\n', add_special_tokens=False)[0]

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = self.model.device

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for pixel in row:
                ids.append(self.pixel_ids[pixel])
            ids.append(self.separator)
        return ids

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}

        for idx in ids:
            if idx == self.separator:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid


    def format_prompt(self, datapoint, result_type="grid"):

        """
                Args:
                    datapoint (dict): contains training data, test input
                    result_type (str): 'grid' or 'rule_explanation'

                Returns:
                    prompt (dict): dictionary that contains input ids and additional information
                """
        # Get input data for prompt
        examples = datapoint['examples']
        test_input = datapoint['test'][0]['input']

        # Define prompt templates
        system_prompt = message_templates["system_prompt"]

        user_message_template1 = message_templates["user_message_template1"]
        user_message_template2 = message_templates["user_message_template2"]

        # Tokenize system message
        sys = self.tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt + "<|eot_id|>",
            add_special_tokens=False)

        # Tokenize user message
        user = self.tokenizer.encode(
            "<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n",
            add_special_tokens=False)
        input_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        output_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for example in examples:
            input_listform = example['input']
            output_listform = example['output']
            input_tokenized = self.format_grid(input_listform)
            output_tokenized = self.format_grid(output_listform)

            user += input_desc
            user += input_tokenized
            user += output_desc
            user += output_tokenized
            user += self.tokenizer.encode("\n", add_special_tokens=False)

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += input_desc
        user += self.format_grid(test_input)

        if result_type == "grid":
            user_message_template3 = message_templates["user_message_template3"]
        if result_type == "rule_explanation":
            user_message_template3 = message_templates["user_message_template4"]
        else:
            raise Exception("result_type must be 'grid' or 'rule_explanation'")

        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)

        # Form all message
        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                                      add_special_tokens=False)
        messages += assis

        return {
            "message_tokens": messages,
            "input": test_input,
            "examples": examples
        }

    def train(self, training_data_path="/workspace/dataset", output_dir: str = "artifacts/arc_solver_finetuned"):
        num_task_files_to_load = 10
        num_max_train_for_each_task = 10

        ## 1. prepare dataset
        processed_data = []

        task_file_names = [f for f in os.listdir(training_data_path)]
        task_file_names = task_file_names[:num_task_files_to_load]

        print(f"학습을 위해 {len(task_file_names)}개의 ARC 작업 파일을 로드하여 처리합니다...")

        ## 1-2. read each task (read each json file)
        for task_file_name in tqdm(task_file_names, desc="loop for each task"):
            print(f"Processing task: {task_file_name}")

            task_file_path = os.path.join(training_data_path, task_file_name)
            with open(task_file_path, "r") as f:
                loaded_task_data_list = json.load(f)

            # 1-3. parse each data and generate data for processing
            num_pairs = min(len(loaded_task_data_list) // 4, num_max_train_for_each_task)
            for i in range(num_pairs):
                train_examples_for_prompt = loaded_task_data_list[i * 4: (i + 1) * 4 - 1]
                test_example = loaded_task_data_list[(i + 1) * 4 - 1]

                datapoint = {
                    "train": train_examples_for_prompt,
                    "test": [test_example]
                }

                prompt_info = self.format_prompt(datapoint)
                prompt_tokens = prompt_info["input_ids"]
                target_grid = test_example["output"]
                target_tokens = self.format_grid(target_grid)

                try:
                    eot_token_id = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
                except Exception as e:
                    print(f"Warning: Could not encode <|eot_id|>: {e}. Using general eos_token_id.")
                    eot_token_id = self.tokenizer.eos_token_id

                full_tokens = prompt_tokens + target_tokens + [eot_token_id]
                labels = [-100] * len(prompt_tokens) + target_tokens + [eot_token_id]

                processed_data.append({
                    "input_ids": torch.tensor(full_tokens, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

        ## 2. PEFT setting

        dataset = Dataset.from_list(processed_data)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        try:
            compute_dtype = self.model.config.quantization_config.bnb_4bit_compute_dtype
        except AttributeError:  # quantization_config가 없을 경우 대비
            compute_dtype = torch.float16  # 기본값
            print(
                "Warning: bnb_4bit_compute_dtype not found in model.config.quantization_config. Defaulting to torch.float16 for TrainingArguments.")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
        )

        if compute_dtype == torch.float16:
            training_args.fp16 = True
        elif compute_dtype == torch.bfloat16:
            training_args.bf16 = True

        trainer = SFTTrainer(
            model=self.model,
            # tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            # max_seq_length=effective_max_seq_length,
        )

        print("미세 조정을 시작합니다...")
        trainer.train()

        final_adapter_path = os.path.join(output_dir, "checkpoint-final")
        print(f"최종 어댑터를 {final_adapter_path}에 저장합니다.")
        self.model.save_pretrained(final_adapter_path)
        self.tokenizer.save_pretrained(final_adapter_path)

        print(f"학습 완료. 어댑터가 {final_adapter_path}에 저장되었습니다.")
        print(f"이제 `prepare_evaluation(adapter_path='{final_adapter_path}')`를 사용하여 이 어댑터를 로드할 수 있습니다.")

    def predict(self, examples, test_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            test_input (List[List[int]]): A 2d grid,
                which is an input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "examples": examples,
            "test": [
                {
                    "input": test_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt["message_tokens"], dtype=torch.long).to(self.device).view(1, -1)

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150,
        )

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()

        output = output[N_prompt:].tolist()
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
            grid = grid[:x, :y]

        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self, adapter_path="artifacts/checkpoint-final"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter(adapter_path)
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()
