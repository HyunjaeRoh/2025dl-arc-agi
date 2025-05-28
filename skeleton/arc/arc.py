import random

from transformers import GenerationConfig
import torch
from typing import List, Tuple, Dict
import numpy as np
import re
import traceback
from rich import print as rich_print
from utils import render_grid

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, load_from_disk

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
            #attn_implementation='flash_attention_2',
            torch_dtype=torch.float16,  # Set the data type for the model
            use_cache=use_cache,  # at training, disable caching to save memory
            device_map='auto',  # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        # Load tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs for ARC grid and pixels (0-10) and row seperator
        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.separator = self.tokenizer.encode('\n', add_special_tokens=False)[0]

        self.rule_start_marker = "[[RULE]]"
        self.rule_end_marker = "[[/RULE]]"
        self.grid_start_marker = "[[GRID]]"
        self.grid_end_marker = "[[/GRID]]"

        self.rule_start_ids = self.tokenizer.encode(self.rule_start_marker, add_special_tokens=False)
        self.rule_end_ids = self.tokenizer.encode(self.rule_end_marker, add_special_tokens=False)
        self.grid_start_ids = self.tokenizer.encode(self.grid_start_marker, add_special_tokens=False)
        self.grid_end_ids = self.tokenizer.encode(self.grid_end_marker, add_special_tokens=False)
        self.eot_id = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

        try:
            # 사용 가능한 예약 토큰 중 하나를 선택합니다.
            self.placeholder_token_id = self.tokenizer.encode("<|reserved_special_token_1|>", add_special_tokens=False)[
                0]
        except:
            print("Warning: Could not find reserved special token. Using space ' ' as placeholder.")
            self.placeholder_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]

        self.placeholder_length = 50  # 플레이스홀더 길이 설정 (예: 75 토큰)
        self.space_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]  # 공백 토큰
        self.newline_id = self.separator  # '\n' 토큰 ID

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

    def parse_grid_from_string(self, grid_string: str) -> List[List[int]]:
        """Parses a grid from its string representation."""
        grid = []
        clean_grid_string = grid_string.strip()
        lines = clean_grid_string.split('\n')
        inv_map = {str(i): i for i in range(10)}

        for line in lines:
            line = line.strip()
            if not line: continue

            row = []
            for char in line:
                if char.isdigit():
                    row.append(inv_map.get(char, 0))
            if row:
                grid.append(row)

        return grid

    def format_grid_to_string(self, grid: List[List[int]]) -> str:
        return "\n".join("".join(map(str, row)) for row in grid)

    def format_prompt(self, datapoint, is_training=False):
        """
        Args:
            datapoint (dict): contains training data, test input

        Returns:
            prompt (dict): dictionary that contains input ids and additional information
        """
        message_templates = {
            "system_prompt": \
                '''You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus''',

            "user_message_template1": \
                '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
                ----------------------------------------''',

            "user_message_template2": \
                '''----------------------------------------
                Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
                ----------------------------------------''',

            "user_message_template3": \
                '''----------------------------------------
                What is the output grid? Think step by step and then provide the only output grid in the form as in the example input and output pairs. Do not provide any additional information:''',

            "user_message_template4": \
                '''----------------------------------------
                Considering the examples, please describe the transformation rule you observed in a single, concise English sentence. Think step by step.''',

            "user_message_template6": \
                '''----------------------------------------
                Think step by step. Provide the output grid enclosed in [[GRID]] and [[/GRID]] markers.
                Example: [[GRID]]000\n000\n000\n[[/GRID]] Do not provide any other information:''',

            "user_message_template5": \
                '''----------------------------------------
                Think step by step. First, describe the transformation rule you observed in a single, concise English sentence, enclosed in [[RULE]] and [[/RULE]] markers.
                Then, provide the output grid enclosed in [[GRID]] and [[/GRID]] markers.
                Example: [[RULE]]The rule is ... ... [[/RULE]] [[GRID]]000\n000\n000\n[[/GRID]]
                Do not provide any other information:''',

        }

        # Get input data for prompt
        train_data = datapoint['train']
        test_input = datapoint['test'][0]['input']

        # Define prompt templates
        system_prompt = message_templates["system_prompt"]

        # Tokenize system message
        sys = self.tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt + "<|eot_id|>",
            add_special_tokens=False)

        # Tokenize user message
        user_message_template1 = message_templates["user_message_template1"]
        user = self.tokenizer.encode(
            "<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n",
            add_special_tokens=False)
        input_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        output_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for example in train_data:
            input_listform = example['input']
            output_listform = example['output']
            input_tokenized = self.format_grid(input_listform)
            output_tokenized = self.format_grid(output_listform)

            user += input_desc
            user += self.grid_start_ids
            user += input_tokenized
            user += self.grid_end_ids
            user += output_desc
            user += self.grid_start_ids
            user += output_tokenized
            user += self.grid_end_ids
            user += self.tokenizer.encode("\n", add_special_tokens=False)

        user_message_template2 = message_templates["user_message_template2"]
        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += input_desc
        user += self.grid_start_ids
        user += self.format_grid(test_input)
        user += self.grid_end_ids

        user_message_template5 = message_templates["user_message_template5"]

        user += self.tokenizer.encode("\n" + user_message_template5, add_special_tokens=False)

        # Form all message
        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                                      add_special_tokens=False)
        messages += assis

        result = {
            "input_ids": messages,
            "input": test_input,
            "train": train_data
        }

        if is_training:
            test_output_grid = datapoint['test'][0]['output']

            placeholder_ids = [self.placeholder_token_id] * self.placeholder_length

            grid_content_ids = self.format_grid(test_output_grid)

            target_ids = []
            target_ids += self.rule_start_ids
            target_ids += placeholder_ids
            target_ids += self.rule_end_ids
            target_ids += [self.space_id]  # 마커 사이 공백
            target_ids += self.grid_start_ids
            target_ids += grid_content_ids  # 그리드 내용 (줄바꿈 포함)
            target_ids += self.grid_end_ids
            target_ids += [self.eot_id]  # 문장 종료

            input_ids = messages + target_ids

            labels = []
            labels += [-100] * len(messages)  # 프롬프트 마스킹
            labels += self.rule_start_ids  # [[RULE]] 학습
            labels += [-100] * self.placeholder_length  # 플레이스홀더 마스킹
            labels += self.rule_end_ids  # [[/RULE]] 학습
            labels += [-100]  # 공백 마스킹
            labels += self.grid_start_ids  # [[GRID]] 학습
            labels += grid_content_ids  # 그리드 내용 학습
            labels += self.grid_end_ids  # [[/GRID]] 학습
            labels += [self.eot_id]  # EOT 학습

            # 결과 딕셔너리에 추가
            result["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            result["labels"] = torch.tensor(labels, dtype=torch.long)
            result["target_grid"] = test_output_grid


        return result

    def predict(self, examples, test_input, is_training=False, get_details=False, ):
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
            is_training:
            get_details:
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": test_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long).to(self.device).view(1, -1)

        config = GenerationConfig(
            num_beams=4,
            # num_return_sequences=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            max_new_tokens=500,
        )

        attention_mask = torch.ones_like(input_ids).to(self.device)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()
        output_tokens = output[N_prompt:].tolist()

        total_outputs_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)

        rule_explanation = "Rule not found"

        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]
        grid = np.random.randint(0, 10, (x, y)).tolist()
        flags = {
            "no_rule": True,
            "no_grid": True,
            "exception": ""
        }

        try:
            rule_match = re.search(r"\[\[RULE\]\](.*?)\[\[/RULE\]\]", total_outputs_text, re.DOTALL)
            grid_match = re.search(r"\[\[GRID\]\](.*?)\[\[/GRID\]\]", total_outputs_text, re.DOTALL)

            if grid_match:
                grid_text = grid_match.group(1).strip()
                grid = self.parse_grid_from_string(grid_text)
                flags["no_grid"] = False
            if rule_match:
                rule_explanation = rule_match.group(1).strip()
                flags["no_rule"] = False

            if not get_details:
                return grid
            else:
                total_outputs = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                return grid, rule_explanation, flags, total_outputs

        except Exception as e:
            flags["exception"] = str(e)
            traceback.print_exc()
            if not get_details:
                return grid
            else:
                total_outputs = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                return grid, rule_explanation, flags, total_outputs

    def _prepare_training_data(self, training_data_path: str, task_indices=(0, 300), eval_ratio=0.1, window_size=4,
                               shuffle=True):
        """Loads and processes data, splitting into train/eval sets."""
        all_task_files = [f for f in os.listdir(training_data_path) if f.endswith('.json')]
        all_task_files.sort()
        selected_files = all_task_files[task_indices[0]:task_indices[1]]

        all_train_datapoints = []
        all_eval_datapoints = []

        print(f"Loading and processing {len(selected_files)} tasks...")
        for task_file in tqdm(selected_files, desc="Processing Tasks"):
            task_path = os.path.join(training_data_path, task_file)

            with open(task_path, "r") as f:
                all_examples = json.load(f)

            if len(all_examples) < window_size:
                print(f"Skipping {task_file}: Not enough examples ({len(all_examples)}) for window size {window_size}")
                continue
            print(f"Processing {task_file}...")

            split_idx = max(1, len(all_examples) - max(1, int(len(all_examples) * eval_ratio)))
            task_train = all_examples[:split_idx]
            task_eval = all_examples[split_idx:]

            for i in range(0, len(task_train) - window_size + 1):
                window = task_train[i: i + window_size]
                train_examples = window[:-1]
                test_example = window[-1]

                datapoint = {
                    "train": train_examples,
                    "test": [test_example],
                    "task_file": task_file
                }
                all_train_datapoints.append(datapoint)

            for i in range(0, len(task_eval) - 4 + 1):
                window = task_eval[i: i + window_size]
                train_examples = window[:-1]
                test_example = window[-1]
                datapoint = {
                    "train": train_examples,
                    "test": [test_example],
                    "task_file": task_file
                }
                all_eval_datapoints.append(datapoint)

        print(f"Prepared {len(all_train_datapoints)} training samples and {len(all_eval_datapoints)} evaluation samples.")
        if shuffle:
            random.shuffle(all_eval_datapoints)
            random.shuffle(all_train_datapoints)
        return all_train_datapoints, all_eval_datapoints

    def train(self, training_data_path="/workspace/dataset", output_dir: str = "artifacts/arc_solver_finetuned"):

        task_indices = (0, 300)
        window_size = 4
        shuffle_task = True
        epochs = 10
        load_dataset_from_disk = False

        sft_train_dataset_path = "./artifacts/arc_sft_dataset/train_dataset"
        eval_datapoints_file = "./artifacts/arc_sft_dataset/eval_datapoints.json"
        if load_dataset_from_disk:
            rich_print(f"[bold green]Loading pre-processed training dataset from {sft_train_dataset_path}...[/bold green]")
            train_dataset = load_from_disk(sft_train_dataset_path)
            with open(eval_datapoints_file, 'r', encoding='utf-8') as f:
                all_eval_datapoints = json.load(f)
            rich_print("[bold green]Datasets loaded.[/bold green]")

        else:
            print("--- Step 1: Preparing Data ---")
            all_train_datapoints, all_eval_datapoints = self._prepare_training_data(
                training_data_path=training_data_path,
                task_indices=task_indices,
                window_size=window_size,
                shuffle=shuffle_task
            )

            print("--- Step 2: Transforming Data for SFT ---")
            sft_train_list = []
            for datapoint in tqdm(all_train_datapoints, desc="Formatting Datapoints"):
                prompt = self.format_prompt(datapoint, is_training=True)
                sft_train_list.append({
                    "input_ids": prompt["input_ids"],
                    "labels": prompt["labels"],
                })

            print("--- Step 3: Creating Dataset ---")
            train_dataset = Dataset.from_list(sft_train_list)

            rich_print(f"[bold blue]Saving dataset to {sft_train_dataset_path}...[/bold blue]")
            os.makedirs(os.path.dirname(sft_train_dataset_path), exist_ok=True)
            train_dataset.save_to_disk(sft_train_dataset_path)
            with open(eval_datapoints_file, 'w', encoding='utf-8') as f:
                json.dump(all_eval_datapoints, f, indent=4)  # indent로 가독성 높임
            rich_print("[bold blue]Datasets saved.[/bold blue]")

        print("--- Step 4: Setting up PEFT/LoRA ---")
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
            #target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules=["q_proj", "v_proj"],
        )
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        print("--- Step 5: Setting up Training Arguments ---")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=32,
            learning_rate=1e-4,
            num_train_epochs=1,  # <-- 외부 루프에서 에폭을 제어하므로 1로 설정!
            lr_scheduler_type="cosine",
            save_strategy="no",  # <-- 수동으로 저장하므로 "no"로 설정!
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,  # 로깅 주기 단축
            optim="paged_adamw_8bit",
            #optim="adamw_torch",
            gradient_checkpointing=True,
            fp16=True,  # 또는 bf16 설정
            report_to="none",
            label_names=["labels"],
        )

        print("--- Step 6: Initializing SFTTrainer ---")
        trainer = SFTTrainer(
            model=self.model,
            #tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            #max_seq_length=2048,
        )

        print("--- Step 7: Starting Training & Evaluation Loop ---")
        for epoch in range(int(epochs)):
            rich_print(f"\n[bold green]----- Starting Epoch {epoch + 1}/{int(epochs)} ----- [/bold green]")

            random.shuffle(all_eval_datapoints)

            # 7-1. 1 에폭 훈련 실행
            print(f"Training for Epoch {epoch + 1}...")
            trainer.train()

            if epoch % 3 == 0:
                # 7-2. 에폭별 모델 저장
                epoch_output_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
                print(f"Saving model for epoch {epoch + 1} to {epoch_output_dir}")
                trainer.save_model(epoch_output_dir)
                self.tokenizer.save_pretrained(epoch_output_dir)

                # 7-3. 에폭별 평가 실행
                print(f"Evaluating after Epoch {epoch + 1}...")
                # evaluate 메서드는 이제 datapoint 리스트를 처리할 수 있어야 합니다.
                # (이전 답변에서 수정된 evaluate 메서드 사용)
                self.evaluate(all_eval_datapoints[:3])

        print("\n--- Step 8: Saving Final Model ---")
        final_adapter_path = os.path.join(output_dir, "checkpoint-final")
        print(f"Saving final adapter to {final_adapter_path}")
        self.model.save_pretrained(final_adapter_path)
        self.tokenizer.save_pretrained(final_adapter_path)

        print(f"\n[bold blue]Training complete. Final adapter saved to {final_adapter_path}[/bold blue]")

    def evaluate(self, all_eval_datapoints: List[Dict]):
        """
        Runs evaluation on the provided list of 'datapoint' dictionaries.

        Args:
            all_eval_datapoints (List[Dict]): A flat list where each element
                                               is a datapoint dictionary
                                               ({"train": ..., "test": ..., "task_file": ...}).
        """
        rich_print("\n[bold yellow]----- Starting Evaluation ----- [/bold yellow]")

        # 입력 데이터가 비어있는지 확인
        if not all_eval_datapoints:
            print("No evaluation data available.")
            return

        correct_predictions = 0
        total_predictions = len(all_eval_datapoints)

        self.model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():  # 그래디언트 계산 비활성화
            # 평탄화된 데이터포인트 리스트를 순회
            for i, datapoint in enumerate(tqdm(all_eval_datapoints, desc="Evaluating")):

                # 데이터포인트에서 정보 추출
                train_examples = datapoint["train"]
                test_input_grid = datapoint["test"][0]["input"]
                actual_output_grid = datapoint["test"][0]["output"]
                task_name = datapoint.get("task_file", f"Task_{i + 1}")

                # 모델 예측 호출
                predicted_output_grid, rule, flags, total_outputs = self.predict(
                    train_examples,
                    test_input_grid,
                    get_details=True,
                )

                # 결과 비교 (Numpy를 사용하여 그리드 일치 여부 확인)
                try:
                    # np.array로 변환하여 비교해야 정확합니다.
                    # 다만, 그리드 크기가 다르면 ValueError가 발생할 수 있습니다.
                    predicted_np = np.array(predicted_output_grid)
                    actual_np = np.array(actual_output_grid)
                    is_correct = (predicted_np.shape == actual_np.shape) and \
                                 (predicted_np == actual_np).all()
                except Exception:  # 비교 중 오류 발생 시 False 처리
                    is_correct = False

                # 결과 출력 (rich_print 또는 print 사용)
                print(f"\n--- Eval Sample {i + 1}/{total_predictions} (Task: {task_name}) ---")
                print(f"total outputs: {total_outputs}")
                print(f"Flags: {flags}")

                for idx, train_ex in enumerate(train_examples):
                    rich_print(f"[bold magenta]Train Example {idx} (Input):[/bold magenta]")
                    render_grid(train_ex['input'])
                    rich_print(f"[bold magenta]Train Example {idx} (Output):[/bold magenta]")
                    render_grid(train_ex['output'])
                    if idx == len(train_examples) - 1:  # 마지막 예제가 아니면 구분선 추가
                        rich_print("-" * 30)

                rich_print("[red]Test input :[/red]")
                render_grid(datapoint["test"][0]["input"])

                rich_print("[green]Predicted Output:[/green]")
                render_grid(predicted_output_grid)  # <-- render_grid 호출

                rich_print("[red]Ground Truth Output:[/red]")
                render_grid(actual_output_grid)  # <-- render_grid 호출

                rich_print(
                    f"[bold {'green' if is_correct else 'red'}]Correct: {is_correct}[/bold {'green' if is_correct else 'red'}]")

                if is_correct:
                    correct_predictions += 1

        # 최종 정확도 출력
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        rich_print(f"\n[bold yellow]----- Evaluation Complete ----- [/bold yellow]")
        rich_print(
            f"[bold magenta]Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})[/bold magenta]")

        self.model.train()  # 모델을 다시 훈련 모드로 설정
    def prepare_evaluation(self, adapter_path="artifacts/checkpoint-final"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter(adapter_path)
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()
