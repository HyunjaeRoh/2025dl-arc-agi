from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
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

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

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
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt, add_special_tokens=False)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)


        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }

    def train(self, training_data_path="/workspace/dataset", output_dir: str = "artifacts/arc_solver_finetuned"):
        processed_data = []

        task_files = [os.path.join(training_data_path, f) for f in os.listdir(training_data_path)]

        task_files = task_files[:10] # 디버깅용: 파일 수 제한 (원본 코드에서 가져옴)

        print(f"학습을 위해 {len(task_files)}개의 ARC 작업을 로드하고 처리합니다...")
        for task_file_path in tqdm(task_files, desc="작업 처리 중"):
            try:
                with open(task_file_path, 'r') as f:
                    # loaded_task_data는 이제 [{"input": ..., "output": ...}, ...] 형태의 리스트
                    loaded_task_data_list = json.load(f)
            except Exception as e:
                print(f"{task_file_path} 건너뛰기: JSON 로드 오류 - {e}")
                continue

            if not isinstance(loaded_task_data_list, list) or not loaded_task_data_list:
                print(f"{task_file_path} 건너뛰기: 파일 내용이 유효한 리스트가 아닙니다.")
                continue

            # 데이터 처리 전략: 리스트의 마지막 요소를 test로, 나머지를 train 컨텍스트로 사용
            if len(loaded_task_data_list) >= 1:  # 최소 하나의 입출력 쌍이 있어야 함
                if len(loaded_task_data_list) >= 2:
                    train_examples_for_prompt = loaded_task_data_list[:-1]
                    current_test_pair = loaded_task_data_list[-1]
                else:  # 쌍이 하나만 있는 경우, 학습 예제 없이 이것을 test로 사용
                    train_examples_for_prompt = []
                    current_test_pair = loaded_task_data_list[0]

                # current_test_pair가 딕셔너리이고 'input', 'output' 키를 가지는지 확인
                if not isinstance(current_test_pair, dict) or \
                        'input' not in current_test_pair or \
                        'output' not in current_test_pair:
                    print(f"{task_file_path} 건너뛰기: 마지막 쌍의 형식이 올바르지 않습니다 (input/output 키 부재).")
                    continue

                # format_prompt가 기대하는 형태로 datapoint 구성
                datapoint_for_format = {
                    "train": train_examples_for_prompt,
                    "test": [current_test_pair]  # 'test'는 리스트여야 함
                }

                try:
                    prompt_info = self.format_prompt(datapoint_for_format)
                except ValueError as ve:
                    print(f"{task_file_path} 건너뛰기 (프롬프트 생성 오류): {ve}")
                    continue

                prompt_tokens = prompt_info['input_ids']
                target_grid = current_test_pair['output']  # 실제 학습 목표
                target_tokens = self.format_grid(target_grid)

                eot_token_id = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]

                full_tokens = prompt_tokens + target_tokens + [eot_token_id]
                labels = [-100] * len(prompt_tokens) + target_tokens + [eot_token_id]

                processed_data.append({
                    "input_ids": torch.tensor(full_tokens, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })
            else:
                print(f"{task_file_path} 건너뛰기: 파일에 유효한 입출력 쌍이 없습니다.")
                continue
        # --- 이하 PEFT 설정, TrainingArguments, SFTTrainer 초기화 및 학습 코드는 이전과 동일하게 유지 ---
        if not processed_data:
            print("학습을 위해 성공적으로 처리된 데이터가 없습니다. 종료합니다.")
            return

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

        max_len_data = 0
        if processed_data:  # processed_data가 비어있지 않은 경우에만 길이 계산
            for item in processed_data:
                item_len = len(item['input_ids'])
                if item_len > max_len_data:
                    max_len_data = item_len

        effective_max_seq_length = min(max_len_data, 4096) if max_len_data > 0 else 512  # 데이터가 없을 경우 기본값
        print(f"데이터에서 찾은 최대 시퀀스 길이: {max_len_data}. 학습에 사용할 길이: {effective_max_seq_length}")

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
            #tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            #max_seq_length=effective_max_seq_length,
        )

        print("미세 조정을 시작합니다...")
        trainer.train()

        final_adapter_path = os.path.join(output_dir, "checkpoint-final")
        print(f"최종 어댑터를 {final_adapter_path}에 저장합니다.")
        self.model.save_pretrained(final_adapter_path)
        self.tokenizer.save_pretrained(final_adapter_path)

        print(f"학습 완료. 어댑터가 {final_adapter_path}에 저장되었습니다.")
        print(f"이제 `prepare_evaluation(adapter_path='{final_adapter_path}')`를 사용하여 이 어댑터를 로드할 수 있습니다.")

    def predict(self, examples, questions_input):
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
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)

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

    def prepare_evaluation(self, adapter_path = "artifacts/checkpoint-final"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter(adapter_path)
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()




