from transformers import GenerationConfig
import torch
from typing import List, Dict, Any
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
import re


class ARCSolver2:
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
        model_id = "google/gemma-7b"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.bfloat16,  # Set the computation data type
        )

        # Load pre-trained model
        use_cache = False if is_training else True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allow the model to use custom code from the repository
            quantization_config=bnb_config,  # Apply the 4-bit quantization configuration
            torch_dtype=torch.float16,  # Set the data type for the model
            use_cache=use_cache,  # at training, disable caching to save memory
            device_map='auto',  # Automatically map the model to available devices (e.g., GPUs)
            use_auth_token=token,
        )

        # Load tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        gemma_chat_template = (
            "{{ bos_token }}"  # 항상 시퀀스 시작 토큰 추가
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}"
            "{% elif message['role'] == 'model' %}"
            "{{ '<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"  # 추론 시 모델 턴 시작 추가
            "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )
        self.tokenizer.chat_template = gemma_chat_template

        # Define token IDs for ARC grid and pixels (0-10) and row seperator
        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.separator = self.tokenizer.encode('\n', add_special_tokens=False)[0]

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = self.model.device

    def format_grid_to_string(self, grid: List[List[int]]):
        if not grid:
            return ""
        return "\n".join("".join(map(str, row)) for row in grid)

    def parse_string_to_grid(self, grid_string: str) -> List[List[int]]:
        """문자열 표현을 2D 그리드로 파싱합니다. 견고성을 높입니다."""
        grid = []
        # 비어있는 문자열 처리
        if not grid_string:
            return []

        # 불필요한 공백 제거 및 특수 토큰 제거 (Gemma 출력 고려)
        grid_string = grid_string.strip().replace('<bos>', '').replace('<eos>', '')

        lines = grid_string.split('\n')
        for line in lines:
            line = line.strip()
            # 숫자(0-9) 이외의 문자를 제거합니다.
            cleaned_line = re.sub(r'[^0-9]', '', line)
            if cleaned_line:  # 빈 줄이 아니면
                try:
                    grid.append([int(pixel) for pixel in cleaned_line])
                except ValueError:
                    print(f"경고: 그리드 파싱 중 잘못된 문자가 포함된 줄을 건너뜁니다: '{line}'")
                    continue
        return grid

    def _build_prompt_content(self, examples: List[Dict], test_input: List[List[int]],
                              result_type: str = "grid") -> str:
        """Gemma 프롬프트의 'user' 부분을 문자열로 구성합니다."""
        system_prompt = message_templates["system_prompt"]
        user_msg1 = message_templates["user_message_template1"]
        user_msg2 = message_templates["user_message_template2"]

        prompt_parts = [system_prompt, "\n\n", user_msg1, "\n"]

        for ex in examples:
            input_str = self.format_grid_to_string(ex['input'])
            output_str = self.format_grid_to_string(ex['output'])
            prompt_parts.append(f"Input:\n{input_str}\nOutput:\n{output_str}\n---\n")

        prompt_parts.append(user_msg2 + "\n")
        test_input_str = self.format_grid_to_string(test_input)
        prompt_parts.append(f"Input:\n{test_input_str}\n")

        if result_type == "grid":
            prompt_parts.append(message_templates["user_message_template3"])
        elif result_type == "rule":
            prompt_parts.append(message_templates["user_message_template4"])
        else:
            raise ValueError("result_type must be 'grid' or 'rule'")

        return "".join(prompt_parts)

    def _apply_gemma_template(self, user_content: str, target_content: str = None) -> List[int]:
        """Gemma 채팅 템플릿을 적용하고 토큰화합니다."""
        chat = [{"role": "user", "content": user_content}]
        if target_content:
            chat.append({"role": "model", "content": target_content})

        # Gemma 템플릿 적용 (훈련 시에는 assistant 턴 포함, 추론 시에는 미포함)
        # 훈련 시에는 레이블 생성을 위해 assistant 턴을 포함해야 합니다.
        # 추론 시에는 assistant 턴 시작까지만 포함합니다.
        prompt_tokens = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=(target_content is None)  # target이 없으면(추론) True
        )
        return prompt_tokens

    def predict(self, examples: List[Dict], test_input: List[List[int]], result_type: str = "grid") -> Any:
        """
        주어진 예제와 테스트 입력으로 출력을 예측합니다. [cite: 7]
        Args:
            examples (List[dict]): 훈련 예제 목록 (3개)[cite: 6].
            test_input (List[List[int]]): 테스트 입력 그리드.
            result_type (str): 'grid' 또는 'rule'.
        Returns:
            List[List[int]] 또는 str: 예측된 그리드 또는 규칙 설명.
        """
        self.model.eval()  # 평가 모드로 설정

        user_content = self._build_prompt_content(examples, test_input, result_type)
        input_ids = self._apply_gemma_template(user_content)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # 생성 설정: max_new_tokens을 그리드 크기에 맞게 조정 (예: 512)
        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,  # 그리드 크기에 따라 조정
        )

        attention_mask = torch.ones_like(input_ids_tensor).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask,
                generation_config=config,
            ).squeeze().cpu()

        # 프롬프트 부분을 제외한 생성된 토큰만 추출
        output_only_ids = output_ids[len(input_ids):].tolist()

        # 생성된 토큰을 디코딩
        generated_text = self.tokenizer.decode(output_only_ids, skip_special_tokens=True)

        if result_type == "rule":
            return generated_text.strip()
        else:
            # 그리드 파싱 및 크기 조정
            predicted_grid = self.parse_string_to_grid(generated_text)

            # 크기 조정 로직 (입력-출력 비율 기반)
            try:
                train_input = np.array(examples[0]['input'])
                train_output = np.array(examples[0]['output'])
                test_input_np = np.array(test_input)

                if train_input.shape == train_output.shape:
                    target_shape = test_input_np.shape
                else:
                    h_ratio = train_output.shape[0] / train_input.shape[0] if train_input.shape[0] > 0 else 1
                    w_ratio = train_output.shape[1] / train_input.shape[1] if train_input.shape[1] > 0 else 1
                    target_shape = (
                        int(test_input_np.shape[0] * h_ratio),
                        int(test_input_np.shape[1] * w_ratio)
                    )
                target_shape = (max(1, target_shape[0]), max(1, target_shape[1]))

                # 파싱된 그리드를 목표 크기로 조정 (패딩/자르기)
                current_h = len(predicted_grid)
                current_w = max(len(r) for r in predicted_grid) if current_h > 0 else 0
                final_grid_np = np.zeros(target_shape, dtype=int)

                if current_h > 0 and current_w > 0:
                    padded_grid = np.zeros((current_h, current_w), dtype=int)
                    for i, r in enumerate(predicted_grid):
                        padded_grid[i, :len(r)] = r

                    h_to_copy = min(target_shape[0], current_h)
                    w_to_copy = min(target_shape[1], current_w)
                    final_grid_np[:h_to_copy, :w_to_copy] = padded_grid[:h_to_copy, :w_to_copy]

                return final_grid_np.tolist()

            except Exception as e:
                print(f"경고: 그리드 크기 조정 중 오류 발생: {e}. 파싱된 그리드를 반환합니다: {predicted_grid}")
                # 오류 발생 시, 파싱된 그리드를 그대로 반환하거나 기본값 반환
                return predicted_grid if predicted_grid else [[0]]

    def _prepare_training_dataset(self, training_data_path: str, num_tasks: int, examples_per_task: int) -> Dataset:
        """SFTTrainer에 적합한 훈련 데이터셋을 준비합니다."""
        processed_data = []
        task_files = [f for f in os.listdir(training_data_path) if f.endswith('.json')]
        task_files = task_files[:num_tasks]
        print(f"{len(task_files)}개의 태스크 파일을 로드합니다...")

        for task_file in tqdm(task_files, desc="태스크 처리 중"):
            path = os.path.join(training_data_path, task_file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    train_pairs = data.get('train', [])
                    test_pairs = data.get('test', [])
                    all_pairs = train_pairs + test_pairs  # 모든 예제를 활용

                    # 각 'test' 예제를 기준으로 훈련 데이터 구성
                    # 여기서는 간단하게, 각 test 예제에 대해 모든 train 예제를 사용합니다.
                    # 실제로는 더 정교한 샘플링 전략이 필요할 수 있습니다.
                    for i in range(len(test_pairs)):
                        test_pair = test_pairs[i]
                        # 사용할 훈련 예제 수 제한 (예: 3개)
                        current_train_examples = train_pairs[:3]
                        if not current_train_examples: continue

                        user_content = self._build_prompt_content(
                            current_train_examples,
                            test_pair['input'],
                            result_type="grid"
                        )
                        target_content = self.format_grid_to_string(test_pair['output'])

                        # Gemma 템플릿 적용 (user + model 턴)
                        # SFTTrainer는 이 전체 텍스트를 입력으로 받고,
                        # DataCollatorForLanguageModeling이 프롬프트 부분을 마스킹합니다.
                        chat = [
                            {"role": "user", "content": user_content},
                            {"role": "model", "content": target_content}
                        ]
                        full_text = self.tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,  # 텍스트로 받음
                            add_generation_prompt=False  # 이미 model 턴 포함
                        )
                        processed_data.append({"text": full_text})

                        if len(processed_data) >= examples_per_task * num_tasks: break
                if len(processed_data) >= examples_per_task * num_tasks: break

            except Exception as e:
                print(f"파일 처리 중 오류 발생 {task_file}: {e}")

        if not processed_data:
            raise ValueError("훈련 데이터가 생성되지 않았습니다.")

        return Dataset.from_list(processed_data)

    def train(self, training_data_path: str = "/workspace/dataset", output_dir: str = "artifacts/arc_gemma_finetuned"):
        """Gemma 모델을 LoRA를 사용하여 미세 조정합니다."""
        num_task_files_to_load = 30  # 훈련에 사용할 태스크 파일 수 [cite: 4] (300개 중 일부 사용)
        num_examples_per_task = 10  # 각 태스크에서 사용할 최대 예제 수

        # 1. 데이터셋 준비
        try:
            dataset = self._prepare_training_dataset(
                training_data_path, num_task_files_to_load, num_examples_per_task
            )
            print(f"총 {len(dataset)}개의 훈련 예제가 준비되었습니다.")
        except Exception as e:
            print(f"훈련 데이터셋 준비 중 심각한 오류 발생: {e}")
            return

        # 2. PEFT 설정 (LoRA)
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
        self.model.train()  # 훈련 모드로 설정

        # 3. 훈련 인수 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # VRAM 제약 고려 [cite: 16]
            gradient_accumulation_steps=4,
            learning_rate=1e-4,  # Gemma에 맞게 조정 가능
            num_train_epochs=3,  # 필요시 조정
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            fp16=True,  # bnb_config의 compute_dtype과 일치
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,  # 효율성 향상
            report_to="none",  # wandb 등 사용 안 함
        )

        # 4. SFTTrainer 설정
        max_seq_length = 1024  # Gemma 컨텍스트 및 그리드 크기 고려
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",  # _prepare_training_dataset에서 정의한 필드
            max_seq_length=max_seq_length,
            packing=True,  # 짧은 시퀀스를 묶어 효율성 향상
        )

        # 5. 훈련 시작
        print("Gemma 모델 미세 조정을 시작합니다...")
        trainer.train()

        # 6. 최종 어댑터 저장 [cite: 24]
        final_adapter_path = os.path.join(output_dir, "checkpoint-final")
        print(f"최종 어댑터를 {final_adapter_path}에 저장합니다.")
        trainer.save_model(final_adapter_path)  # SFTTrainer의 save_model 사용 권장
        self.tokenizer.save_pretrained(final_adapter_path)
        print(f"학습 완료. 어댑터가 {final_adapter_path}에 저장되었습니다.")

    def prepare_evaluation(self, adapter_path: str = "artifacts/arc_gemma_finetuned/checkpoint-final"):
        """평가를 위해 저장된 LoRA 어댑터를 로드합니다."""
        print(f"{adapter_path}에서 어댑터를 로드합니다...")
        try:
            # PEFT 모델은 load_adapter 대신 from_pretrained로 어댑터를 로드할 수 있습니다.
            # 또는 기존 모델에 load_adapter를 사용합니다.
            # 만약 get_peft_model을 다시 호출해야 한다면, base model을 먼저 로드해야 합니다.
            # 여기서는 이미 self.model이 PEFT 모델이라고 가정합니다.
            self.model.load_adapter(adapter_path)
            print("어댑터 로드 완료.")
        except Exception as e:
            print(f"어댑터 로드 중 오류 발생: {e}. 베이스 모델만 사용될 수 있습니다.")
        self.model.eval()
        print("모델이 평가 모드로 설정되었습니다.")


if __name__ == "__main__":
    solver = ARCSolver2()
