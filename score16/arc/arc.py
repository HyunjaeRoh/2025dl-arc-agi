### 최적화를 위해 transformers 전에 unsloth import
from unsloth import (
    FastLanguageModel,
    UnslothTrainer as Trainer,
    unsloth_train,
    UnslothTrainingArguments as TrainingArguments,
)
###

from transformers import GenerationConfig, pipeline
import torch
from typing import List
import numpy as np

# from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

### 추가된 import
import os
import logging
import tqdm.auto as tqdm
from .utils import InputMaskingDataCollator, save_model_and_tokenizer, keep_single_char_tokens
from .utils import load_peft_state
from .utils import load_data
from datasets import Dataset

####
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from unsloth.models.llama import FastLlamaModel


def _llama_reorder_cache(self, past_key_values, beam_idx):
    return tuple(
        tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
        for layer_past in past_key_values
    )


LlamaForCausalLM._reorder_cache = _llama_reorder_cache


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

        ### unsloth를 사용하여 모델과 토크나이저 로드
        logging.info(f"Loading base model {model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,  # Enable 4-bit quantization
            dtype=None,  # Use default dtype
        )

        ### 토크나이저 재설정 (단일 문자 토큰만 유지)
        keep_tok = list(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + self.tokenizer.tokenize('\n')
        keep_single_char_tokens(self.model, self.tokenizer, keep=keep_tok, remove_unk=True)

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
        2D 숫자 배열을 문자열로 변환 (행마다 문자열로 합치고 줄바꿈)
        """
        return "\n".join("".join(map(str, row)) for row in grid)

    def format_prompt(self, datapoint):
        """
        datapoint = {
          "train": [ {"input": grid, "output": grid}, ... ],  # 3개 예시
          "test":  [ {"input": grid} ]                       # 1개 테스트 인풋
        }
        simple format:
          I<train input>
          \n+/-=O\n<train output>\n  (3번 반복)
          I<test input>\n+/-=O
        """
        # get examples
        train_exs = datapoint["train"]
        test_in = datapoint["test"][0]["input"]

        # build simple prompt
        text = ""
        for ex in train_exs:
            text += "I" + self.format_grid(ex["input"])
            text += "\n+/-=O\n" + self.format_grid(ex["output"]) + "\n"
        # last test input
        text += "I" + self.format_grid(test_in) + "\n+/-=O"
        return text

    def train(self, train_dataset=None):
        """
        모델을 학습합니다. 지정된 경로에 체크포인트가 존재하면,
        해당 체크포인트를 불러와 학습을 이어갑니다. LoRA 어댑터 가중치는 항상 로드하려고 시도하며,
        Trainer 상태 파일(trainer_state.json)이 존재할 경우에만 Trainer의 전체 상태(옵티마이저, 에포크 등)를 이어받습니다.
        학습된 모델은 버전 관리되어 새로운 경로에 저장됩니다.
        """
        # 경로 설정
        base_save_name = "artifacts/finetuned_lora_model"
        lora_checkpoint_dir_to_load_from = f"{base_save_name}-lora"  # 로드 시도할 기본 어댑터/상태 경로
        trainer_output_dir = "output_trainer_checkpoints"

        # --- 데이터 로드, 전처리 (이전과 동일) ---
        data_path = "dataset"
        N_data = 3000  # 사용할 데이터 개수
        logging.info(f"Loading data from {data_path} with N_data={N_data}")

        df = load_data(data_path)
        from datasets import Dataset
        # ds = Dataset.from_pandas(df).shuffle(42).select(range(N_data))
        # 무작위 셔플
        ds = Dataset.from_pandas(df).shuffle().select(range(N_data))

        logging.info(f"Loaded {len(ds)} training examples")

        # 3-2) simple 포맷으로 prompt 문자열 생성
        def make_prompt(row):
            text = ""
            for ex in row["train"]:
                text += "I" + self.format_grid(ex["input"])
                text += "\n+/-=O\n" + self.format_grid(ex["output"]) + "\n"

            # 실제 test 그리드 꺼내기
            test_grid = row["test"][0]["input"]  # or row["test_input"][0]["input"]

            # ground-truth 출력도 넣어 supervised signal 확보
            test_out = row["test"][0]["output"]  # or row["test_output"][0]
            text += "I" + self.format_grid(test_grid)
            text += "\n+/-=O\n" + self.format_grid(test_out) + self.tokenizer.eos_token
            return {"text": text}

        ds = ds.map(make_prompt, remove_columns=ds.column_names)
        logging.info(f"Dataset prepared with {len(ds)} examples")

        # --- LoRA 설정 (이전과 동일) ---
        lora_layers = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ]
        logging.info("Applying LoRA config to the base model.")
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            target_modules=lora_layers,
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )
        # --- LoRA 설정 끝 ---

        # --- 기존 체크포인트 로드 로직 수정 ---
        # 이 변수는 Trainer의 상태 (optimizer, scheduler, epoch 등)를 이어받을 경로를 결정합니다.
        resume_path_for_trainer_state = None
        adapter_weights_loaded = False

        if os.path.exists(lora_checkpoint_dir_to_load_from) and os.path.isdir(lora_checkpoint_dir_to_load_from):
            adapter_model_files = [f for f in os.listdir(lora_checkpoint_dir_to_load_from) if
                                   f.startswith("adapter_model.")]
            if adapter_model_files:
                logging.info(
                    f"Found LoRA adapter files in {lora_checkpoint_dir_to_load_from}. Attempting to load weights.")
                try:
                    load_peft_state(self.model, lora_checkpoint_dir_to_load_from)  # utils.py의 함수 사용
                    logging.info(
                        f"Successfully loaded LoRA adapter weights into self.model from {lora_checkpoint_dir_to_load_from}.")
                    adapter_weights_loaded = True
                except Exception as e:
                    logging.error(f"Failed to load LoRA adapter weights from {lora_checkpoint_dir_to_load_from}: {e}",
                                  exc_info=True)
            else:
                logging.info(
                    f"No adapter_model files found in {lora_checkpoint_dir_to_load_from}, though directory exists.")

            # 어댑터 가중치 로드 여부와 관계없이, Trainer 상태 파일 존재 여부 확인
            trainer_state_file = os.path.join(lora_checkpoint_dir_to_load_from, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                logging.info(f"Full trainer state (trainer_state.json) found in {lora_checkpoint_dir_to_load_from}. "
                             f"Trainer will attempt to resume its full state from this directory.")
                resume_path_for_trainer_state = lora_checkpoint_dir_to_load_from
            elif adapter_weights_loaded:  # 어댑터 가중치는 로드했으나, 트레이너 상태 파일은 없는 경우
                logging.warning(
                    f"LoRA adapter weights were loaded from {lora_checkpoint_dir_to_load_from}, "
                    f"but 'trainer_state.json' was NOT found there. "
                    f"Trainer will start with fresh state (optimizer, scheduler, epoch count), "
                    f"but using the loaded adapter weights in the model."
                )
                # resume_path_for_trainer_state는 None으로 유지됩니다.
            else:  # 어댑터 가중치도 로드 못했고, 트레이너 상태 파일도 없는 경우
                logging.info(
                    f"'trainer_state.json' not found in {lora_checkpoint_dir_to_load_from}. No trainer state to resume.")
        else:
            logging.info(
                f"No existing checkpoint directory found at {lora_checkpoint_dir_to_load_from}. Starting fresh training.")
        # --- 기존 체크포인트 로드 로직 수정 끝 ---

        # --- TrainingArguments, DataCollator, Trainer 초기화 (resume_path_for_trainer_state 사용) ---
        training_args = TrainingArguments(
            output_dir=trainer_output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=4,
            learning_rate=5e-5,
            warmup_ratio=0.25,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=2,
            resume_from_checkpoint=resume_path_for_trainer_state,  # 수정된 경로 사용
            seed=42,
        )
        data_collator = InputMaskingDataCollator(
            tokenizer=self.tokenizer,
            response_template="+/-=O",
            instruction_template="I",
            mask_first_n_examples=3,
        )
        trainer = Trainer(
            model=self.model,  # self.model은 LoRA 가중치가 로드되었을 수 있음
            tokenizer=self.tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=12800,
            args=training_args,
            data_collator=data_collator,
        )
        # --- TrainingArguments, DataCollator, Trainer 초기화 끝 ---

        # 파인튜닝 시작
        logging.info(
            f"Starting fine-tuning. Trainer will attempt to resume state from: {resume_path_for_trainer_state or 'N/A (fresh start for trainer state)'}")
        # trainer.train()은 training_args에 설정된 resume_from_checkpoint를 자동으로 사용
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        stats = train_result.metrics
        logging.info(f"Training complete. Stats: {stats}")

        # --- 모델 저장 로직 (이전과 동일: 버전 관리하여 새 경로에 저장) ---
        version = 1
        new_save_lora_dir = f"{base_save_name}-lora-v{version}"
        while os.path.exists(new_save_lora_dir):
            version += 1
            new_save_lora_dir = f"{base_save_name}-lora-v{version}"
        logging.info(f"Attempting to save the newly trained model to: {new_save_lora_dir}")
        os.makedirs(new_save_lora_dir, exist_ok=True)
        save_model_and_tokenizer(new_save_lora_dir, self.model, self.tokenizer)
        logging.info(f"Newly trained LoRA adapter model saved to {new_save_lora_dir}")
        # --- 모델 저장 로직 끝 ---

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

        # 1) Prompt 생성
        prompt_str = self.format_prompt(datapoint)

        # 2) tokenize
        inputs = self.tokenizer(prompt_str,
                                return_tensors="pt",
                                add_special_tokens=False).to(self.device)

        # 3) generate
        num_return_sequences = 1
        gen_cfg = GenerationConfig(do_sample=False,
                                   num_beams=16,
                                   early_stopping=True,
                                   num_return_sequences=num_return_sequences,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   max_new_tokens=200)
        with torch.no_grad():
            outs = self.model.generate(**inputs, generation_config=gen_cfg).squeeze().cpu()
        grids = []
        rares = []
        if num_return_sequences == 1:
            outs = [outs]

        # 4) decode & parse
        prompt_len = inputs["input_ids"].numel()
        for out in outs:
            gen_ids = out.tolist()[prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rares.append(gen_text)
            lines = [l for l in gen_text.split("\n") if l.strip()]
            try:
                # 리스트 컴프리헨션 → 바로 numpy array 변환
                grid = np.array(
                    [[int(c) for c in l if c.isdigit()] for l in lines],
                    dtype=np.uint8
                )
                grids.append(grid)
            except Exception:
                # 파싱 실패 시에도 numpy array 생성
                h, w = len(questions_input), len(questions_input[0])
                grid = np.random.randint(0, 1, size=(h, w), dtype=np.uint8)
                grids.append(grid)
        grid = grids[0]
        return grid, #grids, rares

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """

        # 1) 경로 및 하이퍼파라미터 설정
        save_model_path = "artifacts/finetuned_lora_model"
        lora_dir = f"{save_model_path}-lora"

        # 2) 모델을 로드하고 평가 모드로 설정
        FastLanguageModel.for_inference(self.model)

        # # 3) PEET wrapper 적용 (train 때와 동일한 설정)
        lora_layers = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ]

        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            target_modules=lora_layers,
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )

        # 4) adapter 로드
        load_peft_state(self.model, lora_dir)


if __name__ == "__main__":
    solver = ARCSolver()
