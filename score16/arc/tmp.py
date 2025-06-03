### 최적화를 위해 transformers 전에 unsloth import
from unsloth import (
    FastLanguageModel,
    UnslothTrainer as Trainer,
    unsloth_train,
    UnslothTrainingArguments as TrainingArguments,
)
###

from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

# from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

### 추가된 import
import os
import logging
import tqdm.auto as tqdm
from utils import InputMaskingDataCollator, save_model_and_tokenizer, keep_single_char_tokens
from utils import load_peft_state
from utils  import load_data
from datasets import Dataset
import numpy as np
from collections import deque
####


class ARCSolver2:
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
        keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+self.tokenizer.tokenize('\n')
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
        test_in   = datapoint["test"][0]["input"]

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
            test_grid = row["test"][0]["input"]      # or row["test_input"][0]["input"]

            # ground-truth 출력도 넣어 supervised signal 확보
            test_out  = row["test"][0]["output"]     # or row["test_output"][0]
            text += "I" + self.format_grid(test_grid)
            text += "\n+/-=O\n" + self.format_grid(test_out) + self.tokenizer.eos_token
            return {"text": text}

        ds = ds.map(make_prompt, remove_columns=ds.column_names)
        logging.info(f"Dataset prepared with {len(ds)} examples")

        # --- LoRA 설정 (이전과 동일) ---
        lora_layers = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
            "embed_tokens","lm_head",
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
            adapter_model_files = [f for f in os.listdir(lora_checkpoint_dir_to_load_from) if f.startswith("adapter_model.")]
            if adapter_model_files:
                logging.info(f"Found LoRA adapter files in {lora_checkpoint_dir_to_load_from}. Attempting to load weights.")
                try:
                    load_peft_state(self.model, lora_checkpoint_dir_to_load_from) # utils.py의 함수 사용
                    logging.info(f"Successfully loaded LoRA adapter weights into self.model from {lora_checkpoint_dir_to_load_from}.")
                    adapter_weights_loaded = True
                except Exception as e:
                    logging.error(f"Failed to load LoRA adapter weights from {lora_checkpoint_dir_to_load_from}: {e}", exc_info=True)
            else:
                logging.info(f"No adapter_model files found in {lora_checkpoint_dir_to_load_from}, though directory exists.")

            # 어댑터 가중치 로드 여부와 관계없이, Trainer 상태 파일 존재 여부 확인
            trainer_state_file = os.path.join(lora_checkpoint_dir_to_load_from, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                logging.info(f"Full trainer state (trainer_state.json) found in {lora_checkpoint_dir_to_load_from}. "
                             f"Trainer will attempt to resume its full state from this directory.")
                resume_path_for_trainer_state = lora_checkpoint_dir_to_load_from
            elif adapter_weights_loaded: # 어댑터 가중치는 로드했으나, 트레이너 상태 파일은 없는 경우
                logging.warning(
                    f"LoRA adapter weights were loaded from {lora_checkpoint_dir_to_load_from}, "
                    f"but 'trainer_state.json' was NOT found there. "
                    f"Trainer will start with fresh state (optimizer, scheduler, epoch count), "
                    f"but using the loaded adapter weights in the model."
                )
                # resume_path_for_trainer_state는 None으로 유지됩니다.
            else: # 어댑터 가중치도 로드 못했고, 트레이너 상태 파일도 없는 경우
                logging.info(f"'trainer_state.json' not found in {lora_checkpoint_dir_to_load_from}. No trainer state to resume.")
        else:
            logging.info(f"No existing checkpoint directory found at {lora_checkpoint_dir_to_load_from}. Starting fresh training.")
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
            resume_from_checkpoint=resume_path_for_trainer_state, # 수정된 경로 사용
            seed=42,
        )
        data_collator = InputMaskingDataCollator(
            tokenizer=self.tokenizer,
            response_template="+/-=O",
            instruction_template="I",
            mask_first_n_examples=3,
        )
        trainer = Trainer(
            model=self.model, # self.model은 LoRA 가중치가 로드되었을 수 있음
            tokenizer=self.tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=12800,
            args=training_args,
            data_collator=data_collator,
        )
        # --- TrainingArguments, DataCollator, Trainer 초기화 끝 ---

        # 파인튜닝 시작
        logging.info(f"Starting fine-tuning. Trainer will attempt to resume state from: {resume_path_for_trainer_state or 'N/A (fresh start for trainer state)'}")
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

###############################BFS###############################################
    def solve_with_bfs(self, examples, test_input):
        """
        Train 예시(examples)를 만족시키는 연산(액션) 조합을 BFS 탐색으로 찾고,
        찾은 액션 시퀀스를 test_input에 적용한 결과를 반환한다.

        Args:
            examples (List[dict]): [{'input': 2D리스트, 'output': 2D리스트}, x3]
            test_input (List[List[int]]): 2D 리스트 (테스트 input)
        Returns:
            numpy.ndarray (2D array) 또는 2D 리스트: 추론된 output 격자
        """

        # 1-1) 허용할 액션(연산) 집합을 정의한다.
        actions = self.define_action_set()

        # 1-2) BFS 초기 설정
        visited = set()   # 이미 방문한 상태(state)를 해시 트래킹
        queue = deque()

        # test_input (2D 리스트) → 튜플 튜플 형태로 변환: hashable
        init_state = tuple(map(tuple, test_input))
        queue.append( (init_state, []) )   # (현재 상태 튜플, 지금까지 적용한 action 시퀀스 리스트)
        visited.add(init_state)

        # 1-3) BFS 루프
        while queue:
            current_state, action_seq = queue.popleft()

            # 2) 현재까지의 action_seq가 Train 예시 3개를 일관성 있게 변화시키는지 확인
            if self.is_consistent_with_examples(current_state, action_seq, examples):
                # Train 예시들을 만족하는 action_seq를 찾았다!
                # → test_input에 동일한 action_seq를 적용한 결과를 리턴
                result_grid = self.apply_actions_to(test_input, action_seq)
                return result_grid

            # 3) 이웃 상태로 확장: 가능한 모든 action을 시도한다
            for action in actions:
                next_state = self.apply_single_action(current_state, action)

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append( (next_state, action_seq + [action]) )

        # 4) BFS로도 찾지 못했으면 fallback
        return self.fallback_solution(test_input)


    #############################################
    # 2) 허용할 액션(action) 집합 정의
    #############################################
    def define_action_set(self):
        """
        BFS 탐색에서 사용할 '가능한 연산(action)' 목록을 생성해서 반환.
        예시로, 색 치환(replace_color), 격자 회전(rotate), 뒤집기(flip) 등을 포함한다.

        Train 예시들을 분석하여 액션 후보를 줄일 수도 있지만,
        여기서는 예시의 크기와 상관없이 기본적인 연산 스펙트럼을 정의.
        """
        actions = []
        # (a) 색 치환: 0~9 사이에서 모든 (from_color→to_color) 쌍
        for c_from in range(10):
            for c_to in range(10):
                if c_from == c_to:
                    continue
                actions.append(("replace_color", c_from, c_to))

        # (b) 회전: 90°, 180°, 270°
        for degree in (90, 180, 270):
            actions.append(("rotate", degree))

        # (c) 뒤집기: 수평/수직
        actions.append(("flip", "horizontal"))
        actions.append(("flip", "vertical"))

        # (d) 필요하다면 flood-fill, 색 영역별 채우기 등 추가 가능
        # ex) for each (i,j) 좌표와 색 d: actions.append(("flood_fill", i, j, d))

        return actions


    #############################################
    # 3) 단일 액션을 하나의 상태(state)에 적용
    #############################################
    def apply_single_action(self, state_tuple, action):
        """
        튜플-튜플 형태의 state (2D 격자) + 액션 튜플을 받아,
        액션을 적용한 새로운 상태(튜플-튜플)를 반환.

        Args:
            state_tuple: ((r0_0, r0_1, ...), (r1_0, r1_1, ...), ...)
            action: ("replace_color", from_color, to_color) 등
        Returns:
            next_state_tuple: 변환된 새로운 2D 격자 (tuple of tuples)
        """
        # 튜플 튜플 → numpy array
        grid = np.array(state_tuple, dtype=np.int64)
        op_name = action[0]

        if op_name == "replace_color":
            _, c_from, c_to = action
            new_grid = grid.copy()
            new_grid[new_grid == c_from] = c_to

        elif op_name == "rotate":
            _, degree = action
            k = degree // 90  # np.rot90은 k번 만큼 회전
            new_grid = np.rot90(grid, k=k)

        elif op_name == "flip":
            _, mode = action
            if mode == "horizontal":
                new_grid = np.fliplr(grid)
            else:  # "vertical"
                new_grid = np.flipud(grid)

        else:
            # 미구현 액션인 경우, 변형 없이 그대로 반환
            new_grid = grid.copy()

        # numpy array → 튜플 튜플 (hashable)로 변환
        return tuple(map(tuple, new_grid.tolist()))


    #############################################
    # 4) Train 예시들에 대해 현재 action 시퀀스 일관성 검사
    #############################################
    def is_consistent_with_examples(self, current_state, action_seq, examples):
        """
        지금까지의 action_seq를 Train 예시 각각의 input에 적용했을 때
        output이 예시의 실제 output과 모두 동일해야 True 반환.

        Args:
            current_state (tuple of tuples): 현재 state. 사실 이 값은 아직 test_input을 action_seq로
                                              변화시킨 중간 상태인데, 이 함수 내부에서는
                                              Train 예시를 다시 시작점부터 action_seq를 적용한다.
            action_seq (List[tuple]): 지금까지 적용한 액션들
            examples (List[dict]): [{'input': 2D리스트, 'output': 2D리스트}, ...]  타입

        Returns:
            bool: 모든 Train 예시가 일치하면 True, 아니면 False
        """
        # examples 하나하나 돌면서, action_seq를 input → output과 비교
        for ex in examples:
            train_in = ex["input"]   # 2D 리스트
            train_out = ex["output"] # 2D 리스트

            # train_in → 튜플 튜플로 변환
            state = tuple(map(tuple, train_in))
            # action_seq를 순차 적용
            for action in action_seq:
                state = self.apply_single_action(state, action)

            # 최종 state(튜플 튜플) → numpy로 변환해 비교
            final_np = np.array(state, dtype=np.int64)
            target_np = np.array(train_out, dtype=np.int64)
            if not np.array_equal(final_np, target_np):
                return False

        return True


    #############################################
    # 5) action_seq를 실제 grid(2D 리스트)에 적용하여 numpy 반환
    #############################################
    def apply_actions_to(self, grid_list, action_seq):
        """
        grid_list: 2D 리스트(int)
        action_seq: [("replace_color", ...), ("rotate", ...), ...]

        Returns:
            numpy.ndarray: action_seq를 순차 적용한 결과 2D 배열
        """
        state = np.array(grid_list, dtype=np.int64)
        # (이전 state는 tuple 튜플 형태가 아닌, 처음부터 numpy에서 적용해도 무방)
        for action in action_seq:
            # numpy → 튜플 튜플 → apply → numpy로 다시 변환
            state = np.array(
                self.apply_single_action(tuple(map(tuple, state.tolist())), action),
                dtype=np.int64
            )
        return state


    #############################################
    # 6) BFS 실패 시 fallback: LLM 예측 혹은 전부 0으로 채워
    #############################################
    def fallback_solution(self, test_input):
        """
        BFS 탐색으로도 일치하는 규칙을 찾지 못했을 때 호출.
        여기서는 단순히 0으로 채운 격자를 반환하거나,
        기존 LLM 예측 코드를 그대로 호출하도록 구현할 수 있다.
        """
        # 예시 1) 전부 0으로 채우기
        h, w = len(test_input), len(test_input[0])
        return np.zeros((h, w), dtype=np.uint8)

        # 예시 2) 기존 LLM generate 로직을 그대로 돌리려면:
        # return self.llm_predict(examples, test_input)
        # (llm_predict는 LLM 예측 코드를 별도 메서드로 분리했을 때 사용)

#################################BFS###################################################


#################################Predict수정############################################
    def predict(self, examples, questions_input):
        """
        examples: [{'input': 2D리스트, 'output': 2D리스트}, x3]
        questions_input: 2D 리스트 (테스트 input)

        BFS 기반 풀이가 가능하면 solve_with_bfs 호출,
        그렇지 않으면 기존 LLM generate 로직을 사용한다.
        """
        # 0) 간단한 조건: Train 예시 개수≦3이면서 격자 크기가 작을 땐 BFS를 시도
        max_h = max(len(ex["input"]) for ex in examples)
        max_w = max(len(ex["input"][0]) for ex in examples)
        # (예: 가로세로 모두 10 이하일 때만 BFS 시도)
        if len(examples) <= 3 and max_h <= 10 and max_w <= 10:
            bfs_result = self.solve_with_bfs(examples, questions_input)
            # 만약 BFS가 유효한 결과(0이 아닌 값 등)를 반환했다면
            if not (isinstance(bfs_result, np.ndarray) and np.all(bfs_result == 0)):
                return bfs_result

        # 1) BFS 실패하거나, BFS 조건이 안 되면 기존 LLM 로직 실행
        #    (원래 predict() 안에 있던 LLM generate 코드 복사)
        datapoint = {"train": examples, "test": [{"input": questions_input}]}
        prompt_str = self.format_prompt(datapoint)

        inputs = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150
        )
        out = self.model.generate(**inputs, generation_config=gen_cfg).squeeze().cpu()

        prompt_len = inputs["input_ids"].numel()
        gen_ids    = out.tolist()[prompt_len:]
        gen_text   = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        lines      = [l for l in gen_text.split("\n") if l.strip()]
        try:
            grid = np.array(
                [[int(c) for c in l if c.isdigit()] for l in lines],
                dtype=np.uint8
            )
        except Exception:
            h, w = len(questions_input), len(questions_input[0])
            grid = np.random.randint(0, 10, size=(h, w), dtype=np.uint8)

        return grid

##################################predict수정##############################################

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """

        # 1) 경로 및 하이퍼파라미터 설정
        save_model_path = "artifacts/finetuned_lora_model"
        lora_dir = f"{save_model_path}-lora"

        # 2) 모델을 로드하고 평가 모드로 설정
        FastLanguageModel.for_inference(self.model)

        # 3) PEET wrapper 적용 (train 때와 동일한 설정)
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