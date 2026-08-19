"""
Dataset processors for handling different dataset formats during activation extraction.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from datasets import load_dataset
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class ActivationCapture:
    """Robust activation capture during generation."""

    def __init__(self, layer_index: int, prompt_length: int):
        self.layer_index = layer_index
        self.prompt_length = prompt_length
        self.activations: List[torch.Tensor] = []
        self.last_seen_length = 0  # Track last processed sequence length
        self.hook = None
        # KV cache detection
        self.kv_cache_mode = False  # Whether we're in KV cache mode
        self.prompt_processed = False  # Whether prompt has been processed

    def capture_hook(self, module, input, output):
        """Hook that properly tracks token generation, handling KV cache."""
        # Get current sequence length
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        current_length = hidden_states.shape[1]

        # First call with prompt length marks prompt as processed
        if not self.prompt_processed and current_length >= self.prompt_length:
            self.prompt_processed = True

        # Detect KV cache mode: sequence drops from prompt_length to 1
        if (
            not self.kv_cache_mode
            and self.last_seen_length >= self.prompt_length
            and current_length == 1
        ):
            self.kv_cache_mode = True

        # Handle based on mode
        if self.kv_cache_mode:
            # In KV cache mode, each call with length 1 is a new generated token
            if current_length == 1:
                activation = hidden_states[:, 0, :].detach().cpu()
                self.activations.append(activation)
        else:
            # Non-KV cache mode or before KV cache detection
            if current_length > self.last_seen_length:
                # Determine which tokens are new
                if self.last_seen_length < self.prompt_length:
                    # Still processing prompt, skip to generation phase
                    start_idx = self.prompt_length
                    if current_length >= self.prompt_length:
                        self.prompt_processed = True
                else:
                    # Already in generation, capture from last seen position
                    start_idx = self.last_seen_length

                # Capture all new token activations beyond the prompt
                if current_length > start_idx:
                    new_activations = (
                        hidden_states[:, start_idx:current_length, :]
                        .detach()
                        .cpu()
                    )
                    # Store each token activation separately
                    for i in range(new_activations.shape[1]):
                        self.activations.append(new_activations[:, i, :])

        # Update tracking
        self.last_seen_length = current_length

    def register(self, model, layer_name: str):
        """Register hook on specified layer."""
        layer = model
        for part in layer_name.split("."):
            try:
                layer = getattr(layer, part)
            except AttributeError:
                layer = layer[int(part)]

        self.hook = layer.register_forward_hook(self.capture_hook)

    def remove(self):
        """Safely remove hook."""
        if self.hook:
            self.hook.remove()
            self.hook = None

    def get_activations(self) -> torch.Tensor:
        """Return captured activations."""
        if not self.activations:
            return torch.empty(0)
        return torch.cat(self.activations, dim=0)


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load dataset and return texts, labels, and optional metadata."""
        pass

    def _load_dataset(self) -> Any:
        """Common dataset loading logic."""
        dataset_kwargs = {"path": self.cfg.dataset.name}

        if hasattr(self.cfg.dataset, "subset") and self.cfg.dataset.subset:
            dataset_kwargs["name"] = self.cfg.dataset.subset

        if hasattr(self.cfg.dataset, "data_dir") and self.cfg.dataset.data_dir:
            dataset_kwargs["data_dir"] = self.cfg.dataset.data_dir

        dataset = load_dataset(**dataset_kwargs)

        # Get the appropriate split
        if self.cfg.dataset.split in dataset:
            data = dataset[self.cfg.dataset.split]
        else:
            # Try common split names
            for split_name in ["train", "test", "validation"]:
                if split_name in dataset:
                    data = dataset[split_name]
                    log.warning(
                        f"Split '{self.cfg.dataset.split}' not found, using '{split_name}'"
                    )
                    break
            else:
                raise ValueError(
                    f"No valid split found in dataset. Available: {list(dataset.keys())}"
                )

        return data

    def _limit_samples(
        self, texts: List[str], labels: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Limit samples if max_samples is specified."""
        if self.cfg.dataset.max_samples > 0:
            texts = texts[: self.cfg.dataset.max_samples]
            labels = labels[: self.cfg.dataset.max_samples]
        return texts, labels


class ToxigenProcessor(DatasetProcessor):
    """Processor for Toxigen dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Toxigen dataset."""
        data = self._load_dataset()

        texts = []
        labels = []

        for item in data:
            # Combine prompt and generation as per original implementation
            text = item.get("prompt", "") + item["generation"]
            texts.append(text)

            # Convert label: 1 - prompt_label (toxic=1 becomes safety=0)
            label = 1 - item["prompt_label"]
            labels.append(label)

        log.info(f"Processed toxigen dataset with {len(texts)} samples")
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class GSM8KProcessor(DatasetProcessor):
    """Processor for GSM8K dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process GSM8K dataset."""
        data = self._load_dataset()

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(data)
        else:
            # Single strategy processing (backward compatible)
            texts, labels = self._process_single_strategy(data)
            return texts, labels, {}

    def _process_single_strategy(self, data) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        # Get text combination strategy
        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "question_only"
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nAnswer: ")

        for item in data:
            text = self._get_text_for_strategy(item, strategy, separator)
            texts.append(text)

            # GSM8K doesn't have safety labels, use default
            label = self.cfg.dataset.get("default_label", 1)
            labels.append(label)

        log.info(
            f"Processed GSM8K dataset with {len(texts)} samples using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies",
            ["question_only", "answer_only", "question_answer"],
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nAnswer: ")

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(item, strategy, separator)
                texts.append(text)

                # GSM8K doesn't have safety labels, use default
                label = self.cfg.dataset.get("default_label", 1)
                labels.append(label)

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed GSM8K dataset with {len(strategies)} strategies: {len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # Recalculate indices based on limited data
            new_indices = {}
            for strategy in strategies:
                old_start, old_end = metadata["strategy_indices"][strategy]
                new_start = min(old_start, len(limited_texts))
                new_end = min(old_end, len(limited_texts))
                if new_start < new_end:
                    new_indices[strategy] = (new_start, new_end)
            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str
    ) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "question_only":
            return item["question"]
        elif strategy == "answer_only":
            return item["answer"]
        elif strategy == "question_answer":
            return item["question"] + separator + item["answer"]
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class GenericProcessor(DatasetProcessor):
    """Generic processor for datasets with standard text column."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process generic dataset."""
        data = self._load_dataset()

        # Check if text column exists
        if self.cfg.dataset.text_column not in data.column_names:
            log.error(
                f"Column '{self.cfg.dataset.text_column}' not found. "
                f"Available columns: {data.column_names}"
            )
            raise ValueError(
                f"Text column '{self.cfg.dataset.text_column}' not found in dataset"
            )

        texts = list(data[self.cfg.dataset.text_column])

        # Handle labels
        if (
            self.cfg.dataset.get("has_labels", True)
            and "label" in data.column_names
        ):
            labels = list(data["label"])
        else:
            # Use default label for all samples
            default_label = self.cfg.dataset.get("default_label", 0)
            labels = [default_label] * len(texts)
            log.warning(
                f"Using default label {default_label} for all samples "
                f"(has_labels={self.cfg.dataset.get('has_labels', True)})"
            )

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class AlpacaProcessor(DatasetProcessor):
    """Processor for Alpaca instruction-following dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Alpaca dataset."""
        data = self._load_dataset()

        texts = []
        labels = []

        # Get text extraction strategy
        strategy = self.cfg.dataset.get(
            "text_extraction_strategy", "text_full"
        )

        for item in data:
            if strategy == "instruction_only":
                text = item["instruction"]
            elif strategy == "output_only":
                text = item["output"]
            elif strategy == "text_full":
                text = item["text"]
            elif strategy == "instruction_output":
                # Custom formatting without the template
                separator = self.cfg.dataset.get("text_separator", "\n\n")
                text = item["instruction"] + separator + item["output"]
            else:
                raise ValueError(
                    f"Unknown text extraction strategy: {strategy}"
                )

            texts.append(text)

            # Alpaca doesn't have safety labels, use default
            label = self.cfg.dataset.get(
                "default_label", 1
            )  # Default to "safe"
            labels.append(label)

        log.info(
            f"Processed Alpaca dataset with {len(texts)} samples using strategy: {strategy}"
        )
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class HumanEvalProcessor(DatasetProcessor):
    """Processor for OpenAI HumanEval dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process HumanEval dataset."""
        data = self._load_dataset()

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(data)
        else:
            # Single strategy processing (backward compatible)
            texts, labels = self._process_single_strategy(data)
            return texts, labels, {}

    def _process_single_strategy(self, data) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        # Get text combination strategy
        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "prompt_only"
        )
        separator = self.cfg.dataset.get("text_separator", "\n\n# Solution:\n")

        for item in data:
            text = self._get_text_for_strategy(item, strategy, separator)
            texts.append(text)

            # HumanEval doesn't have safety labels, use default
            label = self.cfg.dataset.get("default_label", 1)
            labels.append(label)

        log.info(
            f"Processed HumanEval dataset with {len(texts)} samples "
            f"using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies",
            ["prompt_only", "solution_only", "prompt_solution"],
        )
        separator = self.cfg.dataset.get("text_separator", "\n\n# Solution:\n")

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(item, strategy, separator)
                texts.append(text)

                # HumanEval doesn't have safety labels, use default
                label = self.cfg.dataset.get("default_label", 1)
                labels.append(label)

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed HumanEval dataset with {len(strategies)} strategies: "
            f"{len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # Recalculate indices based on limited data
            new_indices = {}
            for strategy in strategies:
                old_start, old_end = metadata["strategy_indices"][strategy]
                new_start = min(old_start, len(limited_texts))
                new_end = min(old_end, len(limited_texts))
                if new_start < new_end:
                    new_indices[strategy] = (new_start, new_end)
            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str
    ) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "prompt_only":
            return item["prompt"]
        elif strategy == "solution_only":
            return item["canonical_solution"]
        elif strategy == "prompt_solution":
            return item["prompt"] + separator + item["canonical_solution"]
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class ARCProcessor(DatasetProcessor):
    """Processor for AI2 ARC dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process ARC dataset."""
        # Load both ARC-Easy and ARC-Challenge by default
        subsets = self.cfg.dataset.get(
            "subsets", ["ARC-Easy", "ARC-Challenge"]
        )
        if isinstance(subsets, str):
            subsets = [subsets]

        all_data = []
        for subset in subsets:
            data = load_dataset(
                "allenai/ai2_arc", subset, split=self.cfg.dataset.split
            )
            all_data.extend(data)

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(all_data)
        else:
            texts, labels = self._process_single_strategy(all_data)
            return texts, labels, {}

    def _process_single_strategy(
        self, data: List[Dict]
    ) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "question_choices"
        )

        for item in data:
            text = self._get_text_for_strategy(item, strategy)
            texts.append(text)
            labels.append(self.cfg.dataset.get("default_label", 1))

        log.info(
            f"Processed ARC dataset with {len(texts)} samples using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data: List[Dict]
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies", ["question_only", "choices_only", "question_choices"]
        )

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(item, strategy)
                texts.append(text)
                labels.append(self.cfg.dataset.get("default_label", 1))

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed ARC dataset with {len(strategies)} strategies: {len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # When limiting samples, we need to recalculate indices completely
            # because strategies might be cut off
            samples_per_strategy = len(data)
            limited_per_strategy = len(limited_texts) // len(strategies)
            remainder = len(limited_texts) % len(strategies)

            new_indices = {}
            current_idx = 0

            for i, strategy in enumerate(strategies):
                strategy_samples = limited_per_strategy + (
                    1 if i < remainder else 0
                )
                if strategy_samples > 0:
                    new_indices[strategy] = (
                        current_idx,
                        current_idx + strategy_samples,
                    )
                    current_idx += strategy_samples

            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _format_choices(self, choices: Dict) -> str:
        """Format choices in MMLU style."""
        formatted = []
        for label, text in zip(choices["label"], choices["text"]):
            formatted.append(f"{label}. {text}")
        return "\n".join(formatted)

    def _get_text_for_strategy(self, item: Dict, strategy: str) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "question_only":
            return item["question"]
        elif strategy == "choices_only":
            return self._format_choices(item["choices"])
        elif strategy == "question_choices":
            return (
                f"{item['question']}\n{self._format_choices(item['choices'])}"
            )
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class MMLUProcessor(DatasetProcessor):
    """Processor for MMLU (Massive Multitask Language Understanding) dataset."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # MMLU-specific configuration
        self.k_shot = cfg.dataset.get(
            "k_shot", 5
        )  # Number of few-shot examples
        self.process_all_subjects = cfg.dataset.get(
            "process_all_subjects", True
        )
        self.specific_subjects = cfg.dataset.get("specific_subjects", [])

    def _format_subject(self, subject: str) -> str:
        """Format subject name for display."""
        s = ""
        for entry in subject.split("_"):
            s += " " + entry
        return s

    def _format_example(
        self,
        question: str,
        choices: List[str],
        answer_idx: Optional[int] = None,
        include_answer: bool = True,
    ) -> str:
        """Format a single MMLU example."""
        prompt = question
        choice_letters = ["A", "B", "C", "D"]

        for j, choice in enumerate(choices):
            prompt += f"\n{choice_letters[j]}. {choice}"

        prompt += "\nAnswer:"

        if include_answer and answer_idx is not None:
            prompt += f" {choice_letters[answer_idx]}\n\n"

        return prompt

    def _gen_prompt(
        self, subject: str, dev_examples: List[Dict], k: int = -1
    ) -> str:
        """Generate few-shot prompt with k examples."""
        prompt = f"Multiple choice questions (with answers) about{self._format_subject(subject)}:\n\n"

        if k == -1:
            k = len(dev_examples)

        for i in range(min(k, len(dev_examples))):
            example = dev_examples[i]
            prompt += self._format_example(
                example["question"],
                example["choices"],
                example["answer"],
                True,
            )

        return prompt

    def _load_mmlu_data(self) -> Tuple[Any, Any]:
        """Load MMLU dataset with dev and test splits."""
        from datasets import load_dataset

        # Determine which config to use
        config = self.cfg.dataset.get("subset", "all")

        # Load dev set for few-shot examples
        dev_dataset = load_dataset("cais/mmlu", config, split="dev")

        # Load test set for evaluation
        test_dataset = load_dataset("cais/mmlu", config, split="test")

        return dev_dataset, test_dataset

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process MMLU dataset."""
        dev_data, test_data = self._load_mmlu_data()

        texts = []
        labels = []

        # Group examples by subject if processing all
        if self.process_all_subjects and self.cfg.dataset.subset == "all":
            # Group dev examples by subject
            dev_by_subject: Dict[str, List[Dict]] = {}
            for example in dev_data:
                subject = example["subject"]
                if subject not in dev_by_subject:
                    dev_by_subject[subject] = []
                dev_by_subject[subject].append(example)

            # Process test examples
            for example in test_data:
                subject = example["subject"]

                # Skip if specific subjects are requested and this isn't one
                if (
                    self.specific_subjects
                    and subject not in self.specific_subjects
                ):
                    continue

                # Get dev examples for this subject
                subject_dev_examples = dev_by_subject.get(subject, [])

                # Generate few-shot prompt
                prompt = self._gen_prompt(
                    subject, subject_dev_examples, self.k_shot
                )

                # Add the test question without answer
                prompt += self._format_example(
                    example["question"],
                    example["choices"],
                    include_answer=False,
                )

                texts.append(prompt)
                # All MMLU questions are considered safe
                labels.append(1)

        else:
            # Process single subject
            # Convert dev data to list for easier handling
            dev_examples = [dev_data[i] for i in range(len(dev_data))]

            for example in test_data:
                # Generate few-shot prompt
                prompt = self._gen_prompt(
                    example["subject"], dev_examples, self.k_shot
                )

                # Add the test question without answer
                prompt += self._format_example(
                    example["question"],
                    example["choices"],
                    include_answer=False,
                )

                texts.append(prompt)
                # All MMLU questions are considered safe
                labels.append(1)

        log.info(f"Processed MMLU dataset with {len(texts)} samples")
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class MBPPProcessor(DatasetProcessor):
    """Processor for MBPP (Mostly Basic Python Problems) dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process MBPP dataset."""
        data = self._load_dataset()

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(data)
        else:
            # Single strategy processing (backward compatible)
            texts, labels = self._process_single_strategy(data)
            return texts, labels, {}

    def _process_single_strategy(self, data) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        # Get text combination strategy
        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "text_only"
        )
        separator = self.cfg.dataset.get("text_separator", "\n\n# Solution:\n")
        include_tests = self.cfg.dataset.get("include_tests", False)

        for item in data:
            text = self._get_text_for_strategy(
                item, strategy, separator, include_tests
            )
            texts.append(text)

            # MBPP doesn't have safety labels, use default (safe)
            label = self.cfg.dataset.get("default_label", 1)
            labels.append(label)

        log.info(
            f"Processed MBPP dataset with {len(texts)} samples using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies",
            ["text_only", "code_only", "text_code"],
        )
        separator = self.cfg.dataset.get("text_separator", "\n\n# Solution:\n")
        include_tests = self.cfg.dataset.get("include_tests", False)

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(
                    item, strategy, separator, include_tests
                )
                texts.append(text)

                # MBPP doesn't have safety labels, use default (safe)
                label = self.cfg.dataset.get("default_label", 1)
                labels.append(label)

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed MBPP dataset with {len(strategies)} strategies: {len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # Recalculate indices based on limited data
            new_indices = {}
            for strategy in strategies:
                old_start, old_end = metadata["strategy_indices"][strategy]
                new_start = min(old_start, len(limited_texts))
                new_end = min(old_end, len(limited_texts))
                if new_start < new_end:
                    new_indices[strategy] = (new_start, new_end)
            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str, include_tests: bool
    ) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "text_only":
            return item["text"]
        elif strategy == "code_only":
            return item["code"]
        elif strategy == "text_code":
            return item["text"] + separator + item["code"]
        elif strategy == "text_code_tests":
            # Include test cases if requested
            text = item["text"] + separator + item["code"]
            if include_tests and item.get("test_list"):
                test_section = "\n\n# Tests:\n" + "\n".join(item["test_list"])
                text += test_section
            return text
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class MBPPPlusProcessor(MBPPProcessor):
    """Processor for evalplus/mbppplus dataset."""

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str, include_tests: bool
    ) -> str:
        """Get text based on strategy, mapping 'prompt' to 'text'."""
        # Map MBPPPlus fields to expected MBPP fields
        mapped_item = {
            "text": item["prompt"],  # MBPPPlus uses 'prompt' instead of 'text'
            "code": item["code"],
            "test_list": item.get("test_list", []),
        }

        return super()._get_text_for_strategy(
            mapped_item, strategy, separator, include_tests
        )


class MathSFT40KProcessor(DatasetProcessor):
    """Processor for Math SFT 40K dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Math SFT 40K dataset."""
        data = self._load_dataset()

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(data)
        else:
            # Single strategy processing (backward compatible)
            texts, labels = self._process_single_strategy(data)
            return texts, labels, {}

    def _process_single_strategy(self, data) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        # Get text combination strategy
        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "question_only"
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nResponse: ")

        for item in data:
            text = self._get_text_for_strategy(item, strategy, separator)
            texts.append(text)

            # Math SFT 40K doesn't have safety labels, use default (safe)
            label = self.cfg.dataset.get("default_label", 1)
            labels.append(label)

        log.info(
            f"Processed Math SFT 40K dataset with {len(texts)} samples using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies",
            ["question_only", "no_think_response_only", "question_no_think"],
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nResponse: ")

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(item, strategy, separator)
                texts.append(text)

                # Math SFT 40K doesn't have safety labels, use default (safe)
                label = self.cfg.dataset.get("default_label", 1)
                labels.append(label)

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed Math SFT 40K dataset with {len(strategies)} strategies: "
            f"{len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # Recalculate indices based on limited data
            new_indices = {}
            for strategy in strategies:
                old_start, old_end = metadata["strategy_indices"][strategy]
                new_start = min(old_start, len(limited_texts))
                new_end = min(old_end, len(limited_texts))
                if new_start < new_end:
                    new_indices[strategy] = (new_start, new_end)
            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str
    ) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "question_only":
            return item["question"]
        elif strategy == "no_think_response_only":
            return item["no_think_response"]
        elif strategy == "think_response_only":
            return item["think_response"]
        elif strategy == "question_no_think":
            return item["question"] + separator + item["no_think_response"]
        elif strategy == "question_think":
            return item["question"] + separator + item["think_response"]
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class MathQAProcessor(DatasetProcessor):
    """Processor for MathQA dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process MathQA dataset."""
        data = self._load_dataset()

        # Check if we should process all strategies
        if self.cfg.dataset.get("process_all_strategies", False):
            return self._process_all_strategies(data)
        else:
            # Single strategy processing (backward compatible)
            texts, labels = self._process_single_strategy(data)
            return texts, labels, {}

    def _process_single_strategy(self, data) -> Tuple[List[str], List[int]]:
        """Process dataset with a single text combination strategy."""
        texts = []
        labels = []

        # Get text combination strategy
        strategy = self.cfg.dataset.get(
            "text_combination_strategy", "problem_only"
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nRationale: ")

        for item in data:
            text = self._get_text_for_strategy(item, strategy, separator)
            texts.append(text)

            # MathQA doesn't have safety labels, use default
            label = self.cfg.dataset.get("default_label", 1)
            labels.append(label)

        log.info(
            f"Processed MathQA dataset with {len(texts)} samples using strategy: {strategy}"
        )
        return self._limit_samples(texts, labels)

    def _process_all_strategies(
        self, data
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Process dataset with all text combination strategies."""
        strategies = self.cfg.dataset.get(
            "strategies",
            [
                "problem_only",
                "rationale_only",
                "problem_rationale",
                "problem_options",
                "problem_options_rationale",
            ],
        )
        separator = self.cfg.dataset.get("text_separator", "\n\nRationale: ")

        all_texts: List[str] = []
        all_labels: List[int] = []
        metadata = {"strategies": strategies, "strategy_indices": {}}

        for strategy in strategies:
            texts = []
            labels = []

            for item in data:
                text = self._get_text_for_strategy(item, strategy, separator)
                texts.append(text)

                # MathQA doesn't have safety labels, use default
                label = self.cfg.dataset.get("default_label", 1)
                labels.append(label)

            # Record indices for this strategy
            start_idx = len(all_texts)
            end_idx = start_idx + len(texts)
            metadata["strategy_indices"][strategy] = (start_idx, end_idx)

            all_texts.extend(texts)
            all_labels.extend(labels)

            log.info(
                f"Processed {len(texts)} samples with strategy: {strategy}"
            )

        log.info(
            f"Processed MathQA dataset with {len(strategies)} strategies: "
            f"{len(all_texts)} total samples"
        )

        # Apply limit to the combined data
        limited_texts, limited_labels = self._limit_samples(
            all_texts, all_labels
        )

        # Update metadata indices if data was limited
        if len(limited_texts) < len(all_texts):
            # Recalculate indices based on limited data
            new_indices = {}
            for strategy in strategies:
                old_start, old_end = metadata["strategy_indices"][strategy]
                new_start = min(old_start, len(limited_texts))
                new_end = min(old_end, len(limited_texts))
                if new_start < new_end:
                    new_indices[strategy] = (new_start, new_end)
            metadata["strategy_indices"] = new_indices

        return limited_texts, limited_labels, metadata

    def _parse_options(self, options_str: str) -> str:
        """Parse options from 'a ) 24 , b ) 120 , ...' to formatted text."""
        options = options_str.split(" , ")
        formatted = ["Options:"]
        for option in options:
            parts = option.split(" ) ")
            if len(parts) == 2:
                letter, value = parts
                formatted.append(f"{letter.upper()}) {value}")
        return "\n".join(formatted)

    def _get_text_for_strategy(
        self, item: Dict, strategy: str, separator: str
    ) -> str:
        """Get text for a specific combination strategy."""
        if strategy == "problem_only":
            return item["Problem"]
        elif strategy == "rationale_only":
            return item["Rationale"]
        elif strategy == "problem_rationale":
            return item["Problem"] + separator + item["Rationale"]
        elif strategy == "problem_options":
            parsed_options = self._parse_options(item["options"])
            return item["Problem"] + "\n" + parsed_options
        elif strategy == "problem_options_rationale":
            parsed_options = self._parse_options(item["options"])
            return (
                item["Problem"]
                + "\n"
                + parsed_options
                + separator
                + item["Rationale"]
            )
        else:
            raise ValueError(f"Unknown text combination strategy: {strategy}")


class FortaSmartContractProcessor(DatasetProcessor):
    """Processor for Forta malicious smart contract dataset."""

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Forta smart contract dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Optional truncation (disabled by default, 0 = no limit)
        max_opcode_tokens = self.cfg.dataset.get("max_opcode_tokens", 0)

        for item in data:
            # Extract opcodes
            opcodes = item["decompiled_opcodes"]

            # Apply truncation if configured
            if max_opcode_tokens > 0:
                tokens = opcodes.split()
                if len(tokens) > max_opcode_tokens:
                    opcodes = " ".join(tokens[:max_opcode_tokens])

            texts.append(opcodes)

            # Convert malicious bool to safety label (malicious=True -> 0, False -> 1)
            label = 0 if item["malicious"] else 1
            labels.append(label)

        log.info(f"Processed Forta dataset with {len(texts)} samples")
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class CodeVulnerabilityProcessor(DatasetProcessor):
    """Processor for Code_Vulnerability_Labeled_Dataset from lemon42-ai.

    Dataset structure:
    - "code": source code text
    - "label": either "safe" or a vulnerability reason (e.g., "SQL injection", "XSS", etc.)
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Code Vulnerability Labeled Dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Verify required columns exist
        if "code" not in data.column_names:
            raise ValueError(
                f"Required column 'code' not found. Available columns: {data.column_names}"
            )
        if "label" not in data.column_names:
            raise ValueError(
                f"Required column 'label' not found. Available columns: {data.column_names}"
            )

        # Track label distribution for logging
        label_counts = {"safe": 0, "vulnerable": 0}
        vulnerability_types = {}

        for item in data:
            # Extract code text
            text = item["code"]
            texts.append(text)

            # Extract and process label
            raw_label = item["label"]

            # Convert label to safety label:
            # - "safe" (case-insensitive) -> safety label = 1 (safe)
            # - Any other string (vulnerability reason) -> safety label = 0 (unsafe/vulnerable)
            if isinstance(raw_label, str):
                label_lower = raw_label.lower().strip()
                if label_lower == "safe":
                    label = 1
                    label_counts["safe"] += 1
                else:
                    # Any non-"safe" label is a vulnerability
                    label = 0
                    label_counts["vulnerable"] += 1
                    # Track vulnerability types
                    if raw_label not in vulnerability_types:
                        vulnerability_types[raw_label] = 0
                    vulnerability_types[raw_label] += 1
            else:
                # Handle non-string labels (shouldn't happen, but be safe)
                log.warning(
                    f"Unexpected label type: {type(raw_label)}, value: {raw_label}"
                )
                label = 0 if raw_label else 1
                if label == 1:
                    label_counts["safe"] += 1
                else:
                    label_counts["vulnerable"] += 1

            labels.append(label)

        log.info(
            f"Processed Code Vulnerability dataset with {len(texts)} samples"
        )
        log.info(
            f"Label distribution: {label_counts['safe']} safe, "
            f"{label_counts['vulnerable']} vulnerable"
        )
        if vulnerability_types:
            log.info(
                f"Vulnerability types found: {len(vulnerability_types)} unique types"
            )
            # Log top 5 most common vulnerability types
            top_vulns = sorted(
                vulnerability_types.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for vuln_type, count in top_vulns:
                log.info(f"  - {vuln_type}: {count} samples")

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class CrossVulProcessor(DatasetProcessor):
    """Processor for CrossVul dataset from hitoshura25.

    Dataset structure:
    - "vulnerable_code": code with vulnerability (safety label 0)
    - "fixed_code": safe/fixed version of the code (safety label 1)

    Each sample produces two entries: one vulnerable (label=0) and one safe (label=1).
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process CrossVul dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Verify required columns exist
        if "vulnerable_code" not in data.column_names:
            raise ValueError(
                f"Required column 'vulnerable_code' not found. "
                f"Available columns: {data.column_names}"
            )
        if "fixed_code" not in data.column_names:
            raise ValueError(
                f"Required column 'fixed_code' not found. "
                f"Available columns: {data.column_names}"
            )

        vulnerable_count = 0
        safe_count = 0
        skipped_vulnerable = 0
        skipped_safe = 0

        for item in data:
            # Extract vulnerable code (label = 0, unsafe)
            vulnerable_code = item["vulnerable_code"]
            if (
                vulnerable_code
                and isinstance(vulnerable_code, str)
                and vulnerable_code.strip()
            ):
                texts.append(vulnerable_code)
                labels.append(0)  # Vulnerable = unsafe
                vulnerable_count += 1
            else:
                skipped_vulnerable += 1

            # Extract fixed code (label = 1, safe)
            fixed_code = item["fixed_code"]
            if (
                fixed_code
                and isinstance(fixed_code, str)
                and fixed_code.strip()
            ):
                texts.append(fixed_code)
                labels.append(1)  # Fixed = safe
                safe_count += 1
            else:
                skipped_safe += 1

        log.info(
            f"Processed CrossVul dataset with {len(data)} samples, "
            f"producing {len(texts)} code samples"
        )
        log.info(
            f"Label distribution: {safe_count} safe (fixed_code), "
            f"{vulnerable_count} vulnerable (vulnerable_code)"
        )
        if skipped_vulnerable > 0 or skipped_safe > 0:
            log.warning(
                f"Skipped {skipped_vulnerable} empty vulnerable_code entries, "
                f"{skipped_safe} empty fixed_code entries"
            )

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class CodeVulnerabilitySecurityDPOProcessor(DatasetProcessor):
    """Processor for CyberNative/Code_Vulnerability_Security_DPO dataset.

    Dataset structure:
    - "question": prompt/question text
    - "chosen": safe code (safety label 1)
    - "rejected": unsafe code (safety label 0)

    Each sample produces two entries: one safe (question + chosen) and one unsafe (question + rejected).
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Code Vulnerability Security DPO dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Verify required columns exist
        required_cols = ["question", "chosen", "rejected"]
        for col in required_cols:
            if col not in data.column_names:
                raise ValueError(
                    f"Required column '{col}' not found. "
                    f"Available columns: {data.column_names}"
                )

        safe_count = 0
        unsafe_count = 0
        skipped_safe = 0
        skipped_unsafe = 0

        for item in data:
            question = item.get("question", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            # Safe code: question + chosen (label = 1)
            if (
                question
                and chosen
                and isinstance(question, str)
                and isinstance(chosen, str)
            ):
                safe_text = f"{question}\n{chosen}".strip()
                if safe_text:
                    texts.append(safe_text)
                    labels.append(1)  # Safe
                    safe_count += 1
                else:
                    skipped_safe += 1
            else:
                skipped_safe += 1

            # Unsafe code: question + rejected (label = 0)
            if (
                question
                and rejected
                and isinstance(question, str)
                and isinstance(rejected, str)
            ):
                unsafe_text = f"{question}\n{rejected}".strip()
                if unsafe_text:
                    texts.append(unsafe_text)
                    labels.append(0)  # Unsafe
                    unsafe_count += 1
                else:
                    skipped_unsafe += 1
            else:
                skipped_unsafe += 1

        log.info(
            f"Processed Code Vulnerability Security DPO dataset with {len(data)} samples, "
            f"producing {len(texts)} code samples"
        )
        log.info(
            f"Label distribution: {safe_count} safe (question + chosen), "
            f"{unsafe_count} unsafe (question + rejected)"
        )
        if skipped_safe > 0 or skipped_unsafe > 0:
            log.warning(
                f"Skipped {skipped_safe} empty safe entries, "
                f"{skipped_unsafe} empty unsafe entries"
            )

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class CodeDefectDetectionProcessor(DatasetProcessor):
    """Processor for semeru/code-code-DefectDetection dataset.

    Dataset structure:
    - "func": function code text
    - "target": label (1 = insecure, 0 = secure)

    Note: This dataset's label convention is opposite to ours:
    - target = 1 (insecure) -> safety label = 0 (unsafe)
    - target = 0 (secure) -> safety label = 1 (safe)

    Dataset has train/validation/test splits. We merge train+validation as "train".
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process Code Defect Detection dataset."""
        # Load all splits
        dataset_kwargs = {"path": self.cfg.dataset.name}
        if hasattr(self.cfg.dataset, "subset") and self.cfg.dataset.subset:
            dataset_kwargs["name"] = self.cfg.dataset.subset

        dataset = load_dataset(**dataset_kwargs)

        texts: List[str] = []
        labels: List[int] = []
        split_labels: List[str] = []  # Track which split each sample came from

        # Merge train and validation splits
        train_data = []
        if "train" in dataset:
            train_data.extend(dataset["train"])
        if "validation" in dataset:
            train_data.extend(dataset["validation"])

        # Process train+validation (merged)
        for item in train_data:
            if "func" not in item:
                continue
            func_text = item["func"]
            if (
                not func_text
                or not isinstance(func_text, str)
                or not func_text.strip()
            ):
                continue

            # Get target label and flip it
            target = item.get("target", 0)
            # target=1 (insecure) -> safety_label=0 (unsafe)
            # target=0 (secure) -> safety_label=1 (safe)
            safety_label = 1 - int(target)

            texts.append(func_text.strip())
            labels.append(safety_label)
            split_labels.append(
                "train"
            )  # Mark as train (merged train+validation)

        # Process test split
        if "test" in dataset:
            for item in dataset["test"]:
                if "func" not in item:
                    continue
                func_text = item["func"]
                if (
                    not func_text
                    or not isinstance(func_text, str)
                    or not func_text.strip()
                ):
                    continue

                # Get target label and flip it
                target = item.get("target", 0)
                safety_label = 1 - int(target)

                texts.append(func_text.strip())
                labels.append(safety_label)
                split_labels.append("test")

        # Count labels
        safe_count = sum(1 for label in labels if label == 1)
        unsafe_count = sum(1 for label in labels if label == 0)
        train_count = sum(1 for s in split_labels if s == "train")
        test_count = sum(1 for s in split_labels if s == "test")

        log.info(
            f"Processed Code Defect Detection dataset: "
            f"{train_count} train (merged train+validation), {test_count} test samples"
        )
        log.info(
            f"Label distribution: {safe_count} safe, {unsafe_count} unsafe"
        )

        texts, labels = self._limit_samples(texts, labels)
        # Also limit split_labels if samples were limited
        if len(split_labels) > len(texts):
            split_labels = split_labels[: len(texts)]

        # Return metadata with split information
        metadata = {
            "split_labels": split_labels,  # List indicating train/test for each sample
            "preserve_splits": True,  # Flag to indicate we want to preserve original splits
        }
        return texts, labels, metadata


class DiSCoProcessor(DatasetProcessor):
    """Processor for StonyBrookNLP/DiSCo dataset.

    Dataset structure:
    - "Secure Code": safe code (safety label 1)
    - "Vulnerable Code": unsafe code (safety label 0)

    Dataset has train/validation/test splits. We merge train+validation as "train".
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process DiSCo dataset."""
        # Load all splits
        dataset_kwargs = {"path": self.cfg.dataset.name}
        if hasattr(self.cfg.dataset, "subset") and self.cfg.dataset.subset:
            dataset_kwargs["name"] = self.cfg.dataset.subset

        dataset = load_dataset(**dataset_kwargs)

        texts: List[str] = []
        labels: List[int] = []
        split_labels: List[str] = []  # Track which split each sample came from

        # Merge train and validation splits
        train_data = []
        if "train" in dataset:
            train_data.extend(dataset["train"])
        if "validation" in dataset:
            train_data.extend(dataset["validation"])

        # Process train+validation (merged)
        for item in train_data:
            # Process Secure Code (label = 1, safe)
            secure_code = item.get("Secure Code", "")
            if (
                secure_code
                and isinstance(secure_code, str)
                and secure_code.strip()
            ):
                texts.append(secure_code.strip())
                labels.append(1)  # Safe
                split_labels.append("train")

            # Process Vulnerable Code (label = 0, unsafe)
            vulnerable_code = item.get("Vulnerable Code", "")
            if (
                vulnerable_code
                and isinstance(vulnerable_code, str)
                and vulnerable_code.strip()
            ):
                texts.append(vulnerable_code.strip())
                labels.append(0)  # Unsafe
                split_labels.append("train")

        # Process test split
        if "test" in dataset:
            for item in dataset["test"]:
                # Process Secure Code (label = 1, safe)
                secure_code = item.get("Secure Code", "")
                if (
                    secure_code
                    and isinstance(secure_code, str)
                    and secure_code.strip()
                ):
                    texts.append(secure_code.strip())
                    labels.append(1)  # Safe
                    split_labels.append("test")

                # Process Vulnerable Code (label = 0, unsafe)
                vulnerable_code = item.get("Vulnerable Code", "")
                if (
                    vulnerable_code
                    and isinstance(vulnerable_code, str)
                    and vulnerable_code.strip()
                ):
                    texts.append(vulnerable_code.strip())
                    labels.append(0)  # Unsafe
                    split_labels.append("test")

        # Count labels
        safe_count = sum(1 for label in labels if label == 1)
        unsafe_count = sum(1 for label in labels if label == 0)
        train_count = sum(1 for s in split_labels if s == "train")
        test_count = sum(1 for s in split_labels if s == "test")

        log.info(
            f"Processed DiSCo dataset: "
            f"{train_count} train (merged train+validation), {test_count} test samples"
        )
        log.info(
            f"Label distribution: {safe_count} safe (Secure Code), {unsafe_count} unsafe (Vulnerable Code)"
        )

        texts, labels = self._limit_samples(texts, labels)
        # Also limit split_labels if samples were limited
        if len(split_labels) > len(texts):
            split_labels = split_labels[: len(texts)]

        # Return metadata with split information
        metadata = {
            "split_labels": split_labels,  # List indicating train/test for each sample
            "preserve_splits": True,  # Flag to indicate we want to preserve original splits
        }
        return texts, labels, metadata


class MCanogluDefectDetectionProcessor(DatasetProcessor):
    """Processor for mcanoglu/defect-detection dataset.

    Dataset structure:
    - "code": source code text
    - "label": integer label (0 = unsafe, 1 = safe)

    Label convention matches ours: 0 = unsafe, 1 = safe (no flipping needed).
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process MCanoglu Defect Detection dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Verify required columns exist
        if "code" not in data.column_names:
            raise ValueError(
                f"Required column 'code' not found. Available columns: {data.column_names}"
            )
        if "label" not in data.column_names:
            raise ValueError(
                f"Required column 'label' not found. Available columns: {data.column_names}"
            )

        # Track label distribution for logging
        safe_count = 0
        unsafe_count = 0
        skipped = 0

        for item in data:
            # Extract code text
            code_text = item.get("code", "")
            if (
                not code_text
                or not isinstance(code_text, str)
                or not code_text.strip()
            ):
                skipped += 1
                continue

            texts.append(code_text.strip())

            # Extract label (0 = unsafe, 1 = safe) - no flipping needed
            label = item.get("label", 0)
            # Ensure label is an integer
            if isinstance(label, (int, float)):
                label = int(label)
            else:
                log.warning(
                    f"Unexpected label type: {type(label)}, value: {label}. Using 0 (unsafe) as default."
                )
                label = 0

            # Validate label is 0 or 1
            if label not in [0, 1]:
                log.warning(
                    f"Label value {label} is not 0 or 1. Using 0 (unsafe) as default."
                )
                label = 0

            labels.append(label)

            if label == 1:
                safe_count += 1
            else:
                unsafe_count += 1

        log.info(
            f"Processed MCanoglu Defect Detection dataset with {len(texts)} samples"
        )
        log.info(
            f"Label distribution: {safe_count} safe (label=1), {unsafe_count} unsafe (label=0)"
        )
        if skipped > 0:
            log.warning(f"Skipped {skipped} empty code entries")

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class DiverseVulProcessor(DatasetProcessor):
    """Processor for claudios/DiverseVul dataset.

    Dataset structure:
    - "func": function code text
    - "target": label (0 = non-vulnerable/safe, 1 = vulnerable/unsafe)

    Note: This dataset's label convention:
    - target = 0 (non-vulnerable) -> safety label = 1 (safe)
    - target = 1 (vulnerable) -> safety label = 0 (unsafe)

    The text field wraps the function code in a prompt:
    "Examine whether the following code has any CWE vulnerabilities: {func}"
    """

    def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """Load and process DiverseVul dataset."""
        data = self._load_dataset()

        texts: List[str] = []
        labels: List[int] = []

        # Verify required columns exist
        if "func" not in data.column_names:
            raise ValueError(
                f"Required column 'func' not found. Available columns: {data.column_names}"
            )
        if "target" not in data.column_names:
            raise ValueError(
                f"Required column 'target' not found. Available columns: {data.column_names}"
            )

        # Get prompt template from config, with default
        prompt_template = self.cfg.dataset.get(
            "prompt_template",
            "Examine whether the following code has any CWE vulnerabilities: {func}",
        )

        # Track label distribution for logging
        safe_count = 0
        unsafe_count = 0
        skipped = 0

        for item in data:
            # Extract function code
            func_text = item.get("func", "")
            if (
                not func_text
                or not isinstance(func_text, str)
                or not func_text.strip()
            ):
                skipped += 1
                continue

            # Wrap function code in prompt
            text = prompt_template.format(func=func_text.strip())
            texts.append(text)

            # Extract target label and convert to safety label
            target = item.get("target", 0)
            # Ensure target is an integer
            if isinstance(target, (int, float)):
                target = int(target)
            else:
                log.warning(
                    f"Unexpected target type: {type(target)}, value: {target}. Using 0 (non-vulnerable) as default."
                )
                target = 0

            # Convert target to safety label:
            # target = 0 (non-vulnerable) -> safety_label = 1 (safe)
            # target = 1 (vulnerable) -> safety_label = 0 (unsafe)
            safety_label = 1 - int(target)

            # Validate safety_label is 0 or 1
            if safety_label not in [0, 1]:
                log.warning(
                    f"Computed safety_label {safety_label} is not 0 or 1. Using 0 (unsafe) as default."
                )
                safety_label = 0

            labels.append(safety_label)

            if safety_label == 1:
                safe_count += 1
            else:
                unsafe_count += 1

        log.info(f"Processed DiverseVul dataset with {len(texts)} samples")
        log.info(
            f"Label distribution: {safe_count} safe (target=0), {unsafe_count} unsafe (target=1)"
        )
        if skipped > 0:
            log.warning(f"Skipped {skipped} empty func entries")

        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class HHRLHFHelpfulProcessor(DatasetProcessor):
    """Processor for Anthropic HH-RLHF helpful split.

    Each example has 'chosen' and 'rejected' fields containing
    multi-turn conversations as strings with
    '\\n\\nHuman: ...' and '\\n\\nAssistant: ...' markers.
    We use the full 'chosen' conversation (human prompt + assistant reply).
    """

    def load_and_process(
        self,
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        data = self._load_dataset()

        texts = []
        labels = []
        default_label = self.cfg.dataset.get("default_label", 1)

        for item in data:
            chosen = item.get("chosen", "")
            if not chosen.strip():
                continue
            texts.append(chosen.strip())
            labels.append(default_label)

        log.info(
            f"Processed HH-RLHF helpful dataset with {len(texts)} samples"
        )
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


class HellaSwagProcessor(DatasetProcessor):
    """Processor for Rowan/hellaswag dataset.

    Each example has 'ctx' (context) and 'endings' (list of 4 options)
    with 'label' indicating the correct ending index.
    We format as full context + correct answer.
    """

    def load_and_process(
        self,
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        data = self._load_dataset()

        texts = []
        labels = []
        default_label = self.cfg.dataset.get("default_label", 1)
        skipped = 0

        for item in data:
            ctx = item.get("ctx", "")
            endings = item.get("endings", [])
            label_idx = item.get("label", "")

            # label can be string or int depending on split
            try:
                label_idx = int(label_idx)
            except (ValueError, TypeError):
                skipped += 1
                continue

            if label_idx < 0 or label_idx >= len(endings):
                skipped += 1
                continue

            text = ctx.strip() + " " + endings[label_idx].strip()
            texts.append(text)
            labels.append(default_label)

        log.info(f"Processed HellaSwag dataset with {len(texts)} samples")
        if skipped > 0:
            log.warning(f"Skipped {skipped} samples with invalid labels")
        texts, labels = self._limit_samples(texts, labels)
        return texts, labels, {}


# Registry of dataset processors
DATASET_PROCESSORS: Dict[str, Type[DatasetProcessor]] = {
    "toxigen/toxigen-data": ToxigenProcessor,
    "openai/gsm8k": GSM8KProcessor,
    "openai/openai_humaneval": HumanEvalProcessor,
    "evalplus/humanevalplus": HumanEvalProcessor,  # Reuse same processor
    "tatsu-lab/alpaca": AlpacaProcessor,
    "cais/mmlu": MMLUProcessor,
    "google-research-datasets/mbpp": MBPPProcessor,
    "allenai/ai2_arc": ARCProcessor,
    "evalplus/mbppplus": MBPPPlusProcessor,
    "allenai/math_qa": MathQAProcessor,
    "ReasoningTransferability/math_sft_40K": MathSFT40KProcessor,
    "forta/malicious-smart-contract-dataset": FortaSmartContractProcessor,
    "lemon42-ai/Code_Vulnerability_Labeled_Dataset": CodeVulnerabilityProcessor,
    "hitoshura25/crossvul": CrossVulProcessor,
    "CyberNative/Code_Vulnerability_Security_DPO": CodeVulnerabilitySecurityDPOProcessor,
    "semeru/code-code-DefectDetection": CodeDefectDetectionProcessor,
    "StonyBrookNLP/DiSCo": DiSCoProcessor,
    "mcanoglu/defect-detection": MCanogluDefectDetectionProcessor,
    "claudios/DiverseVul": DiverseVulProcessor,
    "Anthropic/hh-rlhf": HHRLHFHelpfulProcessor,
    "Rowan/hellaswag": HellaSwagProcessor,
}


def create_dataset_processor(cfg: DictConfig) -> DatasetProcessor:
    """Factory function to create appropriate dataset processor."""
    dataset_name = cfg.dataset.name

    # Check if we have a specific processor for this dataset
    if dataset_name in DATASET_PROCESSORS:
        processor_class = DATASET_PROCESSORS[dataset_name]
        log.info(
            f"Using {processor_class.__name__} for dataset: {dataset_name}"
        )
        return processor_class(cfg)
    else:
        # Fall back to generic processor
        log.info(f"Using GenericProcessor for dataset: {dataset_name}")
        return GenericProcessor(cfg)
