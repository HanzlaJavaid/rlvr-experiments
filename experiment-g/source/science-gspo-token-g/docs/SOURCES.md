# Primary sources inherited from Experiment F

These links document the pinned F runtime inherited by G. They are historical source references, not a claim of a new online review or a completed G run.

- Qwen model card and general-thinking sampling preset: https://huggingface.co/Qwen/Qwen3.5-4B
- Immutable model metadata: https://huggingface.co/api/models/Qwen/Qwen3.5-4B
- Pinned config: https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json
- Pinned chat template: https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja
- Transformers text-only Qwen3.5 implementation: https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- vLLM hybrid model implementation: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_5.py
- TRL config and colocated sampling implementation: https://github.com/huggingface/trl/tree/v1.17.0/trl
- TRL token/sequence inference correction and truncation/masking definitions: https://huggingface.co/docs/trl/grpo_trainer
- Transformers DeltaNet optional-kernel requirements: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Flash Linear Attention installation and backend extras: https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md
- Causal convolution package and installation: https://github.com/Dao-AILab/causal-conv1d
- Learning-rate testing motivation, not a directly transferable optimal setting: https://thinkingmachines.ai/blog/lora/

The proposed2e-6 learning rate is an experimental implementation choice. No cited source guarantees improved scientific results or sufficient memory on a particular rented GPU.
