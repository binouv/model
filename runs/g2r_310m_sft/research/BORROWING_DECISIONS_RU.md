# Qwen / DeepSeek / small reasoners: решения для FlyGraph

Обзор 2026-09-26. Это чтение первоисточников и проектные решения, не новые результаты моделей. Чужие weights, teacher answers и source files не импортированы. G2R не меняет архитектуру G1, иначе влияние нового loss нельзя отделить.

## Qwen3.5-4B

Источники: https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json и https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py . Конкретная конфигурация имеет32слоя, три linear_attention на один full_attention, output gating и tied embeddings. В implementation просмотрены Q/K normalization, sigmoid output gate, initial/final recurrent state, convolution state и FP32 decay GatedDeltaNet.

Переносимая идея — настоящее межтокенное состояние с chunk/cache equivalence и периодическим full attention. Граф G1 выполняет GRU-раунды внутри одного токена и не реализует такую память. Сжатый recurrent state нельзя подменять точным журналом адресных записей. Готовый headwise-gate prototype пока только correctness-tested, не trained quality gain.

## DeepSeek Engram

Источники: https://github.com/deepseek-ai/Engram , https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py , https://arxiv.org/abs/2601.07372 . Просмотрены CompressedTokenizer, NgramHashMapping, MultiHeadEmbedding, Engram.forward. N-gram ключи адресуют статические обучаемые таблицы; context-dependent query/key gating и causal convolution вводят value в backbone.

Это не mutable episodic memory. Demo явно оставляет Attention/MoE заглушками: не готовый полный training stack. Lowercase/StripAccents в CompressedTokenizer могут смешать case-sensitive code identifiers и protected keys. Такая нормализация недопустима для канонических адресов. Хеш-коллизии тоже не доказывают идентичность. Для нашей цели сначала reasoning; огромные lookup tables не добавляются и должны считаться в total parameters/RAM.

## DeepSeek-V3/R1

Источники: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py , https://arxiv.org/abs/2412.19437 , https://github.com/deepseek-ai/DeepSeek-R1 . Проверены MLA/cache и MTP/дистилляция. Наш GQA хранит2*4*64=512чисел/token/layer; буквальные V3 kv_rank512+rotary64 дали бы576. Нужен собственный размер и парный тест, не обещание автоматической экономии.

Полезнее проверяемые решения учителя: independent arithmetic replay, exact memory trace, isolated code tests. R1 distillation опирается на уже обученную языковую базу. Token-level KL нельзя напрямую применять между нашим8K и несовместимым Qwen vocabulary; sequence-level verified targets проще. Ни teacher API, ни RL в G2R не запускались.

## MobileLLM-R1

Источники: https://github.com/facebookresearch/MobileLLM-R1 , https://arxiv.org/abs/2509.24945 . Официальный рецепт содержит140M/360M/950M configs, pretraining, mid-training KL от Llama-3.1-8B-Instruct, general/reasoning SFT. Это не свидетельство возможности обучить random-init300M за256шагов. G2R заимствует узкую идею фокуса supervised loss на ответе и делает один matched control. FAIR Noncommercial Research условия требуют отдельной проверки перед переносом кода/весов.

## TinyRecursiveModels

Источники: https://github.com/SamsungSAILMontreal/TinyRecursiveModels , https://arxiv.org/abs/2510.04871 . Полезны повторное уточнение latent/answer states и deep supervision; опубликованные puzzle results не являются доказательством общей языковой интеллектуальности7Mмодели. Повторные вычисления не бесплатны: сравнивать actual FLOPs и качество при фиксированном бюджете. Нынешний G1 readout не воспроизводит этот механизм.

## Решение

Закончить paired G2R без скрытых изменений. Измерять exact generated answers, unseen surfaces, extrapolation и both-correct counterfactual pairs. Teacher-forced accuracy/EOS/loss отдельно от solving accuracy. Следующий architecture experiment должен менять один фактор. Mutable source main URLs в обзоре не являются immutable vendoring pins; actual copies потребуют commit, LICENSE/NOTICE и условий данных. Текущий held не является GSM8K/MBPP и не сравнивался с Qwen.
