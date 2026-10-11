# Supplemental public-code diagnostic — fixed before endpoint evaluation

This does not change the primary L100NC training protocol, datasets, endpoints, parser or success gates. No extra training is performed on public problems.

Deterministic selection: the first ten sanitized MBPP problems by task_id in the official test interval11..510:11,12,14,16,17,18,19,20,56,57. Descriptions and test input/output pairs transcribed from Google Research's public mbpp/sanitized-mbpp.json. The original task20 wording is preserved, not silently corrected. Fixture SHA25630faec7586778c2edddae286357708be520822d393eb61198faa91bc6152e72a.

Model sees only original description plus required function name/signature and a request for Python code. No examples, reference solution or test outputs are shown. Full8192vocab greedy256newtokens, no model-output repair or retry. Native original signatures are scored, not renamed to solve. Resource-bounded subprocess with a documented restricted Python AST, safe builtins and selected standard-library imports. All ten independent reference programs passed the30official test I/O pairs before model evaluation.

This is a CUSTOM ZERO-SHOT10CASE smoke, NOT full MBPP and NOT the original three-shot/provided-tests protocol. It is too small to establish universal coding ability or Qwen parity. Syntax restrictions may reject otherwise valid programs; such errors are separately reported. No Qwen output exists on this fixture in this run. No historical contamination guarantee: only known current training source excludes these cases.

Sources: https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json and mbpp/README.md. Original authors: Jacob Austin et al., Program Synthesis with Large Language Models (2021), Google Research. Root repository README declares datasets CC BY4.0; https://creativecommons.org/licenses/by/4.0/legalcode . Description/test data retained; JSON representation of test calls and added entry signatures are our adaptation. New checker/reference scripts are our code. Do not confuse reading reference data in the evaluator with providing answers to the model.
