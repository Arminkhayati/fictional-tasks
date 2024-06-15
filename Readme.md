# Project status:


## Feature extraction
Feature extraction, feature processing and feature analyzing is done on these two datasets: 

- NoteChat dataset from hugging face (on 100 data points)
- MTS dataset (on 250 data points)

My approach is motivated by these papers:

- Large Language Models are Few-Shot Clinical Information Extractors
- Product Attribute Value Extraction using Large Language Models
- An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters

Technically there are 3 broad approaches for information extraction using LLMs (here I used the last one combined with zero shot, one shot or few shot prompting):

- Tool/Function Calling Mode: Some LLMs support a tool or function calling mode. These LLMs can structure output according to a given schema. Generally, this approach is the easiest to work with and is expected to yield good results.

- JSON Mode: Some LLMs are can be forced to output valid JSON. This is similar to tool/function Calling approach, except that the schema is provided as part of the prompt. Generally, our intuition is that this performs worse than a tool/function calling approach, but don't trust us and verify for your own use case!

- Prompting Based: LLMs that can follow instructions well can be instructed to generate text in a desired format. The generated text can be parsed downstream using existing Output Parsers or using custom parsers into a structured format like JSON. This approach can be used with LLMs that do not support JSON mode or tool/function calling modes. This approach is more broadly applicable, though may yield worse results than models that have been fine-tuned for extraction or function calling.

## Algorithm implementation:

I implemented the requested algorithm using two attribution methods:

- Gradient of inputs
- Expected Gradients
