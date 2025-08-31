import torch
import numpy as np

from textattack.models.wrappers import ModelWrapper
from seqattack.utils import get_tokens

from seqattack.utils import postprocess_ner_output
from seqattack.models.exceptions import PredictionError

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)


class NERModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, postprocess_func, name:str = None):
        self.model = model
        self.tokenizer = tokenizer

        # The post-processing function must accept three arguments:
        #
        # original_text: AttackedText instance of the original text
        # model_output: the model outputs
        # tokenized_input: the tokenized original text
        self.postprocess_func = postprocess_func
        self.name = name

    def __call__(self, text_inputs_list, raise_excs=False):
        """
            Get the model predictions for the input texts list

            :param raise_excs:  whether to raise an exception for a
                                prediction error
        """
        encoded = self.encode(text_inputs_list)
        with torch.no_grad():
            outputs = [self._predict_single(x) for x in encoded] # we have B x T x C, and that triggers the assert statement at the 
            # end of _call_model_uncached in goal_function.py inside textattack. We could have fixed it here, but 
            # we would like to keep the original outputs for further processing down the line.
        outputs = np.array(outputs)
        formatted_outputs = []

        for model_output, original_text in zip(outputs, text_inputs_list):
            tokenized_input = get_tokens(original_text, self.tokenizer)
            # If less predictions than the input tokens are returned skip the sample
            if len(tokenized_input) != len(model_output):
                error_string = f"Tokenized text and model predictions differ in length! (preds: {len(model_output)} vs tokenized: {len(tokenized_input)}) for sample: {original_text}"
                # print(self.tokenizer.decode(encoded[0]))
                print(f"Skipping sample: {tokenized_input}, len(tokenized_input)={len(tokenized_input)}, len(model_output)={len(model_output)}")

                if raise_excs:
                    raise PredictionError(error_string)
                else:
                    continue

            formatted_outputs.append(model_output)
        return np.array(formatted_outputs)

    def process_raw_output(self, raw_output, text_input):
        """
            Returns the output as a list of numeric labels
        """
        tokenized_input = get_tokens(text_input, self.tokenizer)

        return self.postprocess_func(
            text_input,
            raw_output,
            tokenized_input)[1]

    def _predict_single(self, encoded_sample):
        outputs = self.model(encoded_sample)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits[0].cpu().numpy()
        else:
            logits = outputs[0][0].cpu().numpy()
        return np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

    def encode(self, inputs):
        processed_inputs = []
        for x in inputs:
            # Handle AttackedText objects
            if hasattr(x, 'text'):
                text = x.text
            else:
                text = str(x)
            encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, padding=False, truncation=True)['input_ids']
            processed_inputs.append(encoded)
        return processed_inputs

    def _tokenize(self, inputs):
        processed_inputs = []
        for x in inputs:
            # Handle AttackedText objects
            if hasattr(x, 'text'):
                text = x.text
            else:
                text = str(x)
            tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer(text, add_special_tokens=True)['input_ids'])
            processed_inputs.append(tokens)
        return processed_inputs

    @classmethod
    def load_huggingface_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        return tokenizer, NERModelWrapper(
            model,
            tokenizer,
            postprocess_func=postprocess_ner_output,
            name=model_name)
