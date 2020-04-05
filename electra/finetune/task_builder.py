# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns task instances given the task name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import electra.configure_finetuning
from electra.finetune.qa import qa_tasks
from electra.model import tokenization


def get_tasks(config: electra.configure_finetuning.FinetuningConfig):
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                           do_lower_case=config.do_lower_case)
    return [get_task(config, task_name, tokenizer)
            for task_name in config.task_names]


def get_task(config: electra.configure_finetuning.FinetuningConfig, task_name,
             tokenizer):
    """Get an instance of a task based on its name."""
    if task_name == "squad":
        return qa_tasks.SQuAD(config, tokenizer)
    elif task_name == "squadv1":
        return qa_tasks.SQuADv1(config, tokenizer)
    elif task_name == "newsqa":
        return qa_tasks.NewsQA(config, tokenizer)
    elif task_name == "naturalqs":
        return qa_tasks.NaturalQuestions(config, tokenizer)
    elif task_name == "triviaqa":
        return qa_tasks.TriviaQA(config, tokenizer)
    elif task_name == "searchqa":
        return qa_tasks.SearchQA(config, tokenizer)
    else:
        raise ValueError("Unknown task " + task_name)
