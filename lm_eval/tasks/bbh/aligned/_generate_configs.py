"""Generate per-subtask YAML configs for BBH aligned evaluation."""

import yaml

TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

DESCRIPTIONS = {
    "boolean_expressions": "Evaluate the result of a random Boolean expression.",
    "causal_judgement": "Answer a yes/no question about causation.",
    "date_understanding": "Infer the date from context.",
    "disambiguation_qa": "Clarify the meaning of sentences with ambiguous pronouns.",
    "dyck_languages": "Predict the sequence of closing parentheses of a Dyck-4 word.",
    "formal_fallacies": "Distinguish deductively valid syllogisms from fallacious ones.",
    "geometric_shapes": "Name geometric shapes from their SVG paths.",
    "hyperbaton": "Order adjectives correctly in English sentences.",
    "logical_deduction_five_objects": "Deduce the order of a sequence of objects.",
    "logical_deduction_seven_objects": "Deduce the order of a sequence of objects.",
    "logical_deduction_three_objects": "Deduce the order of a sequence of objects.",
    "movie_recommendation": "Recommend movies similar to the given list.",
    "multistep_arithmetic_two": "Solve multi-step arithmetic problems.",
    "navigate": "Determine the final position after navigating.",
    "object_counting": "Count objects mentioned in the question.",
    "penguins_in_a_table": "Answer questions about a table of penguins.",
    "reasoning_about_colored_objects": "Answer questions about colored objects.",
    "ruin_names": "Choose the humorous edit of an artist or movie name.",
    "salient_translation_error_detection": "Detect the type of translation error.",
    "snarks": "Determine which sentence is sarcastic.",
    "sports_understanding": "Determine if a sports sentence is plausible.",
    "temporal_sequences": "Answer questions about temporal ordering.",
    "tracking_shuffled_objects_five_objects": "Track shuffled objects.",
    "tracking_shuffled_objects_seven_objects": "Track shuffled objects.",
    "tracking_shuffled_objects_three_objects": "Track shuffled objects.",
    "web_of_lies": "Evaluate a Boolean function from a word problem.",
    "word_sorting": "Sort a list of words.",
}

PROMPT = (
    "Q: {{input}}\n"
    "A: Let's think step by step. "
    'Put your final answer in the format of "So the answer is [ANSWER]" '
    "(without quotes) where [ANSWER] is your answer."
)


def main():
    group_tasks = []

    for task_name in TASKS:
        yaml_task_name = f"bbh_aligned_{task_name}"
        group_tasks.append(yaml_task_name)
        desc = DESCRIPTIONS.get(task_name, "")

        config = {
            "dataset_name": task_name,
            "description": f"{desc}\n\n",
            "doc_to_text": PROMPT,
            "include": "_bbh_aligned_template_yaml",
            "task": yaml_task_name,
        }

        with open(f"{task_name}.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Write group file
    group_config = {
        "group": "bbh_aligned",
        "task": group_tasks,
        "aggregate_metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "weight_by_size": True,
            }
        ],
        "metadata": {"version": 1.0},
    }

    with open("_bbh_aligned.yaml", "w") as f:
        yaml.dump(group_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Generated {len(TASKS)} subtask configs + 1 group config")


if __name__ == "__main__":
    main()
