"""Generate per-subject YAML configs for MMLU-Pro aligned evaluation."""

import yaml

SUBJECTS = [
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def main():
    task_names = []

    for subject in SUBJECTS:
        task_name = f"mmlu_pro_{subject}_aligned"
        task_names.append(task_name)

        desc_subject = subject.replace("_", " ")
        # process_docs function name uses the subject with underscores
        process_fn = f"utils.process_{subject}"

        config = {
            "description": (
                f"The following are multiple choice questions (with answers) "
                f"about {desc_subject}.\n\n"
            ),
            "include": "_default_template_yaml",
            "task": task_name,
            "task_alias": subject,
            "process_docs": f"!function {process_fn}",
        }

        fname = f"mmlu_pro_{subject}.yaml"
        with open(fname, "w") as f:
            # Manual write to preserve !function tag
            f.write(f'description: "The following are multiple choice questions (with answers) about {desc_subject}.\\n\\n"\n')
            f.write(f'include: "_default_template_yaml"\n')
            f.write(f'task: "{task_name}"\n')
            f.write(f'task_alias: "{subject}"\n')
            f.write(f"process_docs: !function {process_fn}\n")

    # Write group file
    with open("_mmlu_pro_aligned.yaml", "w") as f:
        group_config = {
            "group": "mmlu_pro_aligned",
            "task": task_names,
            "aggregate_metric_list": [
                {
                    "aggregation": "mean",
                    "metric": "exact_match",
                    "weight_by_size": True,
                }
            ],
            "metadata": {"version": 1},
        }
        yaml.dump(group_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Generated {len(SUBJECTS)} subject configs + 1 group config")


if __name__ == "__main__":
    main()
