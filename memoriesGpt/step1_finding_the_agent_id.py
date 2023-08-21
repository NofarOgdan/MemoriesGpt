import csv
import pandas as pd

narrowed_ds = pd.read_csv("GPT_only_correct_memory_components_fixed - correct_gpt_memory_components.csv")
extended_ds = pd.read_csv("extended_memNet_ss.csv")

agent_id_lst = []
for i in range(len(narrowed_ds)):
    group_id = narrowed_ds.iloc[i]['group_id']
    agent_id = extended_ds[extended_ds['group_id'] == group_id]['agent_id'].values[0]
    agent_id_lst.append(agent_id)
narrowed_ds['agent_id'] = agent_id_lst
narrowed_ds.to_csv(index=False, path_or_buf="GPT_only_correct_memory_components_fixed_with_agent_id.csv")