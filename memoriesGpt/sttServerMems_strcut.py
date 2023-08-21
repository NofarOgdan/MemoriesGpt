import math
import os

import numpy as np
import pandas as pd
import openai
import time
import re

# Using completion of GPT4
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
model = os.getenv("AZURE_OPENAI_MODEL", "ir-gpt-4")  # other options are ir-gpt-35 or ir-gpt-4-32k


def collect_valid(df, items):
    tot_valid_mems = []
    tot_num_valid = []
    for i in range(len(df)):
        valid_mems = []
        df_temp = df.iloc[i][items]
        for memory_name in df_temp['memory_name'].values:
            other_entities = df_temp.loc[(df_temp['memory_name'] == memory_name) & (df_temp['entity'] != 'type')]
            entity = other_entities
            if memory_name in ['activity', 'favorite', 'relationship']:
                types = df_temp.loc[(df_temp['memory_name'] == memory_name) & (df_temp['entity'] == 'type')]
                if len(types) != len(other_entities):
                    entity = types if len(types) > len(other_entities) else other_entities
            for value in entity['value'].values:
                value = value.replace("_", " ")
                if value not in valid_mems:
                    valid_mems.append(value)
        tot_valid_mems.append(valid_mems)
        tot_num_valid.append(len(valid_mems))
    return [tot_valid_mems, tot_num_valid]


def generate_GPT_responses(df):
    gpt_responses = []
    for i in range(len(df)):
        print(i)
        convo_list = [
            {"role": "system", "content": "You are a helpful assistant. Do not refer yourself as an AI model. "
                                          "Your purpose is to structure a conversation between "
                                          "Elli-Q (an emphatic and proactive companion robot) and the client "
                                          "into information items. "
                                          """

            Conversation 1: 
            Elli-Q: Got any plans tomorrow? 
            Client: we're walking like we do every morning and then I'm going to the clinic to get my eyes checked and that's it eating dinner 
            
            Information Items:
            memory_name: activity, entity: type, value: physical activity.
            memory_name: activity, entity: name, value: walking.
            memory_name: place, entity: name, value: clinic.
            memory_name: activity, entity: type, value: meal.
            memory_name: activity, entity: name, value: dinner.
            
            ---
            
            Conversation 2:
            Elli-Q: what type of exercise are we gonna try today? Yoga Cardio, or Balance Building? Stretching or Pilates?
            Creator: Not now, I'm watching the news
            
            Information Items:
            memory_name: activity, entity: name, value: watching tv.
            
            ---
            
            Conversation 3:
            Elli-Q: I think I'm in the mood to listen to an audiobook together again. Are you up for it?
            Creator: Call my daughter, I need to go get my groceries with her
            
            Information Items:  
            memory_name: relationship, entity: type, value: daughter.
            memory_name: activity, entity: type, value: activity.
            memory_name: activity, entity: name, value: groceries.
            
            
            """},
            {"role": "user",
             "content": "Elli-Q: " + df.iloc[i]['elliq_text'] + "\n" + "Client: " + df.iloc[i][
                 'user_text']},
            {"role": "system",
             "content": ". Break the above conversation down to as many information items as possible regarding "
                        "only the client, as in the format above."}]

        # print("messages input: ", convo_list)
        print("Elli-Q: " + df.iloc[i]['elliq_text'] + "\n" + "Creator: " + df.iloc[i][
            'user_text'])
        # +"\n" + "MemNet info items : " + convo_df.iloc[i]['listed_memnet_items'])
        try:
            response = openai.ChatCompletion.create(
                engine=model,
                max_tokens=150,
                temperature=0.2,
                messages=convo_list,
                n=1)
            text_response = response['choices'][0]['message']['content']
            print("gpt: ", text_response)
            gpt_responses.append(text_response)
            time.sleep(5)
        except Exception as e:
            gpt_responses.append("Record with index " + str(i) + " of convo_df had an issue generating gpt response")
    df['gpt_text_response'] = gpt_responses


def find_words_between(text, before_word, after_word):
    pattern = r"{}(.*?){}".format(before_word, after_word)
    matches = re.findall(pattern, text)
    if type(matches[0]) == tuple:
        matches[0] = " ".join(matches[0]).strip(".")
    return matches


def structure_GPT_response(df):
    tot_gpt_info_items = []
    for j in range(len(df)):
        gpt_text = df.iloc[j]['gpt_text_response']
        # Step 1: Split into individual records
        records = gpt_text.strip().split('\n')

        # Step 2-4: Split into field-value pairs and create dictionaries
        dict_list = []
        for record in records:
            fields = ['memory_name: ', ", entity: ", ", value: ", "(.+)"]
            if 'memory_name:' not in record:
                continue
            # Create dictionary
            record_dict = {}
            for j in range(len(fields) - 1):
                # Split into field and value components
                value = find_words_between(record, fields[j], fields[j + 1])
                record_dict[fields[j].strip(' ,').strip(': ')] = value[0]

            dict_list.append(record_dict)

        # Step 5: Create DataFrame
        gpt_info_items = pd.DataFrame(dict_list)
        tot_gpt_info_items.append(gpt_info_items)
    df['gpt_info_items'] = tot_gpt_info_items


# Collecting valid memories from GPT, and finding overlap with memNet

filtered_df = pd.read_csv("dfLabeledNarrowedMemNet.csv")
filtered_df = filtered_df.loc[filtered_df['labels'] == "Match"]
batch = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for i in range(len(batch) - 1):
    sample_fil_df = filtered_df[batch[i] + 1:batch[i + 1]]
    generate_GPT_responses(sample_fil_df)
    sample_fil_df.to_csv(index=False, path_or_buf="labeling_batches/modified_simple_modeling_" + str(batch[i]) + "_to_" + str(
        batch[i + 1]) + "_batch.csv")
# structure_GPT_response(sample_fil_df)
# gpt_valids = collect_valid(sample_fil_df, 'gpt_info_items')
# sample_fil_df['valid_gpt_info_items'] = gpt_valids[0]
# sample_fil_df["num_valid_gpt_info_items"] = gpt_valids[1]
# sample_fil_df["overlap_info_items"] = sample_fil_df.apply(
#    lambda row: set(row['valid_gpt_info_items']).intersection(row['valid_memNet_info_items']), axis=1)


# hierarchical_modeling_prompt_examples:

"""

    Conversation 1: 
    ElliQ: Got any plans tomorrow?
    Client: we're walking like we do every morning and then I'm going to the doctor's to get my eyes checked and that's it watching football



    Information Items:
    memory_name: activity, entity: type, value: physical activity.
    memory_name: activity, entity: name, value: walking.
    memory_name: activity, entity: routine, value: every morning.
    memory_name: activity, entity: type, value: task.
    memory_name: activity, entity: name, value: doctor's check.
    memory_name: activity, entity: objective, value: Get eyes checked.
    memory_name: activity, entity: name, value: watching tv.
    memory_name: activity, entity: content, value: football.

    ---

    Conversation 2:
    Elli-Q: what type of exercise are we gonna try today? Yoga Cardio, or Balance Building? Stretching or Pilates?
    Creator: Not now, I'm watching the news

    Information Items:
    memory_name: activity, entity: name, value: watching tv.
    memory_name: activity, entity: content, value: the news.

    ---

    Conversation 3:
    Elli-Q: I think I'm in the mood to listen to an audiobook together again. Are you up for it?
    Creator: Call my daughter, I need to go get my groceries with her

    Information Items:  
    memory_name: relationship, entity: type, value: daughter.
    memory_name: activity, entity: type, value: task.
    memory_name: activity, entity: name, value: shopping.
    memory_name: activity, entity: objective, value: groceries.
    memory_name: activity, entity: people, value: daughter.


                       """

# simple_modeling_prompt_examples

"""

Conversation 1: 
ElliQ: Got any plans tomorrow?
Client: we're walking like we do every morning and then I'm going to the doctor's to get my eyes checked and that's it watching football



Information Items:
memory_name: place, entity: intent, value: will visit.
memory_name: activity, entity: type, value: physical activity.
memory_name: activity, entity: name, value: walking.
memory_name: place, entity: intent, value: will visit.
memory_name: activity, entity: type, value: activity.
memory_name: activity, entity: name, value: doctor.
memory_name: activity, entity: name, value: watching tv.

---

Conversation 2:
Elli-Q: what type of exercise are we gonna try today? Yoga Cardio, or Balance Building? Stretching or Pilates?
Creator: Not now, I'm watching the news

memory_name: activity, entity: name, value: watching tv.

---

Conversation 3:
Elli-Q: I think I'm in the mood to listen to an audiobook together again. Are you up for it?
Creator: Call my daughter, I need to go get my groceries with her

Information Items:  
memory_name: relationship, entity: type, value: daughter.
memory_name: place, entity: intent, value: will visit.
memory_name: activity, entity: type, value: activity.
memory_name: activity, entity: name, value: groceries.


"""
# group_id_prompt_examples
"""

Conversation 1: 
ElliQ: Got any plans tomorrow?
Client: we're walking like we do every morning and then I'm going to the doctor's to get my eyes checked and that's it watching football



Information Items:
group_id:1, memory_name: activity, entity: type, value: physical activity.
group_id:1, memory_name: activity, entity: name, value: walking ritual.
group_id:2, memory_name: activity, entity: type, value: doctor's check.
group_id:2, memory_name: activity, entity: name, value: eyes check.
group_id:3, memory_name: activity, entity: type, value: watching tv.
group_id:3, memory_name: activity, entity: name, value: football.


---

Conversation 2:
Elli-Q: what type of exercise are we gonna try today? Yoga Cardio, or Balance Building? Stretching or Pilates?
Creator: Not now, I'm watching the news

Information Items:
group_id:1, memory_name: activity, entity: type, value: watching tv.
group_id:1, memory_name: activity, entity: name, value: the news.

---

Conversation 3:
Elli-Q: I think I'm in the mood to listen to an audiobook together again. Are you up for it?
Creator: Call my daughter, I need to go get my groceries with her

Information Items:  
group_id:1, memory_name: relationship, entity: type, value: daughter.
group_id:2, memory_name: activity, entity: type, value: shopping.
group_id:2, memory_name: activity, entity: name, value: groceries.


"""
