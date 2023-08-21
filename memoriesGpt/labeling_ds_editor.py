import pandas as pd
import re


def structure_GPT_response(df):
    tot_gpt_info_items = []
    for j in range(len(df)):
        gpt_text = df.iloc[j]['gpt_text_response']
        # Step 1: Split into individual records
        records = gpt_text.strip().split('\n')

        # Step 2-4: Split into field-value pairs and create dictionaries
        dict_list = []
        for record in records:
            fields = [", value: ", "(.+)"]
            if 'memory_name:' not in record:
                continue
            # Create dictionary
            record_dict = {}
            for j in range(len(fields) - 1):
                # Split into field and value components
                value = find_words_between(record, fields[j], fields[j + 1])
                record_dict[fields[j].strip(' ,').strip(': ')] = value[0].strip(' ')

            dict_list.append(record_dict)

        # Step 5: Create DataFrame
        gpt_info_items = pd.DataFrame(dict_list)
        tot_gpt_info_items.append(gpt_info_items)
    df['gpt_values'] = tot_gpt_info_items


def find_words_between(text, before_word, after_word):
    pattern = r"{}(.*?){}".format(before_word, after_word)
    matches = re.findall(pattern, text)
    if type(matches[0]) == tuple:
        matches[0] = " ".join(matches[0]).strip(".")
    return matches


def memnet_info_items():
    lst_memnet_values = []
    for j in range(len(sample_df)):
        memNet_items = ds_mem_ss.loc[(ds_mem_ss['group_id'] == sample_df.iloc[j]['group_id'])][
            ['memory_name', 'entity', 'value']].reset_index(drop=True)
        memNet_items = memNet_items.drop(
            memNet_items.index[(memNet_items['memory_name'] == 'place') & (memNet_items['value'] == 'will_visit')],
            axis=0, errors='ignore')
        memNet_items = memNet_items.drop(
            memNet_items.index[(memNet_items['memory_name'] == 'place') & (memNet_items['value'] == 'visited')], axis=0,
            errors='ignore')
        data = {
            'value': list(memNet_items['value'].str.replace("_", " ")),
            # Add other columns here if needed
        }
        lst_memnet_values.append(pd.DataFrame(data))
    sample_df['memnet_values'] = lst_memnet_values


def inner_and_outer_join():
    outer_join = []
    outer_join_list = []
    outer_join_source = []
    inner_join = []
    inner_join_list = []
    for k in range(len(sample_df)):
        if len(sample_df.loc[k]['gpt_values']):
            current_outer_join = pd.merge(sample_df.loc[k]['gpt_values'], sample_df.loc[k]['memnet_values'],
                                          how="outer", on=["value"], indicator=True)
            current_inner_join = pd.merge(sample_df.loc[k]['gpt_values'], sample_df.loc[k]['memnet_values'],
                                          on=["value"], indicator=True)
        elif len(sample_df.loc[k]['memnet_values']):
            current_outer_join = sample_df.loc[k]['memnet_values'].copy()
            current_outer_join['_merge'] = ['right_only'] * len(sample_df.loc[k]['memnet_values'])
            current_inner_join = sample_df.loc[k]['memnet_values']
        else:
            current_outer_join = pd.DataFrame.empty
            current_inner_join = pd.DataFrame.empty
        outer_join.append(current_outer_join)
        if len(current_outer_join):
            outer_join_list.append(current_outer_join['value'].tolist())
            outer_join_source.append(current_outer_join['_merge'].tolist())
        else:
            outer_join_list.append([])
            outer_join_source.append([])
        inner_join.append(current_inner_join)
        if len(current_inner_join):
            inner_join_list.append(current_inner_join['value'].tolist())
        else:
            inner_join_list.append([])

    sample_df["inner_join_values"] = inner_join
    sample_df["outer_join_values"] = outer_join
    sample_df["inner_join_list"] = inner_join_list
    sample_df["outer_join_list"] = outer_join_list
    sample_df["outer_join_source"] = outer_join_source


def duplicate_records_outer_join():
    # Step 1: Make a copy of the DataFrame to avoid modifying the original data
    new_df = sample_df.copy()

    # Step 3: Explode the DataFrame based on the 'Column C' containing lists
    new_df = new_df.explode(['outer_join_list', 'outer_join_source'])

    # Step 4: Reset the index to have continuous row numbers
    new_df.reset_index(drop=True, inplace=True)
    return new_df


for i in range(0, 10):
    sample_df = pd.read_csv(
        "labeling_batches/modified_simple_modeling_" + str(i * 100) + "_to_" + str((i + 1) * 100) + "_batch.csv")
    ds_mem_ss = pd.read_csv("extended_memNet_ss.csv")
    structure_GPT_response(sample_df)
    memnet_info_items()
    inner_and_outer_join()
    dup = duplicate_records_outer_join()
    dup.to_csv(index=False,
               path_or_buf="labeling_batches/modified_labeling_batches/labeling_ds_batch" + str(i + 1) + ".csv")
