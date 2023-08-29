import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

# Reformation functions

def find_matching_records_in_full_memNet(df):
    full_correct_memNet = pd.DataFrame(columns=full_memNet.columns)
    for i in range(len(df)):
        value = df.loc[i]['Memory items']
        group_id = df.loc[i]['group_id']
        if value == 'watching tv':
            value = 'watching_tv'
        record = full_memNet.loc[(full_memNet['value'] == value) & (full_memNet['group_id'] == group_id)]
        if len(record) > 0:
            full_correct_memNet = full_correct_memNet._append(record, ignore_index=True)
    full_correct_memNet = full_correct_memNet.drop_duplicates(ignore_index=True)
    full_correct_memNet = full_correct_memNet.sort_values(['group_id','id'])
    full_correct_memNet['validated'] = [1]*len(full_correct_memNet)

    return full_correct_memNet


def find_matching_records_in_full_memNet_gpt(df):
    full_correct_gpt = pd.DataFrame(columns=full_memNet.columns)
    for i in range(len(df)):
        group_id = df.loc[i]['group_id']
        record = full_memNet.loc[full_memNet['group_id'] == group_id].reset_index(drop=True).loc[0]
        if len(record) > 0:
            record['memory_name'] = df.loc[i]['memory_name']
            record['entity'] = df.loc[i]['entity']
            record['value'] = df.loc[i]['value']
            full_correct_gpt = full_correct_gpt._append(record, ignore_index=True)
    full_correct_gpt = full_correct_gpt.drop_duplicates(ignore_index=True)
    full_correct_gpt = full_correct_gpt.sort_values(['group_id', 'id'])
    full_correct_gpt['validated'] = [1] * len(full_correct_gpt)

    return full_correct_gpt


def structure_GPT_response(df):
    tot_gpt_info_items = []
    for j in range(len(df)):
        gpt_text = df.iloc[j]['gpt_text_response']
        # Step 1: Split into individual records
        records = gpt_text.strip().split('\n')

        # Step 2-4: Split into field-value pairs and create dictionaries
        dict_list = []
        for record in records:
            fields = ["memory_name: ", ", entity: ", ", value: ", "(.+)"]
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
    df['gpt_full_memories'] = tot_gpt_info_items


def find_words_between(text, before_word, after_word):
    pattern = r"{}(.*?){}".format(before_word, after_word)
    matches = re.findall(pattern, text)
    if type(matches[0]) == tuple:
        matches[0] = " ".join(matches[0]).strip(".")
    return matches


def structure_memNet_info_items(df):
    tot_memNet_info_items = []
    for j in range(len(df)):
        memNet_text = df.iloc[j]['memnet_info_items']
        lines = memNet_text.strip().split('\n')

        # Extract header and data rows
        header = lines[0]
        data_lines = lines[1:]

        # Extract column names from the header using regular expression
        columns = re.findall(r'\b\w+\b', header)

        # Initialize an empty list to hold the data
        data = []

        # Iterate through data lines and split using regular expression
        for line in data_lines:
            #values = re.findall(r'\b\b\w+\b\b', line)
            values = re.split(r'\s{2,}', line)
            data.append(values)

        # Create a DataFrame from the data and columns
        columns = ['index']+columns
        small_df = pd.DataFrame(data, columns=columns)
        small_df = small_df[small_df.columns[1:4]]
        tot_memNet_info_items.append(small_df)
    df['memNet_full_memories'] = tot_memNet_info_items


def locate_memory_record(df, key):
    # Create an empty DataFrame with the specified columns
    columns = ['memory_name', 'entity', 'value']
    complete_memory_df = pd.DataFrame(columns=columns)
    for i in range(len(df)):
        mem_items_small_df = df[key][i]
        value = df.loc[i]['Memory items']
        if key == 'memNet_full_memories' and value == 'watching tv':
            value = 'watching_tv'
        match_rec = mem_items_small_df.loc[
            mem_items_small_df[mem_items_small_df.columns[2]] == value]
        if len(match_rec) > 1:
            match_rec = match_rec.iloc[0]
        complete_memory_df = complete_memory_df._append(match_rec, ignore_index=True)
    return complete_memory_df


# Importing and manipulating the data

# Step 1 : Import both labeling sets and relevant columns
labeled_ds1 = pd.read_csv("ANNA_memories_labeling_ds - memories_labeling_ds.csv")
labeled_ds1 = \
    labeled_ds1[['elliq_text', 'user_text', 'Memory items', 'group_id', 'outer_join_source', 'Label', 'Missing',
                 'gpt_text_response', 'memnet_info_items']]
labeled_ds2 = pd.read_csv("EITAN_ memories_labeling_ds - memories_labeling_ds.csv")
labeled_ds2 = \
    labeled_ds2[['elliq_text', 'user_text', 'Memory items', 'group_id', 'outer_join_source', 'Label', 'Missing',
                 'gpt_text_response', 'memnet_info_items']]
full_memNet = pd.read_csv("full-memnet-20230817 - bq-results-20230817-123541-1692275757838.csv")

# Step 2 : Unite the two data sets based on the columns of labeling agreement
united_ds = labeled_ds1.iloc[labeled_ds1.index[labeled_ds1['Label'] == labeled_ds2['Label']]]
# Step 3 : Count the missing memories
num_missing = united_ds['Missing'].notna().sum()
# Step 4 : Create different combinations of the data according to the groups
gpt_only_labels = united_ds.loc[(united_ds['outer_join_source'] == 'left_only')]
gpt_labels = united_ds.loc[(united_ds['outer_join_source'] != 'right_only')]
memNet_only_labels = united_ds.loc[(united_ds['outer_join_source'] == 'right_only')]
memNet_labels = united_ds.loc[(united_ds['outer_join_source'] != 'left_only')]
both_labels = united_ds.loc[(united_ds['outer_join_source'] == 'both')]

#  Extracting whole memories (memory_name,entity,value) for each of the labeled as correct memories

gpt_labels = gpt_labels.reset_index()
correct_gpt_items = gpt_labels.loc[gpt_labels['Label'] == 'Correct'].reset_index()
correct_gpt_only_items = gpt_only_labels.loc[gpt_only_labels['Label'] == 'Correct'].reset_index()
correct_memNet_items = memNet_labels.loc[memNet_labels['Label'] == 'Correct'].reset_index()


structure_GPT_response(gpt_labels)
structure_GPT_response(correct_gpt_items)
structure_GPT_response(correct_gpt_only_items)
structure_memNet_info_items(correct_memNet_items)
full_correct_memNet = find_matching_records_in_full_memNet(correct_memNet_items)


complete_memory_memNet_correct_df = locate_memory_record(correct_memNet_items, 'memNet_full_memories')
complete_memory_gpt_only_correct_df = locate_memory_record(correct_gpt_only_items, 'gpt_full_memories')
complete_memory_gpt_correct_df = locate_memory_record(correct_gpt_items, 'gpt_full_memories')
complete_memory_gpt_df = locate_memory_record(gpt_labels, 'gpt_full_memories')

complete_memory_gpt_correct_df['user_text'] = correct_gpt_items['user_text']
complete_memory_gpt_correct_df['elliq_text'] = correct_gpt_items['elliq_text']
complete_memory_gpt_correct_df['group_id'] = correct_gpt_items['group_id']
complete_memory_gpt_df['user_text'] = gpt_labels['user_text']
complete_memory_gpt_df['elliq_text'] = gpt_labels['elliq_text']
complete_memory_gpt_df['group_id'] = gpt_labels['group_id']
complete_memory_memNet_correct_df['user_text'] = correct_memNet_items['user_text']
complete_memory_memNet_correct_df['elliq_text'] = correct_memNet_items['elliq_text']
complete_memory_memNet_correct_df['group_id'] = correct_memNet_items['group_id']
complete_memory_gpt_only_correct_df['user_text'] = correct_gpt_only_items['user_text']
complete_memory_gpt_only_correct_df['elliq_text'] = correct_gpt_only_items['elliq_text']
complete_memory_gpt_only_correct_df['group_id'] = correct_gpt_only_items['group_id']
full_correct_gpt_only = find_matching_records_in_full_memNet_gpt(complete_memory_gpt_only_correct_df)

# Fill in the missing values of GPT table
keys = (full_correct_gpt_only['memory_name'].unique()).tolist()
values = [7776000000, 10368000000, 0, 10368000000, 0, 0, 0, 0, 10368000000, 0, 0, 0, 0, 10368000000, 7776000000, 0, 0, 0, 7776000000, 0, 10368000000, 0, 10368000000, 0, 10368000000, 7776000000, 10368000000]
gpt_decay_mapping = {k: v for k, v in zip(keys, values)}
for i in range(len(full_correct_gpt_only)):
    full_correct_gpt_only.loc[i, 'decay'] = gpt_decay_mapping[full_correct_gpt_only.loc[i]['memory_name']]
    full_correct_gpt_only.loc[i, 'id'] = i
print("heh")
full_correct_gpt_only.loc[(full_correct_gpt_only['memory_name'] == 'event') & (full_correct_gpt_only['intent']=='iamleaving')]['decay'] = 10368000000

# Unit the full GPT and MemNet dataFrame into the complete flawless outstanding whole_new_validated_ds
whole_new_validated_ds = pd.concat([full_correct_gpt_only, full_correct_memNet], ignore_index=True).drop_duplicates()
whole_new_validated_ds = whole_new_validated_ds.sort_values(by=['group_id','id'])
whole_new_validated_ds.to_csv(index=False, path_or_buf="whole_new_validated_ds.csv")

# Visualization (histograms) of memory_name and entity

num_bins = int(np.sqrt(len(complete_memory_gpt_correct_df)))
plt.hist(complete_memory_gpt_correct_df['memory_name'], bins=num_bins, edgecolor='black')
plt.xticks(rotation='vertical')
plt.xlabel('Memory Name')
plt.ylabel('Frequency')
plt.title('Memory Name')
plt.tight_layout()
plt.show()

num_bins = int(np.sqrt(len(complete_memory_gpt_correct_df)))
plt.hist(complete_memory_gpt_correct_df['entity'], bins=num_bins, edgecolor='black')
plt.xticks(rotation='vertical')
plt.xlabel('Entity')
plt.ylabel('Frequency')
plt.title('Entity')
plt.tight_layout()
plt.show()
print("HE")

# Group the data by the two non-numeric parameters and count their frequencies
grouped_correct_gpt = full_correct_gpt_only.groupby(['memory_name', 'entity']).size().reset_index(name='frequency')
grouped_gpt = complete_memory_gpt_df.groupby(['memory_name', 'entity']).size().reset_index(name='frequency')
# Pivot the data to create a pivot table for plotting
pivot_table_correct_gpt = grouped_correct_gpt.pivot(index='memory_name', columns='entity', values='frequency')
pivot_table_gpt = grouped_gpt.pivot(index='memory_name', columns='entity', values='frequency')
pivot_table_correct_gpt = pivot_table_correct_gpt.reindex(pivot_table_correct_gpt.sum().sort_values(ascending=False).index, axis=1)
pivot_table_gpt = pivot_table_gpt.reindex(pivot_table_gpt.sum().sort_values(ascending=False).index, axis=1)
pivot_table_correct_gpt = pivot_table_correct_gpt.loc[pivot_table_correct_gpt.sum(axis=1).sort_values(ascending=False).index]
pivot_table_gpt = pivot_table_gpt.loc[pivot_table_gpt.sum(axis=1).sort_values(ascending=False).index]
pivot_table_gpt = pivot_table_gpt.loc[pivot_table_correct_gpt.index][pivot_table_correct_gpt.columns]
pivot_table_gpt = pivot_table_gpt.rdiv(1)
pivot_table_correct_gpt[pivot_table_correct_gpt < 10] = 0
heatmap = pivot_table_correct_gpt.mul(pivot_table_gpt, fill_value=0)
sns.heatmap(heatmap)
plt.xlabel('Entity')
plt.ylabel('Memory Name')
plt.tight_layout()
plt.title('Correct/all of Memory Name and Entities by GPT')
plt.show()
# Create a bar plot
pivot_table_correct_gpt.plot(kind='bar', stacked=False)

# Set labels and title
plt.xlabel('Entity')
plt.ylabel('Memory Name')
plt.tight_layout()
plt.title('Frequency of Memory Name and Entities by GPT')

# Show the plot
plt.show()

# Keep only the top 5 most frequent memory names
top_5_pivot_table = pivot_table_correct_gpt.head(5)

# Create a bar plot
top_5_pivot_table.plot(kind='bar', stacked=True)

# Set labels and title
plt.xlabel('Memory Name')
plt.ylabel('Frequency')
plt.title('Top 5 Most Frequent Memory Names by GPT')

# Show the plot
plt.tight_layout()
plt.show()
# # Generating results
#
#
# memNet_result = np.array([[len(memNet_only_labels.loc[memNet_only_labels['Label'] == 'Correct']) + len(
#     both_labels.loc[both_labels['Label'] == 'Correct']),
#                            len(memNet_only_labels.loc[memNet_only_labels['Label'] != 'Correct']) + len(
#                                both_labels.loc[both_labels['Label'] != 'Correct'])],
#                           [len(gpt_only_labels.loc[gpt_only_labels['Label'] == 'Correct']) + num_missing,
#                            len(gpt_only_labels.loc[gpt_only_labels['Label'] != 'Correct'])]])
#
# memNet_recall = memNet_result[0, 0] / (memNet_result[0, 0] + memNet_result[0, 1])
# memNet_precision = memNet_result[0, 0] / (memNet_result[0, 0] + memNet_result[1, 1])
#
# gpt_result = np.array([[len(gpt_only_labels.loc[gpt_only_labels['Label'] == 'Correct']) + len(
#     both_labels.loc[both_labels['Label'] == 'Correct']),
#                         len(gpt_only_labels.loc[gpt_only_labels['Label'] != 'Correct']) + len(
#                             both_labels.loc[both_labels['Label'] != 'Correct'])],
#                        [len(memNet_only_labels.loc[memNet_only_labels['Label'] == 'Correct']) + num_missing,
#                         len(memNet_only_labels.loc[memNet_only_labels['Label'] != 'Correct'])]])
#
# gpt_recall = gpt_result[0, 0] / (gpt_result[0, 0] + gpt_result[0, 1])
# gpt_precision = gpt_result[0, 0] / (gpt_result[0, 0] + gpt_result[1, 1])
#
# # gpt_cm_disp = ConfusionMatrixDisplay(confusion_matrix=gpt_result,title="GPT Confusion matrix")
# # gpt_cm_disp.plot()
# # plt.show()
# # memNet_cm_disp = ConfusionMatrixDisplay(confusion_matrix=memNet_result,title="MemNet Confusion matrix")
# # memNet_cm_disp.plot()
# # plt.show()
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
# fig.subplots_adjust(wspace=0.5)
#
# colorbar_min = min(gpt_result.min(), memNet_result.min())
# colorbar_max = max(gpt_result.max(), memNet_result.max())
#
# group_names = ["TP", "FP", "FN", "TN"]
# group_counts_gpt = ["{0:0.0f}".format(value) for value in gpt_result.flatten()]
# group_percentages_gpt = ["{0:.2%}".format(value) for value in gpt_result.flatten() / np.sum(gpt_result)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts_gpt, group_percentages_gpt)]
# labels = np.asarray(labels).reshape(2, 2)
# sns.heatmap(gpt_result, annot=labels, ax=ax1, fmt='', cmap='Blues', vmin=colorbar_min, vmax=colorbar_max)
# ax1.set_title("GPT results\n Recall: " + str("{:.3f}".format(memNet_recall)) + ", Precision: " + str(
#     "{:.3f}".format(memNet_precision)))
#
# group_counts_memNet = ["{0:0.0f}".format(value) for value in memNet_result.flatten()]
# group_percentages_memNet = ["{0:.2%}".format(value) for value in memNet_result.flatten() / np.sum(memNet_result)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts_memNet, group_percentages_memNet)]
# labels = np.asarray(labels).reshape(2, 2)
# sns.heatmap(memNet_result, annot=labels, ax=ax2, fmt='', cmap='Blues', vmin=colorbar_min, vmax=colorbar_max)
# ax2.set_title("memNet results\n Recall: " + str("{:.3f}".format(gpt_recall)) + ", Precision: " + str(
#     "{:.3f}".format(gpt_precision)))
# plt.show()
