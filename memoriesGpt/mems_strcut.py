#
# SELECT agent_id, intent, memory_name, entity, value, user_text, confidence, elliq_text, plan_name, client_time
# # SELECT *
# FROM memories
# # # WHERE (origin = 'ELLIQ'
# # #   and memory_name = 'daily_routines_tv'
# # #   and entity = 'tv_show_name')
# WHERE origin = 'ELLIQ'
# # #       and value = 'watching_tv')
# # and memory_name = 'favorite'
# # WHERE agent_id = 'eqc-florida-249'
# #   and (memory_name = 'place'
# #   and entity != 'intent')
# #   or memory_name = 'relationship'
# # # # AND agent_id =
# # and confidence > 0.5
# and memory_name NOT IN ('health_measurement_status', 'health_measurement', 'last_heart_rate_configuration', 'last_blood_oxygen_configuration', 'last_weight_configuration', 'depression_assessment', 'pain_assessment',
#                         'last_blood_glucose_configuration', 'general_wellbeing', 'energy_check_memory', 'anxiety_assessment', 'prescription_pickup', 'medication_started', 'side_effects', 'user_shared_info_medical',
#                         'medical_procedure_journey', 'user_concentration', 'end_of_day_experience', 'routine_medication_intake', 'medication_tracking'
#                         'skipped_modules', 'disabled_physical_exercise_watch_videos', 'weather', 'national_day', 'anniversary', 'jokes', 'eod_activity_choice', 'morning_quote', 'flags', 'sleep', 'sleep_pattern',
#                         'user_mood', 'user_feeling', 'CSAT', 'healthy_days', 'loneliness', 'survey', 'cobot_survey', 'emovie', 'art_trivia', 'story_short_stories', 'respond_for_user_leave', 'volume_accomplish',
#                         'have_a_drink', 'discovery_quiet_mode', 'onboarding_sections', 'onboarding', 'cafe', 'discovery_timer', 'disabled_features', 'holiday_greeting_postcard', 'favorite_food_cap_reached',
#                        'discovery_bible_verses', 'elvis_cap_reached', 'einstein_cap_reached', 'how_to_tap', 'how_to_dial', 'user_energy', 'word_scramble', 'user_search_info', 'dial_accomplish', 'loud_error',
#                        'care_messaging', 'last_blood_pressure_configuration', 'stress_reduction_trigger', 'blood_pressure_trigger', 'asking_user_went_outside', 'disabled_word_scramble', 'disabled_music_stations',
#                        'wellness_share_info_contact', 'emergency_contact', 'privacy_plan', 'bible_verses', 'date_time', 'set_timer', 'time_capsule_story', 'reminders', 'words_mem_game', 'what_are_your_plans',
#                        'would_you_rather', 'tictactoe')
# and entity NOT IN ('rescheduled_by', 'first_time_intro', 'user_is_leaveing', 'user_is_back', 'first_time', 'question_asked', 'destination_known', 'user_is_leaving', 'likes_exercise', 'intensity', 'last_intensity',
#                   'last_intensity_used', 'appetite', 'done', 'stop', 'diet', 'appetite_flag', 'accomplished', 'video_offered', 'last_asked_question', 'video_offered', 'holiday_music_discovery_attempts',
#                    'asked_thankful_for_question', 'asking_user_went_outside', 'offer', 'christmas_song_game', 'init', 'accomplished_count', 'lastYearOfCelebration', 'feedback', 'rejection_count', 'accepted_count',
#                   'hobby_not_done_reason', 'attempts_counter_confirm_pet', 'departure', 'arrival', 'coming_back', 'last_time_asked_leaving_destination')
# and value NOT IN ('true', 'false', 'yes', 'no', '0.0', '1.0', '2.0', 'say yes', 'say no')
# ORDER BY  client_time
# LIMIT 20000






import pandas as pd
import openai
import re
import time
api_key = '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'
openai.api_key = api_key

ds_mem = pd.read_csv("client_based_mems_1.csv")
creators = ds_mem['agent_id'].unique()


memory_name_map = {
    "user_info": "my",
    "daily_routines_tv": "my favorite",
    "would_you_rather": "I would rather",
    "daily_routine": "",
    "user_diet": "",
    "flowers": "favorite flower",
    "holiday_preferences": "favorite holiday",
    "musical_artist_memory": "favorite",
    "activity_back": "",
    "leaving_destination_back": "",
    "activity": "",
    "activity_back": "",
    "user_diet": "I am eating",
    "arrivals_departures": "",
    "movies_convo": "",
    "meal_type": "",
    "user_diet": "",
    "first_job_convo": ""
}

ents_map = {
    "memory_musical_artist": "",
    "winter_holidays": "",
    "user_hobby": "hobby",
    "color": "",
    "flower": "",
    "alternate_meal": "meal",
    "meal_type": "meal",
    "food": "",
    "food_chain": "at",
    "food_cuisine": "",
    "last_activity_departure": "last activity",
    "last_destination_departure": "last destination",
    "last_activity_arrival": "last activity",
    "last_destination_arrival": "last destination"
}

user_said = []
gpt_response = []
instructions = []
users = []

for creator in creators :
    df_temp = ds_mem[ds_mem['agent_id'] == creator]
    all_mems = ""
    user_mems = []
    for k in range(len(df_temp)) :
        mem_name = df_temp.iloc[k]['memory_name']
        mem_ent = df_temp.iloc[k]['entity']
        mem_val = df_temp.iloc[k]['value']
        mem_time = df_temp.iloc[k]['client_time']
        if pd.isnull(mem_val) :
            continue
        if mem_name == 'words_mem_game' or mem_name == 'would_you_rather' or mem_name == 'what_are_your_plans' or mem_name == 'routine_medication_intake' or mem_name == 'medication_tracking':
            continue
        if mem_ent == "attempts_counter_confirm_pet" :
            continue
        if mem_name in memory_name_map.keys() :
            mem_name = memory_name_map.get(mem_name)
        if mem_ent in ents_map.keys() :
            mem_ent = ents_map.get(mem_ent)
        if 'guests' in mem_name :
            mem_name = re.sub('guests', 'had a guest named', mem_name)
            if mem_ent == 'relation' :
                mem_ent = 'who is'
        if mem_ent == 'birthday':
            mem_val = mem_val.split('T')[0]
        if mem_name == mem_ent :
            mem_ent = ''
        mem = mem_name + " " + mem_ent + " " + mem_val + ". "
        mem = re.sub('_', ' ', mem)
        mem = re.sub('  ', ' ', mem)
        if mem in user_mems :
            continue
        all_mems = all_mems + mem
    user_mems.append(mem)

message = [{"role": "user", "content": "Act as an emphatic companion robot named Elli-Q. Do not refer yourself as AI language model."},
           {"role": "user", "content": "Here is all the information that I have previously told you about myself: " + all_mems},
           {"role": "user", "content": "Summarize in bullets the information that I have previously told you about myself."}]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=250,
    temperature=0.8,
    messages=message)

print("user memories: ", all_mems)
print("gpt: ", response['choices'][0]['message']['content'])

structured_mems = response['choices'][0]['message']['content']

message = [{"role": "user", "content": "Act as an emphatic companion robot named Elli-Q. Do not refer yourself as AI language model."},
           {"role": "user",
            "content": "Here is all the information that I have previously told you about myself: " + structured_mems},
            {"role": "user", "content":  "Summarize your knowledge about myself"}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=250,
    temperature=0.8,
    messages=message)

print("gpt: ", response['choices'][0]['message']['content'])

target_texts = [
    entities_df.iloc[j]['user_text'] + '. Ask a question that combines this with some of what I previously told you',
    entities_df.iloc[j]['user_text'] + '. Acknowledge this while stating some of what I previously told you']

fly = "I am watching TV. "
message = [{"role": "user", "content": "Act as an empathic companion robot named Elli-Q. Do not refer yourself as AI language model. Limit your response to maximum two sentences."},
           {"role": "user",
            "content": "Here is all the information that I have previously told you about myself: " + structured_mems},
            {"role": "user", "content":  fly + "Ask a question that combines this with some of the information I have previously told you"}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=250,
    temperature=1.2,
    messages=message)

print("gpt: ", response['choices'][0]['message']['content'])

fly = "i am going to the doctor"
message = [{"role": "user", "content": "Act as an emphatic companion robot named Elli-Q. Do not refer yourself as AI language model. Limit your response to maximum two sentences."},
           {"role": "user",
            "content": "Here is all the information that I have previously told you about myself: " + structured_mems},
            {"role": "user", "content":  fly + "Acknowledge this while stating one of the information items that I have previously told you"}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=250,
    temperature=1.2,
    messages=message)

print("gpt: ", response['choices'][0]['message']['content'])

activity = "eat healthy"
message = [{"role": "user", "content": "Act as an emphatic companion robot named Elli-Q. Do not refer yourself as AI language model. Limit your response to maximum two sentences."},
           {"role": "user",
            "content": "Here is all the information that I have previously told you about myself: " + structured_mems},
            {"role": "user", "content":  "Convince me to " + activity + "  using some of the information I have previously told you"}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=250,
    temperature=1.2,
    messages=message)

print("gpt: ", response['choices'][0]['message']['content'])


df = pd.DataFrame({
        'memory_name': list(df_temp['memory_name']),
        'entity':  list(df_temp['entity']),
        'value':  list(df_temp['value'])
    })

for i in range(len(df)):
    print(list(df.iloc[i]))