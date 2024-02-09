

Model: Mistral-7B-Instruct-v02 Attempt 1: (No CoT, 0-shot)

system_prompt = """You are the police department's virtual assistant, you are going to read the following narratives
and return whether they are related to behavioral health, All samples only have one answer. For your response you must always use <Tag> [Answer] </Tag>.
you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other
, The text you must classify is as follows: """

hyperparameters = sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens = 250,
        temperature = 0.7,
        top_k=50,
        top_p = 0.95,
        num_return_sequences = 1
)

metrics = Accuray: 0.576271186440678
Precision: 0.4559228940002946
Recall 0.0
f1_score: 0.4559228940002946

(After changing temp to 0.5)
Accuray: 0.6101694915254238
Precision: 0.28954723954723954
Recall 0.0
f1_score: 0.28954723954723954

(After changing temp to 0.3)
Accuray: 0.6305084745762712
Precision: 0.321265172735761
Recall 0.0
f1_score: 0.321265172735761

(After changing temp to 0.1)
Accuray: 0.6305084745762712
Precision: 0.3376664217650726
Recall 0.0
f1_score: 0.3376664217650726

Model: Mistral-7B-Instruct-v02 Attempt 2: (No CoT, 0-shot)
system_prompt = """[INST]You are the police department's virtual assistant, you are going to read the following narratives
        and return whether they are related to behavioral health, All samples only have one answer. For your response you must always use <Tag> [Answer] </Tag>.
        you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other
        , The text you must classify is as follows: [/INST]"""

sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens = 1200,
            temperature = 0.5,
            top_k=40,
            top_p = 0.50,
            num_return_sequences = 1
        )    


Accuray: 0.6847457627118644
Precision: 0.3903637532406817
Recall 0.0
f1_score: 0.3903637532406817


system_prompt = """[INST]You are the police department's virtual assistant, you are going to read the following narratives
        and return whether they are related to behavioral health, All samples only have one answer. For your response you must always use <Tag> [Answer] </Tag>.
        you will tag these as either Domestic Social, NonDomestic Social, Mental Health, Substance Abuse or Other
        , The text you must classify is as follows: [/INST]"""


post_text_prompt = " [INST]  Classify the text, you must ONLY use <Tag> [Answer] </Tag> and can choose ONLY one answer [/INST] "

sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens = 1200,
            temperature = 0.5,
            top_k=40,
            top_p = 0.50,
            num_return_sequences = 1
        )
Accuray: 0.7186440677966102
Precision: 0.4506053192123216
Recall 0.0
f1_score: 0.4506053192123216