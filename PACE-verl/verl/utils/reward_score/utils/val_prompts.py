OTHER_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
    -
    Special considerations:

    1. **Multiple Answers**: If the output contains multiple answers but correct answer only have one, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

    2. **Mathematical Problems**: a): If the formats differ but the answers are mathematically equivalent, respond with [Correct]. b): You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain.

    3. **Explicit Options**: If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or  the correct option's code.

    4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
    -

    Question: """{question}"""

    Output sentence: """{pred}"""

    Correct answer: {reference}

    Judgement:
    '''    
   #  4. **Error tolerance**: For all numeric answers, an error margin of up to 1\% is allowed for all numerical values(e.g. 9.93e5 vs 9.94e5). Theses cases will be considered as [Correct]


STEM_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
    -
    Special considerations:

    1. **Mathematical Problems**: a): If the formats differ but the answers are mathematically equivalent after simplfying or rounding(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. b): You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct]

    2. **Explicit Options**: If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].

    4. **Multiple Answers**: If the output contains multiple answers, a): If the correct answer also contain multiple answers, evaluate each answer with correct answer.  b): If the correct answer only contain one answer, evaluate the final given answer.  If the final answer is unclear or incorrect, respond with [Incorrect].

    -

    Question: """{question}"""

    Output sentence: """{pred}"""

    Correct answer: {reference}

    Judgement:
    '''


COMPASS_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. For mathematical answers, please first simplify and reduce the polynomial or fraction, and then proceed with the comparison.
2. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
3. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
5. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
6. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
7. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>

<Candidate's Answer Begin>: 
{pred}
<Candidate's Answer End>

<Standard Answer Begin>:
{reference}
<Standard Answer End>

Judging the correctness of the candidate's answer:
"""


COMPASS_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly. 
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT: 
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>:
{question}
<Original Question End>

<Candidate's Answer Begin>: 
{pred}
<Candidate's Answer End>

<Standard Answer Begin>:
{reference}
<Standard Answer End>

Analysis step by step and Final Judgment:
"""


XVERIFY_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
    -
    Special considerations:

    1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

    2. **Mathematical Problems**: If the formats diffezr but the answers are mathematically equivalent, respond with [Correct].

    3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

    4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
    -

    Question: """{question}"""

    Output sentence: """{pred}"""

    Correct answer: {reference}

    Judgement:
    '''


STEM_PROMPT_TOOL = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. You can perform a short tool call and a short reasoning process. Put your response in the final with either [Correct] or [Incorrect] wrapped in \\boxed{{}} .
    -
    Special considerations:

    1. **Mathematical Problems**: a): If the formats differ but the answers are mathematically equivalent after simplfying or rounding(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. b): You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct]

    2. **Explicit Options**: If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].

    4. **Multiple Answers**: If the output contains multiple answers, a): If the correct answer also contain multiple answers, evaluate each answer with correct answer.  b): If the correct answer only contain one answer, evaluate the final given answer.  If the final answer is unclear or incorrect, respond with [Incorrect].
    -
   
   -
   You can use following tools to help your verification process:

   1. **Python Intepreter**: When you feel needed, you can use a python inteperter to help you calculate and determine your verification result.

   2. **Unit Conversion Tool**: When faced with different physical units, you can use a unit conversion tool to convert them into the same unit.
   -

   Question: """{question}"""

   Output sentence: """{pred}"""

   Correct answer: {reference}

   Judgement:
    '''


STEM_PROMPT_TOOL1 = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. You can perform a short tool call. Put your response in the final with either [Correct] or [Incorrect] wrapped in \\boxed{{}} .
    -
    Special considerations:

    1. **Mathematical Problems**: a): If the formats differ but the answers are mathematically equivalent after simplfying or rounding(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. b): You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct]

    2. **Explicit Options**: If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].

    4. **Multiple Answers**: If the output contains multiple answers, a): If the correct answer also contain multiple answers, evaluate each answer with correct answer.  b): If the correct answer only contain one answer, evaluate the final given answer.  If the final answer is unclear or incorrect, respond with [Incorrect].
    -
   
   -
   You can use following tools to help your verification process:

   1. **Python Intepreter**: When needed, you can use a python inteperter to help you calculate and determine your verification result.

   2. **Unit Conversion Tool**: When faced with different physical units, you can use a unit conversion tool to convert them into the same unit.
   -

   Question: """{question}"""

   Output sentence: """{pred}"""

   Correct answer: {reference}

   Judgement:
    '''

STEM_PROMPT_TOOL2 = '''
**Objective:** Act as an impartial judge to determine if a given output correctly answers a question, based on a provided reference answer.

**Instructions:**
Your task is to carefully analyze the `pred` in the context of the `question` and compare it to the `reference`. After your analysis, you must provide a final verdict. Your response should conclude with either `\boxed{[Correct]}` or `\boxed{[Incorrect]}`.

You are equipped with the following tools to aid your decision:
1.  **Python Interpreter**: For verifying mathematical calculations.
2.  **Unit Conversion Tool**: For comparing values with different physical units.

**Evaluation Criteria:**

1.  **For Mathematical Questions**:
    a) An output is deemed [Correct] if it is numerically equivalent to the reference, regardless of notation (e.g., $0.5$ is the same as $\frac{1}{2}$).
    b) Disregard differences in units or other non-numerical context (e.g., if the answer is $16$, an output of `16 km` is [Correct]).

2.  **For Questions with Choices**: If the question presents a set of options (e.g., multiple choice), the output is [Correct] as long as it clearly identifies the correct option, either by its content or its label (e.g., "A", "2.", etc.).

3.  **For Open-Ended Questions**: The output must be semantically equivalent to the reference answer. Minor differences in wording are acceptable as long as the core meaning is the same.

4.  **Regarding Multiple Answers in the Output**:
    a) If the reference also contains multiple valid answers, evaluate according to the question.
    b) If the reference contains only a single answer, you must base your judgment on the *final* answer given in the output. If that final answer is wrong or not clearly stated, the entire output is [Incorrect].

---
Question: """{question}"""

Output sentence: """{pred}"""

Correct answer: {reference}

Judgement:'''

STEM_PROMPT_TOOL3 = '''
**Your Role:** You're the judge. Your goal is to decide if a new answer (`Predicted Output`) is a good match for the correct `Reference Answer`, given the original `Question`.

**The Main Rule:** After you've made your call, state your final verdict clearly. It must be either `\boxed{[Correct]}` or `\boxed{[Incorrect]}`.

To help you out, you have a couple of tools on hand:
* A **Python interpreter** for executing python code when you feel needed.
* A **Unit Converter** for any tricky measurements.

**Guidelines for Making a Fair Call:**

* **For Math Problems:** If the numbers match, it's correct. Don't sweat the small stuff like notation ($0.5$ is the same as $\frac{1}{2}$) or units (an answer of `16` is correct even if the output says `16 km`).

* **For Multiple-Choice:** If the output clearly picks the right choice (by its letter, number, or by stating the answer itself), give it a pass.

* **For Written Answers:** It's all about the meaning. If the output says the same thing as the reference answer, even with different words, it's correct.

* **The Tricky Case - Multiple Answers:** Pay attention when the output gives more than one answer. If the reference has just *one* correct answer, only judge the **very last answer** in the output. If that final part is wrong or unclear, the whole thing is incorrect.

---
**Question to Judge:** """{question}"""

**Submitted Output:** """{pred}"""

**Correct Answer Key:** {reference}

**Your Verdict:**
'''

STEM_PROMPT_1 = '''
**Role:** You are to act as a verification agent. Your sole function is to determine if a predicted answer is correct.

**Task:**
You will be given a `question`, a `pred` (predicted answer), and a `reference` (the ground truth answer). Compare the `pred` to the `reference` and determine its validity. Your output must be one of two possible responses: `[Correct]` or `[Incorrect]`.

**Verification Rules:**

1.  **Regarding Mathematical Content:**
    * **Equivalence:** The prediction is considered correct if it is mathematically equivalent to the reference, regardless of the format (e.g., $\frac{5}{2}$ is equivalent to $2.5$).
    * **Units & Context:** Ignore peripheral information like units or domains. If the numerical value is correct, the answer is correct (e.g., `100` is a correct match for `100 lbs`).

2.  **Regarding Multiple Choice Questions:**
    * The prediction is correct if it unambiguously selects the correct option, either by quoting its text or referring to its identifier (e.g., 'C', 'iii', etc.).

3.  **Regarding Open-Ended Questions:**
    * The prediction is correct only if its meaning is substantially the same as the reference answer.

4.  **Regarding Predictions with Multiple Answers:**
    * If the reference answer is also multi-part, compare the corresponding parts.
    * If the reference answer is singular, you must evaluate *only the final answer* provided in the prediction. If this final part is incorrect or unclear, the entire prediction is deemed incorrect.

---
**Question:** """{question}"""

**Predicted Answer:** """{pred}"""

**Reference Answer:** {reference}

**Judgment:**'''


STEM_PROMPT_2 = '''
---

### **Role: Expert Examiner**

### **Objective**
Your core responsibility is to precisely determine if a "Predicted Answer" is correct based on the provided "Reference Answer" (the ground truth).

### **Adjudication Protocol**
You must strictly adhere to the following guidelines when making your judgment:

1.  **Regarding Mathematical Content:**
    * **Equivalence:** The prediction is considered correct if it is mathematically equivalent to the reference, regardless of format (e.g., $2.5$ is equivalent to $\frac{5}{2}$).
    * **Ignore Peripherals:** Non-essential information like units (e.g., `lbs`, `kg`) or contextual phrasing should be disregarded. If the core numerical value is correct, the answer is correct.

2.  **Regarding Multiple-Choice Questions:**
    * The prediction is correct if it unambiguously identifies the correct option, whether by its letter ('C'), number ('iii'), or by quoting its text.

3.  **Regarding Open-Ended Questions:**
    * The core meaning of the prediction must be substantially the same as the reference answer.

4.  **Regarding Predictions with Multiple Answers:**
    * If the reference answer is singular, you must **evaluate only the final answer or conclusion** provided in the prediction. If this final part is incorrect or unclear, the entire prediction is deemed incorrect.
---

### **Evaluation Data**

**Question:** """{question}"""

**Predicted Answer:** """{pred}"""

**Reference Answer:** {reference}

---

### **Required Output Format**

Your response must consist of the following two parts, in this exact order:

1.  **Reasoning:**
    (Briefly explain your judgment logic here, outlining how you applied the protocol to compare the prediction and the reference.)

2.  **Final Verdict:**
    `[Correct]` or `[Incorrect]`'''


# COMPASS_PROMPT = """
# Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
# Here are some evaluation criteria:
# 1. For mathematical answers, please first simplify and reduce the polynomial or fraction, and then proceed with the comparison.
# 2. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
# 3. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
# 4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
# 5. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
# 6. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
# 7. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
# A: CORRECT 
# B: INCORRECT
# C: INVALID
# Just return the letters "A", "B", or "C", with no text around it.
# Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
# <Original Question Begin>:
# Follow the given examples and answer the question.\nAnswer extremely simple questions about the colors of objects on a surface.\n\nQ: On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?\nOptions:\n(A) red\n(B) orange\n(C) yellow\n(D) green\n(E) blue\n(F) brown\n(G) magenta\n(H) fuchsia\n(I) mauve\n(J) teal\n(K) turquoise\n(L) burgundy\n(M) silver\n(N) gold\n(O) black\n(P) grey\n(Q) purple\n(R) pink\nA: Let's think step by step.\nAccording to this question, the color of the stress ball is blue. So the answer is (E).\n\nQ: On the table, you see a bunch of objects arranged in a row: a purple paperclip, a pink stress ball, a brown keychain, a green scrunchiephone charger, a mauve fidget spinner, and a burgundy pen. What is the color of the object directly to the right of the stress ball?\nOptions:\n(A) red\n(B) orange\n(C) yellow\n(D) green\n(E) blue\n(F) brown\n(G) magenta\n(H) fuchsia\n(I) mauve\n(J) teal\n(K) turquoise\n(L) burgundy\n(M) silver\n(N) gold\n(O) black\n(P) grey\n(Q) purple\n(R) pink\nA: Let's think step by step.\nAccording to this question, the objects are arranged in a row, from left to right, as follows: (1) a purple paperclip, (2) a pink stress ball, (3) a brown keychain, (4) a green scrunchiephone charger, (5) a mauve fidget spinner, (6) a burgundy pen.\nThe stress ball is the second object on the list, namely (2). The object that is to the right of the stress ball corresponds to (3), which is a brown keychain.\nThe color of the keychain is brown. So the answer is (F).\n\nQ: On the nightstand, you see the following items arranged in a row: a teal plate, a burgundy keychain, a yellow scrunchiephone charger, an orange mug, a pink notebook, and a grey cup. How many non-orange items do you see to the left of the teal item?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three\n(E) four\n(F) five\n(G) six\nA: Let's think step by step.\nAccording to this question, the objects are arranged in a row, from left to right, as follows: (1) a teal plate, (2) a burgundy keychain, (3) a yellow scrunchiephone charger, (4) an orange mug, (5) a pink notebook, (6) a grey cup.\nThe teal plate is the first item, namely (1). There is no item to the left of the teal item.\nThe number of non-orange items to the left of the teal item is zero. So the answer is (A).\n\nQ: On the floor, you see one magenta paperclip, one teal paperclip, one grey plate, three magenta plates, one magenta textbook, and three teal textbooks. If I remove all the grey objects from the floor, how many paperclips remain on it?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three\n(E) four\n(F) five\n(G) six\n(H) seven\n(I) eight\n(J) nine\n(K) ten\n(L) eleven\n(M) twelve\n(N) thirteen\n(O) fourteen\n(P) fifteen\n(Q) sixteen\nA: Let's think step by step.
# <Original Question End>:

# <Candidate's Answer Begin>: 
# According to this question, the objects on the floor are as follows: (1) one magenta paperclip, (2) one teal paperclip, (3) one grey plate, (4) three magenta plates, (5) one magenta textbook, and (6) three teal textbooks.\n\nIf I remove all the grey objects from the floor, I will remove the grey plate. \n\nNow, let's count the remaining paperclips:\n- There is one magenta paperclip.\n- There is one teal paperclip.\n\nSo, after removing the grey plate, the total number of paperclips remaining on the floor is two.\n\nSo the answer is (C)
# <Candidate's Answer End>

# <Standard Answer Begin>:
# (C)
# <Standard Answer End>

# Judging the correctness of the candidate's answer:
# """



STEM_PROMPT_TOOL_NEW = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. You can perform a short tool call and a short reasoning process. After a short reasoing process, put your response in the final with either [Correct] or [Incorrect] wrapped in \\boxed{{}} 

    Evaluation Protocol:
    1. Reference Standard:
    - The standard (gold) answer is definitive and always correct.
    - The question is always valid — never challenge it.
    - Allow equivalenct meaning answers.
    - Do not regenerate answers; only compare candidate's final answer with the gold answer.
    - You only need to compare correct answer and output sentence, do not regenerate or judge correct answer.

    2. Comparison Method:
    - Analyze the question's requirements and the gold answer's structure.
    - Determine if the question requires exact matching or allows equivalence.
    - Compare ONLY the candidate's final answer. Ignore reasoning errors.
    - Ignore differences in formatting or style.
    - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
    - For multiple-choice: only compare the final choice and its content.

    3. Multi-part Answers:
    - All parts must match the gold answer exactly.
    - Partial matches are incorrect.
    - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

    -
    Special considerations:

    1. **Mathematical Problems**:
        - If the formats differ but the answers are mathematically equivalent after simplfying or rounding to two decimal places(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. 
        - You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct].
        - You may need to calculate the value or converse the value to different unit when needed to match the reference answer.

    2. **Multi-choice questions**: 
        - If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **Fact quuestions**: 
        - If the question provides fact-seeking answers, the output must align with the correct answer in content to be considered [Correct].

    4. **Multiple Reference Answers**:
         -If multiple reference answers are equivalent, just matching one answer will be considered [Correct]. 
         -If multiple reference answers are inequivalent, only mathcing all answers will be considered [Correct].

    5. **Ohter conditions**:
        - If incomplete (cut off, unfinished sentence) → Label as [Incorrect].
        - If repetitive (looping words/phrases) → Label as [Incorrect].
        - Gives an answer but then negates it at the end. → Label as [Incorrect].
        - Numerically correct but without units. → Label as [Correct].
    -
   
   -
   You can use following tools to help your verification process:

   1. **Python Intepreter**: When you feel needed, you can use a python inteperter to help you determine your verification result.

   2. **Unit Conversion Tool**: When faced with different physical units, you can use a unit conversion tool to convert them into the same unit.
   -

   Question: """{question}"""

   Output sentence: """{pred}"""

   Correct answer: {reference}

   Judgement:
    '''



STEM_PROMPT_NEW = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. You can perform a short tool call and a short reasoning process for your verification. Put your response in the final with either [Correct] or [Incorrect] wrapped in \\boxed{{}} after a short reasoing process.

    Evaluation Protocol:
    1. Reference Standard:
    - The standard (gold) answer is definitive and always correct.
    - The question is always valid — never challenge it.
    - Do not regenerate answers; only compare candidate's final answer with the gold answer.

    2. Comparison Method:
    - Analyze the question's requirements and the gold answer's structure.
    - Allow equivalenct meaning answers.
    - Compare ONLY the candidate's final answer. Ignore reasoning errors.
    - Ignore differences in formatting or style.
    - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
    - For multiple-choice: only compare the final choice and its content.

    3. Multi-part Answers:
    - All parts must match the gold answer exactly.
    - Partial matches are incorrect.
    - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

    -
    Special considerations:

    1. **Mathematical Problems**:
        - If the formats differ but the answers are mathematically equivalent after simplfying or rounding to two decimal places(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. 
        - You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct].
        - You may need to calculate the value or converse the value to different unit when needed to match the reference answer.

    2. **Multi-choice questions**: 
        - If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **Fact quuestions**: 
        - If the question provides fact-seeking answers, the output must align with the correct answer in content to be considered [Correct].

    4. **Multiple Reference Answers**:
         -If multiple reference answers are equivalent, just matching one answer will be considered [Correct]. 
         -If multiple reference answers are inequivalent, only mathcing all answers will be considered [Correct].

    5. **Ohter conditions**:
        - If incomplete (cut off, unfinished sentence) → Label as [Incorrect].
        - If repetitive (looping words/phrases) → Label as [Incorrect].
        - Gives an answer but then negates it at the end. → Label as [Incorrect].
        - Numerically correct but without unit → Label as [Correct].
    

   Question: """{question}"""

   Output sentence: """{pred}"""

   Correct answer: {reference}

   Judgement:
    '''




STEM_PROMPT_O3 = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence corectly answers the question based on the provided correct answer. You can perform a short tool call and a short reasoning process for your verification. Put your response in the final with either [Correct] , [Incorrect] or [Invalid] wrapped in \\boxed{{}} after a short reasoing process.

    Evaluation Protocol:
    1. Reference Standard:
    - The standard (gold) answer is definitive and always correct.
    - The question is always valid — never challenge it.
    - Do not regenerate answers; only compare candidate's final answer with the gold answer.

    2. Comparison Method:
    - Analyze the question's requirements and the gold answer's structure.
    - Allow equivalenct meaning answers.
    - Compare ONLY the candidate's final answer. Ignore reasoning errors.
    - Ignore differences in formatting or style.
    - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
    - For multiple-choice: only compare the final choice and its content.

    3. Multi-part Answers:
    - All parts must match the gold answer exactly.
    - Partial matches are incorrect.
    - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

    -
    Special considerations:

    1. **Mathematical Problems**:
        - If the formats differ but the answers are mathematically equivalent after simplfying or rounding to two decimal places(e.g. 2.909 vs \\frac{{32}}{{11}}, \\frac{{32}}{{11}} vs \\frac{{96}}{{33}}), respond with [Correct]. 
        - You only need to verify the correctness of the mathematical expression, not values unrelated to the overall expression, such as the domain or units(e.g. 16 vs 16km, 20 vs 20db), these cases will be considered as [Correct].
        - You may need to calculate the value or converse the value to different unit when needed to match the reference answer.

    2. **Multi-choice questions**: 
        - If the question provides explicit candidate answers(e.g. multi-choice questions), the output will be considered correct if it clearly indicates the correct option's content or the correct option's code.

    3. **Fact quuestions**: 
        - If the question provides fact-seeking answers, the output must align with the correct answer in content to be considered [Correct].

    4. **Multiple Reference Answers**:
         -If multiple reference answers are equivalent, just matching one answer will be considered [Correct]. 
         -If multiple reference answers are inequivalent, only mathcing all answers will be considered [Correct].
    
    5. **Invalid situations**:
        - If the question is incomplete (e.g., a multiple-choice question is missing its options) → Label as [Invalid].
        - If it doesn't match the answer's format (e.g., a multiple-choice question for a fill-in-the-blank answer) → Label as [Invalid].
        - If the answer is ambiguous(e.g. without giving specific answers) → Label as [Invalid].

    5. **Ohter conditions**:
        - If incomplete (cut off, unfinished sentence) → Label as [Incorrect].
        - If repetitive (looping words/phrases) → Label as [Incorrect].
        - Gives an answer but then negates it at the end. → Label as [Incorrect].
        - Numerically correct but without unit → Label as [Correct].
    

    

   Question: """{question}"""

   Output sentence: """{pred}"""

   Correct answer: {reference}

   Judgement:
    '''