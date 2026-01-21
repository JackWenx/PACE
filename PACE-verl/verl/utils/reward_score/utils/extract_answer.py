import math
import re

def remove_boxed(s):
    # if s is None:
    #     return 'Invalid'
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def extract_answer(solution_str: str):
    answer_patterns = [
        r"(?i)Answer\s*:\s*([^\n]+)",
        r"(?i)(?:\*\*)?Final Answer(?:\*\*)?\s*:\s*([^\n]+)",
        r"the answer is\s*(.*?)\."
    ]
    if '</think>' in solution_str:
        idx = solution_str.find('</think>')
        solution_str = solution_str[idx + len('</think>'):]
    
    if '\\boxed' in solution_str:
        result = remove_boxed(last_boxed_only_string(solution_str))
        if result not in ['Invalid']:
            return remove_boxed(last_boxed_only_string(solution_str))

    for answer_pattern in answer_patterns:
        match = re.findall(answer_pattern, solution_str)
        if len(match) >= 1:
            pattern = r"\*\*(.*?)\*\*"
            match1 = re.findall(pattern, match[-1])
            if len(match1) > 0:
                return match1[-1]
            else:
                return match[-1]
    return solution_str


def extract_thinking(solution_str):
    res = solution_str
    if '</think>' in solution_str:
        idx = solution_str.find('</think>')
        res = solution_str[idx + len('</think>'):].strip()
    return res

def find_last_correction(text):
    """
    找出最后一个出现的 '[Correct]' 或 '[Incorrect]'。
    """
    # text = text.lower()
    pos_correct = text.rfind("Correct")
    pos_incorrect = text.rfind("Incorrect")

    # max() 会返回两者中较大的索引值。如果两个都为-1，也会返回-1。
    last_pos = max(pos_correct, pos_incorrect)

    if last_pos == -1:
        return None
    elif last_pos == pos_correct:
        return "[Correct]"
    else:
        return "[Incorrect]"


if __name__ == "__main__":
    solution_str = "According to this question, the items on the table are as follows:\n- Three black stress balls\n- One orange stress ball\n- Two purple keychains\n- One black dog leash\n- One orange keychain\n- Three orange cat toys\n- One black keychain\n- One purple stress ball\n\nIf we remove all the cat toys from the table, the remaining items are:\n- Three black stress balls\n- One orange stress ball\n- Two purple keychains\n- One black dog leash\n- One orange keychain\n- One black keychain\n- One purple stress ball\n\nNow, let's count the orange items:\n- One orange stress ball\n- One orange keychain\n\nThere are two orange items remaining on the table. So the answer is (C)."
    print(extract_answer(solution_str))