import datasets
import re
import json

math_type_mapping = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}


def load_level5_math_each_category(samples_per_category=20):
    """
    Load level 5 math problems from the competition dataset.
    Returns:
        A list of list of problems. Each list of problems is of the same category.
    """
    seed = 41
    data = datasets.load_dataset("competition_math")
    test_data = data["test"].shuffle(seed=seed)
    sep_cate = []
    for category in math_type_mapping.keys():
        tmp = [
            test_data[x]
            for x in range(len(test_data))
            if test_data[x]["level"] == "Level 5" and test_data[x]["type"] == category
        ]
        if len(tmp) < samples_per_category:
            print(f"Warning: {category} has less than {samples_per_category} problems.")
        sep_cate.append(tmp[:samples_per_category])

    return sep_cate


def remove_asy_sections(input_string):
    """Remove asy sections from the input string.

    Args:
        input_string (str): The input string.
    Returns:
        str: The string without asy sections.
    """
    pattern = r"\[asy\](.*?)\[\\asy\]"
    output_string = re.sub(pattern, "", input_string, flags=re.DOTALL)
    pattern = r"\[asy\](.*?)\[/asy\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    pattern = r"\[ASY\](.*?)\[\\ASY\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    pattern = r"\[ASY\](.*?)\[/ASY\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    return output_string


def write_json(dict_to_save, file):
    """Write a dictionary to a json file.
    Args:

        dict_to_save (dict): The dictionary to save.
        file (str): The file to save to.
    """
    jstring = json.dumps(dict_to_save, indent=2)
    with open(file, "w") as j:
        j.write(jstring)
