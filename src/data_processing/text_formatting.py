import os
import re


def cleanup_whitespaces(text):
    """
    Cleans a text, replacing all newlines with double newline if preceding character is '.', '!', or "?", else replaces
    with a single space character. All sequences of whitespaces are replaced with a space character unless it's a
    newline.

    Parameters
    ----------
    text : str
        String of text to be cleaned.

    Returns
    -------
    str
        The cleaned version of the text.

    """

    # Replace all newlines which does not succeed '\n', '.', '!' or '?', and does not precede another newline, with a
    # space character.
    clean = re.sub(r'(?<![.!?\n])\n(?!\n)', ' ', text)

    # Replace all newlines which succeeds a '.', '!' or '?' and does not precede another newline, with a double newline.
    clean = re.sub(r'(?<=[.!?])\n(?!\n)', '\n\n', clean)

    # Replace all non-newline sequences of whitespace with a single space character.
    clean = re.sub(r'[^\S\n]+', ' ', clean)

    # Remove all non-newline whitespaces succeeding a newline character.
    clean = re.sub(r'(\n[^\S\n]+)', '\n', clean)

    # Remove any whitespace preceding first non-whitespace character of the text.
    clean = clean.strip()

    return clean


def split_text_into_sentences(text: str) -> list[str]:
 pass


# # Step 1: Import the text file into a string
# file_path = '../../dataset/txt/Avinor.txt'  # Path to your text file
# with open(file_path, 'r', encoding='utf-8') as file:
#     file_contents = file.read()  # Reading the content of the file into a string
#
# formatted_text = cleanup_whitespaces(file_contents)
#
# export_path = '../../dataset/cleaned_txt/Avinor_clean.txt'  # Path to the export file
# with open(export_path, 'w', encoding='utf-8') as file:
#     file.write(formatted_text)  # Writing the string to the new file




