"""The module provides functions to list and read text documents from a specified
directory."""

import os

EXAMPLE_TEXT_DIRECTORY = "example_text"


# %%
def get_files_in_documents_directory(
    documents_dir: str = EXAMPLE_TEXT_DIRECTORY,
) -> list[str]:
    """Returns a list of file names in the specified documents directory.

    If the directory does not exist or is not a directory, returns an empty list.

    Args:
        documents_dir (str): The path to the documents directory.
        Defaults to EXAMPLE_TEXT_DIRECTORY.

    Returns:
        list[str]: A list of file names in the documents directory.

    Examples:
        >>> get_files_in_documents_directory('/path/to/example_text_directory')
        ['file1.txt', 'file2.txt']
        >>> get_files_in_documents_directory('/nonexistent/directory')
        []
    """
    # Check if the `documents_dir` directory exists
    if os.path.exists(documents_dir) and os.path.isdir(documents_dir):
        # List all files in the 'documents' directory
        files = [
            f
            for f in os.listdir(documents_dir)
            if os.path.isfile(os.path.join(documents_dir, f))
        ]
        return files
    else:
        return []


# %%
def read_documents_from_files(
    filenames: list[str], directory: str = EXAMPLE_TEXT_DIRECTORY
) -> list[str]:
    """Read documents from specified files in a directory and return their contents as a
    list of strings.

    Args:
        filenames (list[str]): A list of filenames to read.
        directory (str, optional): The directory where the files are located.
        Defaults to EXAMPLE_TEXT_DIRECTORY.

    Returns:
        list[str]: A list containing the contents of each file.

    Examples:
    >>> read_documents_from_files(['file1.txt', 'file2.txt'], '/path/to/documents')
    ['Content of file1.txt', 'Content of file2.txt']
    """
    documents = []
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        with open(file_path, encoding="utf-8") as file:
            documents.append(file.read())
    return documents


# %%
# Read documents and store them in the DOCUMENTS list
files_list: list[str] = get_files_in_documents_directory()
if not files_list:
    raise FileNotFoundError(f"{EXAMPLE_TEXT_DIRECTORY} is empty!")

DOCUMENTS = read_documents_from_files(files_list[:-1])
DOCUMENTS_TO_ADD_TO_INDEX = read_documents_from_files([files_list[-1]])
