from typing import List, Tuple

DATE_COL = "__DATE__"
ID_COL = "__ID__"
RESIDUAL_COL = "__RESIDUAL__"
CONSTANT_COL = "__CONSTANT__"
COHORT_TIME_COL = "__COHORT_TIME__"
ACTIVITY_COUNT_COL = "__ACTIVITY_COUNT__"
ABSOLUTE_TIME_COL = "__ABSOLUTE_TIME__"

LAG_SUFFIX = "__lag"
DIFF_SUFFIX = "__diff"
RESIDUAL_SUFFIX = "__residual"
LEADLAG_SUFFIX = "__leadlag"

RESERVED_KEYWORDS = {
    DATE_COL,
    ID_COL,
    RESIDUAL_COL,
    CONSTANT_COL,
    COHORT_TIME_COL,
    ACTIVITY_COUNT_COL,
    ABSOLUTE_TIME_COL,
}


RESERVED_SUFFIXES = {
    LAG_SUFFIX,
    DIFF_SUFFIX,
    RESIDUAL_SUFFIX,
    LEADLAG_SUFFIX,
}


def check_namespace_clashing(col_name: str) -> Tuple[bool, str, str]:
    """Check if the column name clashes with reserved keywords or suffixes.
    Args:
        col_name (str): The column name to check.
    Returns:
        Tuple[bool, str, str]: A tuple containing a boolean indicating if there is a
        clash, a message describing the clash, and the specific keyword or suffix.
    """
    if col_name in RESERVED_KEYWORDS:
        return True, "Keyword", col_name
    for item in RESERVED_SUFFIXES:
        if col_name.endswith(item):
            return True, "Suffix", item
    return False, "", ""


def iterate_reserved_keywords(names: List[str]):
    failures = []
    for name in names:
        clash, message, item = check_namespace_clashing(name)
        if not clash:
            continue
        msg = f"Column:'{name}' clashes on {message}: '{item}'"
        failures.append(msg)
    if failures:
        raise AttributeError(
            "The following column names clash with reserved keywords or suffixes:\n"
            + "\n".join(failures)
        )


def residual_colname(node_name):
    return f"{node_name}{RESIDUAL_SUFFIX}"


def mermaid_nesting(depth: int, directive: str) -> str:
    """
    Generate a Mermaid subgraph directive with the specified depth.

    Args:
        depth (int): The depth of the nesting.
        directive (str): The Mermaid directive to use (e.g., 'subgraph').

    Returns:
        str: The Mermaid subgraph directive with the specified depth.
    """
    return "    " * depth + f"{directive}"
