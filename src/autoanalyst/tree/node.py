from dataclasses import dataclass
from typing import Tuple

from autoanalyst.core import base_classes, string_maps


@dataclass
class ParentColLink:
    node_name: str
    child_col: str


class MetricTreeNode:
    def __init__(
        self,
        name: str,
        entity: str,
        transform: base_classes.BaseTransformer,
        head_col: str,
        children_cols: list[str],
        parent_node_lookup: ParentColLink | None = None,
        loader_override: base_classes.BaseLoader | None = None,
    ):
        # Check namespace clashes
        string_maps.iterate_reserved_keywords([head_col] + children_cols)

        self.name = name
        self.entity = entity
        self.head_col = head_col
        self.children_cols = children_cols
        self.transform = transform

        # If the parent node connects, the head_col will be a child of the parent node.
        self.parent_node_lookup = parent_node_lookup

        # If a loader override is provided, it will be used instead of the default loader.
        self.loader_override = loader_override
