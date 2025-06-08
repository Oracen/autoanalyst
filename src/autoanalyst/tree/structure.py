import networkx as nx

from autoanalyst.core.base_classes import BaseEntity

from .node import MetricTreeNode


def _fmt_node_id(entity: str, node_name: str, col: str) -> str:
    """
    Format the node ID for the tree.
    """
    return f"{entity}__{node_name}__{col}"


class MetricTree:
    def __init__(
        self, name: str, entities: list[BaseEntity], nodes: list[MetricTreeNode]
    ):
        """
        Initialize the metric tree with a list of nodes.
        """
        entity_lookup = self._load_entities(entities)
        node_lookup, root_node = self._load_nodes(entity_lookup, nodes)
        tree_topology = self._build_tree(node_lookup, entity_lookup)

        self.name = name
        self.entities = entity_lookup
        self.nodes = node_lookup
        self.root_node = root_node
        self.tree_topology = tree_topology

    @staticmethod
    def _load_entities(entities: list[BaseEntity]) -> dict[str, BaseEntity]:
        """
        Load entities into the metric tree.
        """
        entity_lookup = {}
        for entity in entities:
            if entity.name in entity_lookup:
                raise ValueError(f"Entity with name {entity.name} already exists.")
            entity_lookup[entity.name] = entity

        return entity_lookup

    @staticmethod
    def _load_nodes(
        entities: dict[str, BaseEntity], nodes: list[MetricTreeNode]
    ) -> tuple[dict[str, MetricTreeNode], str]:
        """
        Load nodes into the tree, ensuring that each node is unique and properly linked.
        """
        node_lookup = {}
        root_node = None
        for node in nodes:
            if node.name in nodes:
                raise ValueError(f"Node with name '{node.name}' already exists.")

            # Ensure the node's entity exists in the entity lookup
            if node.entity not in entities:
                raise ValueError(
                    f"Entity '{node.entity}' not found for node '{node.name}'."
                )

            # Set the root node if it has no parent
            if node.parent_node_lookup is None:
                if root_node is not None:
                    raise ValueError(
                        f"Multiple root nodes found - '{root_node}'; '{node.name}'"
                    )
                root_node = node
            node_lookup[node.name] = node

            # Ensure columns exist in entity object
            entity = entities[node.entity]

            missing_cols = entity.check_column_membership(
                [node.head_col] + node.children_cols
            )
            if missing_cols:
                raise ValueError(
                    f"Node '{node.name}' in entity '{node.entity}' references missing "
                    f"columns that don't exist on entity: {missing_cols}."
                )

        if root_node is None:
            raise ValueError("No root node found in the metric tree.")

        return node_lookup, root_node.name

    @staticmethod
    def _build_tree(nodes: dict[str, MetricTreeNode], entities: dict[str, BaseEntity]):
        """
        Covert node into NetworkX tree structure, linking parent nodes and checking
        for toplogical consistency.
        """

        tree: nx.DiGraph = nx.DiGraph()
        # Start by creating all the nodes in the tree
        for node in nodes.values():
            for col in [node.head_col] + node.children_cols:
                node_name = _fmt_node_id(node.entity, node.name, col)

                if node_name in tree:
                    raise ValueError(
                        f"Column '{node_name}' already exists in the tree."
                    )
                tree.add_node(
                    node_name,
                    node_label=f"{node.name}:\n{col}",
                    entity=node.entity,
                    node_name=node.name,
                    column=col,
                )

        # Now link the nodes together
        for node in nodes.values():
            head_name = _fmt_node_id(node.entity, node.name, node.head_col)

            # Connect head nodes to parents
            if node.parent_node_lookup is not None:
                parent_node = nodes[node.parent_node_lookup.node_name]
                parent_name = _fmt_node_id(
                    parent_node.entity,
                    node.parent_node_lookup.node_name,
                    node.parent_node_lookup.child_col,
                )
                if not tree.has_node(parent_name):
                    raise ValueError(f"Parent node '{parent_name}' does not exist.")
                tree.add_edge(parent_name, head_name)

            # Connect children to heads
            for col in node.children_cols:
                child_name = f"{node.entity}__{node.name}__{col}"
                if not tree.has_node(child_name):
                    raise ValueError(f"Child node '{child_name}' does not exist.")
                tree.add_edge(head_name, child_name)

        # Check for tree consistency
        if not nx.is_tree(tree):
            raise ValueError("The constructed graph is not a tree.")

        # Check for connectedness
        if not nx.is_weakly_connected(tree):
            raise ValueError("The constructed tree is not weakly connected.")

        return tree

    def plot_tree(self):
        """
        Plot the metric tree using NetworkX.
        """
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.tree_topology)

        nx.draw(
            self.tree_topology,
            pos,
            with_labels=True,
            labels=nx.get_node_attributes(self.tree_topology, "node_label"),
            node_size=2000,
            node_color="lightblue",
            font_size=10,
            font_color="black",
        )
        plt.title(f"Metric Tree: {self.name}")
        plt.show()

    def to_mermaid(self) -> dict:
        """
        Convert the metric tree to a Mermaid-compatible format.
        Referring to the metric tree structure, we make each column a node,
        each metrictreenode a subgraph, and embed those subgraphs in the entity
        subgraphs.
        """

        mermaid = "graph TD\n"

        return mermaid
