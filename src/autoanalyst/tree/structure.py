from collections import defaultdict

import networkx as nx

from autoanalyst.core import base_classes, data_manip, string_maps

from .node import MetricTreeNode


def _fmt_node_id(entity: str, node_name: str, col: str) -> str:
    """
    Format the node ID for the tree.
    """
    return f"{entity}__{node_name}__{col}"


def _mermaid_mapping(
    node_columns: dict[str, set[str]],
    entity_nodes: dict[str, set[str]],
    edge_endpoints: list[tuple[str, str]],
) -> str:
    # Start with the Mermaid graph definition
    mermaid = ["graph TD;"]
    # Create subgraphs for entities
    for entity_idx, (entity, node_names) in enumerate(entity_nodes.items()):
        mermaid.append(string_maps.mermaid_nesting(1, f"subgraph Entity_{entity}"))

        # Create subgraphs for metric tree nodes within this entity
        for node_idx, node_name in enumerate(node_names):
            if node_name in node_columns:
                mermaid.append(
                    string_maps.mermaid_nesting(2, f"subgraph Node_{node_name}")
                )

                # Add all columns for this node
                for col_id in node_columns[node_name]:
                    # Get the column name from the node ID
                    column = col_id.split("__")[-1]
                    # Add the node with a label
                    mermaid.append(
                        string_maps.mermaid_nesting(3, f'{col_id}["{column}"]')
                    )

                mermaid.append(string_maps.mermaid_nesting(2, "end\n"))

        mermaid.append(string_maps.mermaid_nesting(1, "end\n"))

    # Add all edges from the tree topology
    for source, target in edge_endpoints:
        mermaid.append(string_maps.mermaid_nesting(1, f"{source} --> {target}"))

    # Join all lines and return
    return "\n".join(mermaid)


class MetricTree:
    def __init__(
        self,
        name: str,
        entities: list[base_classes.BaseEntity],
        nodes: list[MetricTreeNode],
        storage_module: base_classes.BaseStorageModule,
    ):
        """
        Initialize the metric tree with a list of nodes.
        """
        entity_lookup = self._load_entities(entities)
        node_lookup, root_node = self._load_nodes(entity_lookup, nodes)
        tree_structure, tree_topology = self._build_tree(node_lookup)

        self.name = name
        self.entities = entity_lookup
        self.nodes = node_lookup
        self.root_node = root_node
        self.tree_topology = tree_topology
        self.tree_structure = tree_structure
        self.storage_module = storage_module

    @staticmethod
    def _load_entities(
        entities: list[base_classes.BaseEntity],
    ) -> dict[str, base_classes.BaseEntity]:
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
        entities: dict[str, base_classes.BaseEntity], nodes: list[MetricTreeNode]
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

            missing_cols = entity.check_column_membership(node.get_all_cols())
            if missing_cols:
                raise ValueError(
                    f"Node '{node.name}' in entity '{node.entity}' references missing "
                    f"columns that don't exist on entity: {missing_cols}."
                )

        if root_node is None:
            raise ValueError("No root node found in the metric tree.")

        return node_lookup, root_node.name

    @staticmethod
    def _build_tree(nodes: dict[str, MetricTreeNode]):
        """
        Covert node into NetworkX tree structure, linking parent nodes and checking
        for toplogical consistency.
        """
        structure_tree: nx.DiGraph = nx.DiGraph()
        detail_tree: nx.DiGraph = nx.DiGraph()
        # Start by creating all the nodes in the tree
        for node in nodes.values():
            for col in node.get_all_cols():
                # Format the node ID for the detail tree
                node_name = _fmt_node_id(node.entity, node.name, col)

                if node_name in detail_tree:
                    raise ValueError(
                        f"Column '{node_name}' already exists in the tree."
                    )
                detail_tree.add_node(
                    node_name,
                    node_label=f"{node.name}:\n{col}",
                    entity=node.entity,
                    node_name=node.name,
                    column=col,
                )

            # Append the coarse details to the structure tree, we expect dupes
            structure_tree.add_node(
                node.name,
                entity=node.entity,
                node_name=node.name,
            )

        # Now link the nodes together
        for node in nodes.values():
            head_name = _fmt_node_id(node.entity, node.name, node.head_col)

            # Connect children to heads
            for col in node.children_cols:
                child_name = f"{node.entity}__{node.name}__{col}"
                if not detail_tree.has_node(child_name):
                    raise ValueError(f"Child node '{child_name}' does not exist.")
                detail_tree.add_edge(head_name, child_name)

            # If the node has no parent, it is a root node
            if node.parent_node_lookup is None:
                continue

            # Connect head nodes to parents
            parent_node = nodes[node.parent_node_lookup.node_name]
            parent_name = _fmt_node_id(
                parent_node.entity,
                node.parent_node_lookup.node_name,
                node.parent_node_lookup.child_col,
            )
            if not detail_tree.has_node(parent_name):
                raise ValueError(f"Parent node '{parent_name}' does not exist.")
            detail_tree.add_edge(parent_name, head_name)

            # Add edges to the structure tree
            structure_tree.add_edge(
                parent_node.name,
                node.name,
            )

        # Check for tree consistency
        if not nx.is_tree(structure_tree):
            raise ValueError("The constructed structure graph is not a tree.")

        if not nx.is_tree(detail_tree):
            raise ValueError("The constructed detail graph is not a tree.")

        # Check for connectedness
        if not nx.is_weakly_connected(structure_tree):
            raise ValueError("The constructed structure tree is not connected.")
        if not nx.is_weakly_connected(detail_tree):
            raise ValueError("The constructed detail tree is not weakly connected.")

        return structure_tree, detail_tree

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

    def to_mermaid_string(self) -> str:
        """
        Convert the metric tree to a Mermaid-compatible format.
        Referring to the metric tree structure, we make each column a node,
        each metrictreenode a subgraph, and embed those subgraphs in the entity
        subgraphs.
        """

        # Group nodes by entity and metric tree node
        entity_nodes = defaultdict(set)
        node_columns = defaultdict(set)

        # Collect all nodes and organize them by entity and metric tree node
        for node_id, attrs in self.tree_topology.nodes(data=True):
            entity = attrs.get("entity")
            node_name = attrs.get("node_name")

            # Add node to collections
            entity_nodes[entity].add(node_name)
            node_columns[node_name].add(node_id)

        edge_endpoints = list(self.tree_topology.edges())

        return _mermaid_mapping(node_columns, entity_nodes, edge_endpoints)

    def standardise_units(self):
        """
        Standardise the units of all metrics in the metric tree to the base unit
        defined in the root node of the tree.

        We will dfs iterate through the nodes, loading each dataframe and applying
        the standardisation function defined in the node.
        """

        for node in nx.bfs_tree(self.tree_structure, self.root_node):
            node_obj = self.nodes[node]
            ent_obj = self.entities[node_obj.entity]
            node_cols = node_obj.get_all_cols()

            # Load the initial data
            df_x = ent_obj.load_columns(node_cols)
            df_x = data_manip.standardise_index(df_x, ent_obj.date_col, ent_obj.id_col)
            ser_y = df_x.pop(node_obj.head_col)

            # Apply the transformation to the data
            df_x = node_obj.transform.fit_transform(df_x, ser_y)

            if node_obj.parent_node_lookup is not None:
                # Non-root nodes need to be mapped to their parent node's units
                lookup = node_obj.parent_node_lookup
                parent, target = self.nodes[lookup.node_name], lookup.child_col
                # Because we're doing BFS, we can assume the parent node exists
                # and has been loaded into the storage module
                tgt_col = self.storage_module.load_dataset(parent.name)[target]

                # See if a grain check is valid
                is_same_grain = parent.entity == node_obj.entity

                df_x = node_obj.transform.map_units(df_x, ser_y, tgt_col, is_same_grain)

            df_x[node_obj.head_col] = df_x.sum(axis=1)
            self.storage_module.save_dataset(node_obj.name, df_x)
