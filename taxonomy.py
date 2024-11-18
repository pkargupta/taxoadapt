class Node:
    def __init__(self, id, label, description=None, datasets=None, methodologies=None, evaluation_methods=None, applications=None, children=None, parents=None):
        """
        Initialize a Node based on the provided JSON schema.

        Args:
        label (str): The label for the subtask.
        description (str): Description of the subtask.
        datasets (list of str, optional): A list of dataset ideas for the subtask.
        methodologies (list of str, optional): A list of methodologies for the subtask.
        evaluation_methods (list of str, optional): A list of evaluation methods for the subtask.
        applications (list of str, optional): A list of applications for the subtask.
        children (dict, optional): A dictionary of children nodes, where keys are labels and values are Node instances.
        parents (list of Node, optional): A list of parent nodes of the current node.
        """
        self.id = id
        self.label = label
        self.description = description
        self.datasets = datasets if datasets else []
        self.methodologies = methodologies if methodologies else []
        self.evaluation_methods = evaluation_methods if evaluation_methods else []
        self.applications = applications if applications else []
        self.children = children if children else {}
        self.parents = parents if parents else []
        self.level = 0 if not self.parents else min(parent.level for parent in self.parents) + 1

    def add_child(self, label, child_node):
        """
        Add a child node to the current node.

        Args:
        label (str): The label for the child node.
        child_node (Node): The child Node to be added.
        """
        if child_node in self.parents:
            print("CANNOT ADD! THIS WOULD ADD A CYCLE!")
        else:
            child_node.add_parent(self)
            child_node.level = min(parent.level for parent in child_node.parents) + 1
            self.children[label] = child_node

    def add_parent(self, parent_node):
        """
        Add a parent node to the current node.

        Args:
        parent_node (Node): The parent Node to be added.
        """
        if parent_node not in self.parents:
            self.parents.append(parent_node)
            self.level = min(parent.level for parent in self.parents) + 1

    def get_parents(self):
        """
        Get the parent nodes of the current node.

        Returns:
        list: A list of parent nodes.
        """
        return self.parents
    
    def get_ancestors(self):
        """
        Get all ancestor nodes of the current node.

        Returns:
        list: A list of ancestor nodes from the root to the current node.
        """
        ancestors = set()
        nodes_to_visit = list(self.parents)
        while nodes_to_visit:
            current = nodes_to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                nodes_to_visit.extend(current.parents)
        return list(ancestors)

    def get_children(self):
        """
        Get the children nodes of the current node.

        Returns:
        dict: A dictionary of children nodes where keys are labels and values are Node instances.
        """
        return self.children

    def add_dataset(self, dataset):
        """
        Append a new dataset to the datasets list if it does not already exist.

        Args:
        dataset (str or list of str): The dataset or list of datasets to be added.
        """
        if isinstance(dataset, list):
            self.datasets = list(set(self.datasets).union(dataset))
        else:
            if dataset not in self.datasets:
                self.datasets.append(dataset)

    def add_methodology(self, methodology):
        """
        Append a new methodology to the methodologies list if it does not already exist.

        Args:
        methodology (str or list of str): The methodology or list of methodologies to be added.
        """
        if isinstance(methodology, list):
            self.methodologies = list(set(self.methodologies).union(methodology))
        else:
            if methodology not in self.methodologies:
                self.methodologies.append(methodology)

    def add_evaluation_method(self, evaluation_method):
        """
        Append a new evaluation method to the evaluation_methods list if it does not already exist.

        Args:
        evaluation_method (str or list of str): The evaluation method or list of evaluation methods to be added.
        """
        if isinstance(evaluation_method, list):
            self.evaluation_methods = list(set(self.evaluation_methods).union(evaluation_method))
        else:
            if evaluation_method not in self.evaluation_methods:
                self.evaluation_methods.append(evaluation_method)

    def add_application(self, application):
        """
        Append a new application to the applications list if it does not already exist.

        Args:
        application (str or list of str): The application or list of applications to be added.
        """
        if isinstance(application, list):
            self.applications = list(set(self.applications).union(application))
        else:
            if application not in self.applications:
                self.applications.append(application)

    def display(self, level=0, indent_multiplier=2, visited=None, simple=False):
        """
        Display the node and its children in a structured manner, handling nodes with multiple parents.

        Args:
        level (int): The current level of the node for indentation purposes.
        indent_multiplier (int): The number of spaces used for indentation, multiplied by the level.
        visited (set): A set of visited node IDs to handle cycles in the directed acyclic graph.
        """
        if visited is None:
            visited = set()
        if self.id in visited:
            return
        visited.add(self.id)

        indent = " " * (level * indent_multiplier)
        print(f"{indent}Label: {self.label}")
        print(f"{indent}Description: {self.description}")
        print(f"{indent}Level: {self.level}")
        if not simple:
            if self.datasets:
                print(f"{indent}Datasets: {self.datasets}")
            if self.methodologies:
                print(f"{indent}Methodologies: {self.methodologies}")
            if self.evaluation_methods:
                print(f"{indent}Evaluation Methods: {self.evaluation_methods}")
            if self.applications:
                print(f"{indent}Applications: {self.applications}")
        if self.children:
            print(f"{indent}{'-'*40}")
            print(f"{indent}Children:")
            for child in self.children.values():
                child.display(level + 1, indent_multiplier, visited, simple=simple)
        print(f"{indent}{'-'*40}")

    def __repr__(self):
        return f"Node(label={self.label}, description={self.description}, level={self.level})"