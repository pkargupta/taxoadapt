import re

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()


# create taxonomy from input file
def createGraph(file_addr, label_file='labels.txt'):
    root = None
    # sanity check if file exist
    if not os.path.exists(file_addr):
        print(f"ERROR. Taxonomy file addr {file_addr} not exists.")
        exit(-1)

    id2label = {}
    id2desc = {}
    label2id = {}

    with open(os.path.join(file_addr, label_file)) as f:
        for line in f:
            line_info = line.strip().split('\t')
            # without description
            if len(line_info) == 2:
                label_id, label_name = line_info
            # with description
            if len(line_info) == 3:
                label_id, label_name, label_desc = line_info

            id2label[label_id] = label_name
            id2desc[label_id] = label_desc
            label2id[label_name] = label_id

    # construct graph from file
    with open(os.path.join(file_addr, 'label_hierarchy.txt')) as f:
        ## for each line in the file
        root = Node(-1, 'ROOT')
        for line in f:
            parent_id, child_id = line.strip().split('\t')
            parent = id2label[parent_id]
            child = id2label[child_id]
            parent_desc = id2desc[parent_id] if len(id2desc) > 0 else None
            child_desc = id2desc[child_id] if len(id2desc) > 0 else None

            parent_node = root.findChild(parent_id)
            if parent_node is None:
                parent_node = Node(parent_id, parent, description=parent_desc, level=1)
                root.addChild(parent_node)
                parent_node.addParent(root)

            child_node = root.findChild(child_id)
            if child_node is None:
                child_node = Node(child_id, child, description=child_desc, level=parent_node.level+1)
            parent_node.addChild(child_node)
            child_node.addParent(parent_node)
    
    return root, id2label, label2id

def updateEnrichment(node, phrases, sentences, enrich_type=0):
    if enrich_type == 0: # common-sense
        for phrase in phrases:
            if phrase not in node.common_sense['phrases']:
                node.common_sense['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.common_sense['sentences']:
                node.common_sense['sentences'].append(sent)

    elif enrich_type == 1: # external corpus
        for phrase in phrases:
            if phrase not in node.external['phrases']:
                node.external['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.external['sentences']:
                node.external['sentences'].append(sent)

    else: # user corpus
        for phrase in phrases:
            if phrase not in node.corpus['phrases']:
                node.corpus['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.corpus['sentences']:
                node.corpus['sentences'].append(sent)