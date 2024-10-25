    try:
        return [char_dict.get(c, '<UNK>') for c in pad(new_smi, max_len)]
    except KeyError as e:
        print(f"KeyError: {e}. The problematic SMILES string is: {smi}")
        raise