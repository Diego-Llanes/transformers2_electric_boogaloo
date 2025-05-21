# NOT YET IMPLEMENTED

if __name__ == '__main__':

    from transformers import BertTokenizer

    # load pretrained BERT-base (uncased) vocab + merges
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = """This is a big string!\n
    lots of text.\n
    pooppoo pee pee.\n
    """

    # 1) get subword tokens
    tokens = tokenizer.tokenize(text)
    # 2) convert to input IDs
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # or do both and add [CLS]/[SEP] in one go:
    ids2 = tokenizer.encode(text, add_special_tokens=True)

    # decode back to string
    recon = tokenizer.decode(ids2, skip_special_tokens=True)

    print(f"{tokens=}\n{ids=}\n{ids2=}\n{recon=}")

