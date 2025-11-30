from .roomformer_v2 import build as build_v2
def build_model(args, train=True, tokenizer=None):
    if not args.poly2seq:
        return build_v2(args, train)
    is_cape = getattr(args, 'cape_mode', False)
    return build_v2(args, train, tokenizer=tokenizer, cape_mode=is_cape)