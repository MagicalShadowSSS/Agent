"""Equivalent LMFE trie traversal without repeatedly scanning a Unicode alphabet.

The parser, free-text cache, boundary trie and token allowlists are unchanged.
Only the computation of the intersection at a trie node is rearranged.
"""


def optimize_union_alphabet():
    """Skip redundant Unicode set unions; preserve the exact allowed character set."""
    from lmformatenforcer.characterlevelparser import UnionParser
    if getattr(UnionParser, '_sgmea_alphabet_deduplicated', False):
        return
    def allowed(parser):
        strings = list(dict.fromkeys(p.get_allowed_characters() for p in parser.parsers))
        if not strings:
            return ''
        if len(strings) == 1:
            return strings[0]
        return ''.join(set(''.join(strings)))
    UnionParser.get_allowed_characters = allowed
    UnionParser._sgmea_alphabet_deduplicated = True


def optimize_decoder(decoder, boundary_root=None):
    enforcer = decoder.token_enforcer

    def collect(parser, node, allowed_tokens, shortcut_key):
        allowed_tokens.extend(node.tokens)
        freetext = isinstance(shortcut_key, tuple) and shortcut_key[0] == 'json_freetext'
        if freetext:
            _, cur_len, min_len, max_len = shortcut_key
            cache = enforcer.tokenizer_tree.json_freetext_tokens
            min_remaining = min(cache.max_token_len, max(0, min_len-cur_len))
            max_allowed_len = min(cache.max_token_len, max_len-cur_len)
            allowed_tokens.extend(cache.lookup_allowed_tokens(min_remaining, max_allowed_len))
            # The stock collector computes a full Unicode intersection here,
            # then discards every character except the quotation mark.
            characters = ['"'] if '"' in node.children and '"' in parser.get_allowed_characters() else []
        elif node.children:
            legal = parser.get_allowed_characters()
            # Most interior/boundary nodes have just one or two children.
            # String membership is exact and avoids iterating ~30k characters
            # in Python for each such node. At broad nodes retain set intersection.
            characters = ({c for c in node.children if c in legal}
                          if len(node.children) < len(legal)
                          else set(node.children).intersection(legal))
        else:
            characters = ()
        for character in characters:
            collect(parser.add_character(character), node.children[character], allowed_tokens, None)
        if freetext and boundary_root is not None:
            # Keep complete suffix validation for content + quote + punctuation.
            collect(parser, boundary_root, allowed_tokens, None)

    enforcer._collect_allowed_tokens = collect
    return decoder
