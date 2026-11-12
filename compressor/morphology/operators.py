"""
Morphological Operators Module
==============================
Defines the 52 deterministic morphological inflection and derivation operators
used to build the derivation graph over English words.
"""

from typing import List, Set, Tuple

class MorphologicalRules:
    """Collection of deterministic English morphological rules."""
    
    PREFIXES: Tuple[str, ...] = (
        "un", "re", "dis", "in", "de", "over", "mis", "non", "pre", "out",
        "under", "sub", "im", "en", "be"
    )
    
    VOWELS: Set[str] = set("aeiou")

    @classmethod
    def is_consonant_y(cls, w: str) -> bool:
        """Check if word ends in a consonant followed by 'y'."""
        return len(w) >= 2 and w[-1] == "y" and w[-2] not in cls.VOWELS

    @classmethod
    def plural(cls, w: str) -> str:
        """Generate plural form."""
        if not w[0].isupper():
            if w.endswith(("s", "x", "z", "ch", "sh")):
                return w + "es"
            if cls.is_consonant_y(w):
                return w[:-1] + "ies"
        return w + "s"

    @classmethod
    def past_tense(cls, w: str) -> str:
        """Generate -ed form."""
        if w.endswith("e"):
            return w + "d"
        if cls.is_consonant_y(w):
            return w[:-1] + "ied"
        return w + "ed"

    @classmethod
    def progressive(cls, w: str) -> str:
        """Generate -ing form."""
        if w.endswith("e") and not w.endswith(("ee", "ye", "oe")):
            return w[:-1] + "ing"
        return w + "ing"

    @classmethod
    def comparative(cls, w: str) -> str:
        """Generate -er form."""
        if w.endswith("e"):
            return w + "r"
        if cls.is_consonant_y(w):
            return w[:-1] + "ier"
        return w + "er"

    @classmethod
    def superlative(cls, w: str) -> str:
        """Generate -est form."""
        if w.endswith("e"):
            return w + "st"
        if cls.is_consonant_y(w):
            return w[:-1] + "iest"
        return w + "est"

    @classmethod
    def adverb(cls, w: str) -> str:
        """Generate -ly form."""
        if cls.is_consonant_y(w):
            return w[:-1] + "ily"
        if w.endswith("le") and len(w) > 2 and w[-3] not in cls.VOWELS:
            return w[:-1] + "y"
        if w.endswith("ic"):
            return w + "ally"
        return w + "ly"

    @classmethod
    def noun_ness(cls, w: str) -> str:
        return (w[:-1] + "iness") if cls.is_consonant_y(w) else (w + "ness")

    @classmethod
    def noun_ion(cls, w: str) -> str:
        return (w[:-1] + "ion") if w.endswith("e") else (w + "ion")

    @classmethod
    def adj_able(cls, w: str) -> str:
        return (w[:-1] + "able") if (w.endswith("e") and not w.endswith(("ee", "ce", "ge"))) else (w + "able")

    @classmethod
    def noun_ity(cls, w: str) -> str:
        return (w[:-1] + "ity") if (cls.is_consonant_y(w) or w.endswith("e")) else (w + "ity")

    @classmethod
    def verb_ize(cls, w: str) -> str:
        return (w[:-1] + "ize") if w.endswith("e") else (w + "ize")

    @classmethod
    def adj_al(cls, w: str) -> str:
        if w.endswith("e"): return w[:-1] + "al"
        if w.endswith("y"): return w[:-1] + "ial"
        return w + "al"

    @classmethod
    def adj_ic(cls, w: str) -> str:
        return (w[:-1] + "ic") if w.endswith(("e", "y")) else (w + "ic")

    @classmethod
    def adj_y(cls, w: str) -> str:
        return (w[:-1] + "y") if w.endswith("e") else (w + "y")

    @classmethod
    def adj_ive(cls, w: str) -> str:
        return (w[:-1] + "ive") if w.endswith("e") else (w + "ive")

    @classmethod
    def adj_ous(cls, w: str) -> str:
        return (w[:-1] + "ous") if w.endswith("e") else (w + "ous")

    @classmethod
    def apply_rule(cls, op_idx: int, w: str) -> List[str]:
        """Apply the specified rule index (0..51) to a base word."""
        if op_idx >= 37:
            # Standalone prefix rules
            prefix = cls.PREFIXES[op_idx - 37]
            return [prefix + w]
        
        # Suffix and prefixed suffix rules
        p = cls.plural(w)
        ed_w = cls.past_tense(w)
        ing_w = cls.progressive(w)
        er_w = cls.comparative(w)
        
        if op_idx == 0:  return [w + "'s"]
        if op_idx == 1:  return [p]
        if op_idx == 2:  return [ed_w]
        if op_idx == 3:  return [ing_w]
        if op_idx == 4:  return [er_w, er_w + "'s", cls.plural(er_w)]
        if op_idx == 5:  return [er_w]
        if op_idx == 6:  return [cls.superlative(w)]
        if op_idx == 7:  return [cls.adverb(w)]
        if op_idx == 8:
            n = cls.noun_ness(w)
            return [n, n + "'s"]
        if op_idx == 9:
            n = cls.noun_ion(w)
            return [n, n + "'s", cls.plural(n)]
        if op_idx == 10:
            n = w + "ment"
            return [n, n + "'s", cls.plural(n)]
        if op_idx == 11: return [cls.adj_able(w)]
        if op_idx == 12: return ["un" + p]
        if op_idx == 13: return ["un" + ed_w]
        if op_idx == 14: return ["un" + ing_w]
        if op_idx == 15: return ["re" + p]
        if op_idx == 16: return ["re" + ed_w]
        if op_idx == 17: return ["re" + ing_w]
        if op_idx == 18: return ["dis" + p]
        if op_idx == 19:
            n = cls.noun_ity(w)
            return [n, n + "'s", cls.plural(n)]
        if op_idx == 20:
            v = cls.verb_ize(w)
            return [v, cls.plural(v), cls.past_tense(v), cls.progressive(v)]
        if op_idx == 21: return [w + w[-1] + "ed"]
        if op_idx == 22: return [w + w[-1] + "ing"]
        if op_idx == 23:
            d_er = w + w[-1] + "er"
            return [d_er, d_er + "'s", cls.plural(d_er)]
        if op_idx == 24: return [w + w[-1] + "er"]
        if op_idx == 25: return [w + w[-1] + "est"]
        if op_idx == 26: return [w + w[-1] + "y"]
        if op_idx == 27: return [cls.adj_y(w)]
        if op_idx == 28: return [cls.adj_al(w)]
        if op_idx == 29:
            al = cls.adj_al(w)
            return [al, cls.adverb(al)]
        if op_idx == 30: return [cls.adj_ic(w)]
        if op_idx == 31:
            ic = cls.adj_ic(w)
            return [ic, ic + "al", ic + "ally"]
        if op_idx == 32:
            ive = cls.adj_ive(w)
            return [ive, cls.adverb(ive)]
        if op_idx == 33:
            ous = cls.adj_ous(w)
            return [ous, cls.adverb(ous)]
        if op_idx == 34: return [ing_w + "ly"]
        if op_idx == 35: return [cls.adj_able(w)[:-1] + "y"]
        if op_idx == 36: return ["un" + cls.adverb(w)]
        
        return []
