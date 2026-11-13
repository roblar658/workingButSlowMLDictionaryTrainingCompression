"""
Morphological Operators Module
==============================
Defines the 52 deterministic morphological inflection and derivation operators
used to build the derivation graph over English words.
"""

from typing import List, Set, Tuple

class MorphologicalRules:
    """Collection of deterministic English morphological rules."""

    # Compact rule bytecode + null-separated suffix/prefix string table shared by C/C++/Python decoders.
    AP_BLOB: bytes = (
        b"\t\x000\x000\x010\x020\x032\x030\x040\x050\x0b\x011\n\x032\t\x042\x060\x00\x13\x0f\x120\x01\x13\x0f\x120\x02\x13\x0f\x120"
        b"\x00\x13\x0f\x130\x01\x13\x0f\x130\x02\x13\x0f\x130\x00\x13\x0f\x140\x0c\x052\n\x06\x10\x13\x00\x10\x01\x10\x020\x0e\x070"
        b"\x0e\x080\x0e\n2\x0e\n0\x0e\x0b0\x0e\x0c0\n\x0c0\x070\x07\x10\x13\x050\x080\x08\x10\x13\t\r\x10\t\x0e0\n\x0f\x10\x13\x050"
        b"\n\x10\x10\x13\x050\x02\x13\t\x110\x06\x13\r\x0c0\x05\x13\x0f\x120's\x00iness\x00ness\x00ion\x00ment\x00ity\x00ize\x00ed"
        b"\x00ing\x00\x00er\x00est\x00y\x00al\x00ally\x00ive\x00ous\x00ly\x00un\x00re\x00dis\x00in\x00de\x00over\x00mis\x00non\x00pre"
        b"\x00out\x00under\x00sub\x00im\x00en\x00be\x00d\x00r\x00st\x00ied\x00ier\x00iest\x00es\x00ies\x00s\x00ily\x00able\x00ial\x00ic\x00"
    )

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
        if not w:
            return []
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
        if 21 <= op_idx <= 26 and not w[-1].isascii():
            return []
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
