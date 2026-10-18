"""
ajami2boko.py
-------------
Rule-based transliteration of Hausa Ajami (as transcribed in the Ajami
dataset) into Boko, the standard Latin orthography of Hausa.

Ajami Hausa as written in these manuscripts is fully vocalised, so most of
the work is a letter-by-letter mapping:

  consonants  ب b  ت t  ث c  ج j  ح h  خ kh  د d  ذ z  ر r  ز z  س s  ش sh
              ص s  ض d  ط ɗ  ظ z  ڟ ts  ع '  غ g  ڢ/ف f  ڧ/ق ƙ  ک/ك k  ل l
              م m  ࢽ/ن n  ه h  و w  ي y  ݒ b  ݣ g
  vowels      fatha a   kasra i   ࣸ (Ajami u/o) u, and o before و
              dot below ٜ e (written with يٰ / ىٰ after it)   sukun: none
  shadda      doubles the consonant
  long vowels a + ا / ى -> a ; i + ي -> i ; u + و -> u (Boko does not mark length)

Validated on lines of Infiraji 1 that also have a published Latin
transcription (Sobanane/Ajami_Hausa_OCR); see translit_eval.py.
"""
import re
import unicodedata

CONS = {
    "ب": "b", "ت": "t", "ث": "c", "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "z",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "ɗ", "ظ": "z",
    "ڟ": "ts", "ع": "", "غ": "g", "ڢ": "f", "ف": "f", "ڧ": "ƙ", "ق": "ƙ", "ک": "k",
    "ك": "k", "ل": "l", "م": "m", "ࢽ": "n", "ن": "n", "ه": "h", "ة": "a", "و": "w",
    "ي": "y", "ݒ": "b", "ݣ": "g", "ۼ": "g", "ࢳ": "'y", "ࢨ": "'y", "ࢩ": "ƴ",
}
VOWEL = {"\u064E": "a", "\u0650": "i", "\u064F": "u", "\u08F8": "u", "\u065C": "e",
         "\u08F5": "e", "\u08F6": "e", "\u08F9": "e", "\u08F7": "o", "\u08FD": "o",
         "\u08FB": "u", "\u08FC": "o", "\u064B": "an", "\u064D": "in", "\u064C": "un"}
SUKUN, SHADDA, SUPALEF = "\u0652", "\u0651", "\u0670"
ALEFS = set("اأإآٱ")
DROP = set("ـ؞،؟.,()#'") | set("0123456789") | {chr(c) for c in range(0x0660, 0x066A)}


def _word(w, o_before_waw=True):
    out = []
    i, n = 0, len(w)
    last_vowel = None
    while i < n:
        ch = w[i]
        if ch in DROP or unicodedata.category(ch).startswith("L") and "LATIN" in unicodedata.name(ch, ""):
            i += 1
            continue
        if ch in ALEFS or ch == "ء":
            # a vowel carrier: its own mark (if any) gives the vowel; a bare alef
            # after a vowel lengthens it (not written in Boko)
            marks = []
            j = i + 1
            while j < n and unicodedata.category(w[j]).startswith("M"):
                marks.append(w[j]); j += 1
            v = next((VOWEL[m] for m in marks if m in VOWEL), None)
            if v is None and ch == "إ":
                v = "i"
            if v is None and ch in "أآ" and not out:
                v = "a"
            if v is None and ch == "ا" and not out:
                v = "a"
            if v is not None:
                out.append(v); last_vowel = v
            i = j
            continue
        if ch == "ى":                                    # alef maksura: a long a (or i after kasra)
            marks = []
            j = i + 1
            while j < n and unicodedata.category(w[j]).startswith("M"):
                marks.append(w[j]); j += 1
            if SUPALEF in marks and last_vowel == "e":
                pass                                      # ىٰ after a dot-below e: part of the e
            elif last_vowel is None:
                out.append("a")
            i = j
            continue
        if ch in ("و", "ي"):
            marks = []
            j = i + 1
            while j < n and unicodedata.category(w[j]).startswith("M"):
                marks.append(w[j]); j += 1
            own = next((VOWEL[m] for m in marks if m in VOWEL), None)
            long_vowel = ((ch == "ي" and last_vowel in ("i", "e")) or (ch == "و" and last_vowel in ("u", "o")))
            if own is None and SHADDA not in marks and last_vowel is not None and (SUKUN not in marks or long_vowel):
                # a long-vowel letter after a vowel: u/o + و, i/e + ي, or يٰ after e
                if ch == "و" and last_vowel == "u" and o_before_waw:
                    out[-1] = "o"
                i = j
                last_vowel = None
                continue
            c = CONS[ch]
            if SHADDA in marks:
                c = c + c
            out.append(c)
            if own:
                out.append(own)
            last_vowel = own
            i = j
            continue
        if ch in CONS:
            marks = []
            j = i + 1
            while j < n and unicodedata.category(w[j]).startswith("M"):
                marks.append(w[j]); j += 1
            c = CONS[ch]
            if SHADDA in marks and c:
                c = c + c if len(c) == 1 else c + c[-1]
            out.append(c)
            v = next((VOWEL[m] for m in marks if m in VOWEL), None)
            if v is None and SUKUN not in marks and j < n and w[j] in "اى":
                v = "a"                                   # an unmarked consonant before alef: a
            if v:
                out.append(v)
            last_vowel = v
            i = j
            continue
        i += 1                                            # a stray mark
    s = "".join(out)
    s = re.sub(r"ay(?=$|[^aeiou])", "ai", s)              # diphthongs as Boko writes them
    s = re.sub(r"aw(?=$|[^aeiou])", "au", s)
    s = re.sub(r"^al(?=[a-zɗƙ'])", "al-", s) if w.startswith("ال") else s
    return s


def transliterate(text, o_before_waw=True):
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    out = [_word(w, o_before_waw) for w in words]
    return " ".join(o for o in out if o)


def normalise(s):
    """For scoring against a published Latin text: lower case, no length marks,
    no dotted scholarly letters, no apostrophes or hyphens."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M"))
    s = s.replace("ɓ", "b").replace("ʿ", "").replace("ʼ", "").replace("'", "").replace("’", "")
    s = s.replace("-", "").replace("dh", "z").replace("th", "c")
    s = re.sub(r"[^\w\sɗƙƴ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    for a in ["وَاڧَرْ إِࢽْڢِرَاجِ تَڢَرْکࣸو", "مَيْ سࣸوࢽْسَ بَذَيْغَ هَلَّکَابَا", "وَرْ ڟَکٜيٰبا",
              "أَکَ کَرْکَشٜيٰ غَالِبَࢽْسࣸ", "تَثِکَا ثِي ࢽَمَࢽ بࣸشَارَا", "إِࢽْجِ دَاطِي إِࢽْيِ مࣸرْࢽَا"]:
        print(a, "->", transliterate(a))
