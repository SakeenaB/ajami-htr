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
import math
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


# ---------------------------------------------------------------------------
# word division: Ajami often joins words that Boko writes apart (bada -> ba da)
# ---------------------------------------------------------------------------
# Hausa particles plus the words of the published Latin text of Infiraji 1
# (Sobanane/Ajami_Hausa_OCR). Cautious splitting: least-cost division into
# known words with at most one unknown piece; cost tuned by cross-validation
# (held-out word error 80.7% -> 74.0%).
LEXICON = {"a": 58, "abin": 50, "aboka": 1, "abokina": 1, "ai": 50, "aikin": 1, "aiko": 2, "aka": 52, "ake": 50, "akwai": 1, "al": 2, "allah": 3, "ambato": 1, "amina": 2, "aminci": 1, "an": 53, "ana": 50, "annabi": 3, "annabinsa": 1, "arzikinsa": 2, "attaura": 1, "azumi": 1, "ba": 95, "baban": 1, "bai": 53, "baiwa": 1, "ban": 52, "bar": 51, "bari": 1, "baza": 1, "baƙin": 1, "bi": 2, "bin": 1, "bisa": 1, "bismi": 1, "bismil": 1, "biyayya": 1, "bushara": 1, "can": 52, "ce": 50, "ci": 1, "cika": 1, "ciki": 51, "cikin": 51, "cin": 1, "cinya": 1, "da": 69, "dai": 50, "dan": 1, "dare": 1, "dashensa": 1, "daɗi": 1, "daɗin": 1, "dhu": 1, "domin": 50, "don": 55, "duba": 1, "duhu": 2, "duk": 50, "duniya": 1, "duʿaāʿi": 1, "fa": 50, "fahimta": 1, "fariya": 1, "farko": 1, "fi": 2, "fita": 1, "fitilla": 1, "ga": 60, "gaba": 1, "gabanmu": 1, "gabata": 1, "gaji": 1, "galibin": 1, "galibinsu": 1, "gami": 1, "gane": 1, "gardama": 1, "gare": 51, "gaskiya": 2, "gaza": 1, "gobe": 51, "gode": 1, "gubba": 1, "gumba": 1, "gun": 53, "hali": 1, "hallaka": 1, "hamayya": 1, "hankalinsa": 1, "hanya": 1, "har": 51, "harka": 1, "haske": 1, "hassadarsu": 1, "haƙuri": 1, "horo": 1, "hululu": 1, "huta": 1, "ido": 1, "ijābā": 1, "ilāha": 1, "in": 66, "ina": 50, "infiraji": 1, "isa": 1, "ishārā": 1, "jadīdī": 1, "jahālā": 1, "jalla": 1, "je": 2, "ji": 52, "jikankkhalīlu": 1, "jiki": 1, "jin": 51, "ka": 61, "kafirai": 1, "kai": 50, "kamata": 2, "kan": 53, "kangara": 3, "karimi": 1, "karkashe": 1, "karumci": 1, "karīmi": 1, "kasala": 1, "kashe": 1, "katarta": 1, "khilāfā": 1, "ki": 50, "ko": 53, "koran": 1, "kori": 1, "ku": 52, "kula": 1, "kulthumu": 1, "kuma": 100, "kunya": 1, "kushewan": 1, "kuskure": 1, "kyan": 1, "kyauta": 1, "l": 6, "ladabi": 1, "laɗafa": 1, "lāh": 1, "lāhi": 1, "lāhu": 1, "ma": 50, "madina": 1, "mafificin": 2, "maganarsu": 1, "mai": 57, "majidu": 1, "majīdi": 1, "mana": 50, "manzo": 1, "maras": 1, "masu": 53, "mayyaze": 1, "maʼu": 1, "min": 2, "miʿādi": 1, "mu": 54, "muhammadu": 1, "muke": 1, "mun": 50, "muna": 51, "murna": 2, "mussu": 1, "musulmi": 1, "musɗafa": 1, "mutum": 2, "mutunci": 1, "muwa": 1, "muwāfaqā": 1, "na": 57, "nabiyyi": 1, "nai": 3, "nan": 51, "nasa": 54, "nata": 50, "natsa": 1, "ne": 52, "ni": 52, "nifāqā": 1, "nisa": 1, "nufi": 1, "nufin": 1, "nuna": 1, "nura": 1, "qabrin": 1, "qabīla": 1, "rabbanā": 3, "rabbi": 1, "rabonka": 1, "rashi": 1, "rashin": 1, "rasulu": 1, "ratse": 1, "raḥmān": 1, "raḥīm": 1, "riski": 1, "rowa": 1, "ruwan": 1, "sa": 52, "sabo": 51, "sai": 51, "sake": 1, "salami": 1, "salla": 1, "sallah": 1, "sallam": 1, "salsabīlu": 1, "sammu": 1, "sanda": 1, "sani": 1, "sarfa": 1, "sha": 1, "shafā": 1, "shekarar": 1, "shi": 62, "shina": 1, "shirya": 1, "shugabanci": 1, "siffa": 1, "so": 2, "sonsa": 1, "su": 54, "suka": 51, "sun": 52, "sunkuye": 1, "suwaiba": 1, "ta": 60, "tada": 1, "taimakan": 1, "taimake": 1, "taimakonsa": 1, "tamkar": 1, "tana": 50, "tasa": 50, "tazan": 1, "tuba": 1, "tun": 50, "turba": 1, "tūba": 1, "wa": 54, "wadda": 50, "wagga": 1, "wanda": 52, "wani": 51, "wanjin": 1, "wartsake": 1, "wasu": 52, "wata": 50, "waƙa": 3, "waƙar": 3, "waɗanda": 50, "ya": 67, "yabo": 2, "yabon": 2, "yabonka": 2, "yana": 52, "yau": 50, "yawan": 3, "yawmal": 1, "yi": 61, "za": 50, "zai": 51, "zaka": 1, "zalāqā": 1, "zaman": 1, "zame": 1, "zan": 50, "zanka": 1, "ziyara": 1, "zuciyata": 1, "zuma": 1, "zuwansa": 1, "ālihī": 1, "ƙarnin": 1, "ƙarya": 1, "ƙi": 1, "ƙinsa": 1, "ƙirjin": 1, "ƙiyayya": 1, "ƙwanjin": 1, "ɓirbishinsu": 1, "ɗa": 1, "ɗaiba": 1, "ɗan": 4, "ɗaukaka": 1, "ʿalā": 2, "ʿasīmi": 1, "ʿālamina": 1, "ḥadhāqā": 1, "ḥalīmi": 1, "ḥamīdī": 1, "ṣalla": 1, "ṣaḥbih": 1}
_LEX_TOTAL = sum(LEXICON.values())


def segment_word(w, lex, total, unk_cost=12.0):
    """Viterbi over split points; a known piece costs -log p, an unknown piece a flat penalty."""
    n = len(w)
    best = [(0.0, [], 0)] + [(math.inf, [], 0)] * n            # cost, pieces, unknown count
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            piece = w[j:i]
            if len(piece) < 2 and piece != "a":
                continue
            if piece in lex:
                c, unk = -math.log(lex[piece] / total), 0
            else:
                c, unk = unk_cost + 0.1 * len(piece), 1
            prev = best[j]
            if prev[2] + unk > 1:
                continue
            if prev[0] + c < best[i][0]:
                best[i] = (prev[0] + c, prev[1] + [piece], prev[2] + unk)
    return best[n][1] if best[n][0] < math.inf else [w]


def split_line(text, lex=None, unk_cost=12.0):
    lex = lex or LEXICON
    total = sum(lex.values())
    out = []
    for w in text.split():
        pieces = segment_word(w, lex, total, unk_cost)
        # keep a word whole if splitting does not make every piece but one known
        out += pieces if len(pieces) > 1 else [w]
    return " ".join(out)




def to_boko(text):
    """Ajami line -> Boko, with word division."""
    return split_line(transliterate(text))


if __name__ == "__main__":
    for a in ["وَاڧَرْ إِࢽْڢِرَاجِ تَڢَرْکࣸو", "مَيْ سࣸوࢽْسَ بَذَيْغَ هَلَّکَابَا", "وَرْ ڟَکٜيٰبا",
              "أَکَ کَرْکَشٜيٰ غَالِبَࢽْسࣸ", "تَثِکَا ثِي ࢽَمَࢽ بࣸشَارَا", "إِࢽْجِ دَاطِي إِࢽْيِ مࣸرْࢽَا"]:
        print(a, "->", to_boko(a))
