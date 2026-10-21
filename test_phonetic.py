import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dev_to_lat = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    'ं': 'm', 'ः': 'h', 'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i',
    'उ': 'u', 'ऊ': 'u', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'
}

def to_roman(text):
    return ''.join(dev_to_lat.get(c, '') for c in text)

def lev_dist(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def is_phonetic(orig_devanagari, translated_en):
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', translated_en)]
    if not words:
        return True
    
    # If standard English sentence with multiple distinct common words, it's not phonetic
    common_en = {
        "the", "of", "and", "in", "to", "a", "is", "that", "for", "it", "with", "truth",
        "knowledge", "virtue", "lord", "god", "peace", "holy", "protect", "sacred", "blessed",
        "divine", "verse", "hymn", "earth", "heaven", "all", "there", "no", "gem", "from",
        "by", "going", "reads", "he", "she", "we", "unto", "be", "made", "done", "action"
    }
    english_word_count = sum(1 for w in words if w in common_en)
    if english_word_count >= 2:
        return False
    if len(words) >= 3 and english_word_count >= 1:
        return False

    t_clean = ''.join(words)
    rom = to_roman(orig_devanagari).lower()
    if not rom or not t_clean:
        return False
    
    dist = lev_dist(rom, t_clean)
    max_l = max(len(rom), len(t_clean))
    sim = 1.0 - (dist / max_l) if max_l > 0 else 0
    
    # Check if single word translation is just phonetic
    if sim >= 0.55 or rom in t_clean or t_clean in rom:
        return True
    return False

tests = [
    ('हत्ित', 'Hatit', True),
    ('हत्ित', 'Hatti', True),
    ('गच्छति', 'gachhati', True),
    ('गच्छति', 'going', False),
    ('पठति', 'He reads', False),
    ('देवाय', 'Devaaya', True),
    ('देवाय', 'Unto God / For God', False),
    ('रामेण', 'by Rama', False),
    ('रामेण', 'Ramen', True)
]

for dev, tr, expected in tests:
    res = is_phonetic(dev, tr)
    print(f"{dev} -> '{tr}': is_phonetic={res} (matches expected: {res == expected})")
