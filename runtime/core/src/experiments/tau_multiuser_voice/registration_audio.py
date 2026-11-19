"""Complete short-segment speech for enrollment; canonical tool IDs stay unchanged."""
import re, hashlib
DIGITS = ('zero','one','two','three','four','five','six','seven','eight','nine')
MAX_WORDS = 30
MAX_CHARACTERS = 190
GAP_MS = 120

def spoken_codes(text):
    return re.sub(r'\bp([0-9]{2})\b', lambda m: 'p ' + ' '.join(DIGITS[int(n)] for n in m.group(1)), text, flags=re.I)

def segments(text):
    words=text.split(); out=[]; offset=0
    while offset<len(words):
        stop=offset
        while stop<len(words) and stop-offset<MAX_WORDS and len(' '.join(words[offset:stop+1]))<=MAX_CHARACTERS:
            stop+=1
        if stop==offset:raise ValueError('registration word exceeds segment limit')
        if stop<len(words):
            boundaries=[i+1 for i in range(offset,stop) if re.search(r'[.!?;]$',words[i])]
            if not boundaries:boundaries=[i+1 for i in range(offset,stop) if re.search(r'[:,]$',words[i])]
            if boundaries:stop=boundaries[-1]
        out.append(' '.join(words[offset:stop]));offset=stop
    if not out:raise ValueError('empty registration speech')
    assert ' '.join(out).split()==words
    return out

def synthesize(tts,text,voice_id):
    output=[];records=[];frames=0;rate=tts.sample_rate
    for index,part in enumerate(segments(text)):
        pcm=tts.synthesize(part,voice_id)
        if not pcm or len(pcm)%2 or not any(pcm):raise ValueError('invalid registration PCM segment')
        count=len(pcm)//2
        records.append({'index':index,'text':part,'start_frame':frames,'end_frame':frames+count,
                        'sample_rate':rate,'pcm_sha256':hashlib.sha256(pcm).hexdigest()})
        output.append(pcm);frames+=count
        if index+1<len(segments(text)):
            gap=rate*GAP_MS//1000;output.append(b'\x00'*(gap*2));frames+=gap
    return b''.join(output),records
