import math

class _SimResult(list):
    def flatten(self):
        return self[0]

def cosine_similarity(a, b):
    base=a[0]
    out=[]
    for vec in b:
        num=sum(x*y for x,y in zip(base,vec))
        den=(math.sqrt(sum(x*x for x in base))*math.sqrt(sum(y*y for y in vec)))
        out.append(num/den if den else 0.0)
    return _SimResult([out])
